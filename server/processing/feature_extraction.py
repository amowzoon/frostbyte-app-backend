"""
feature_extraction.py
Main FeatureExtractor — orchestrates all per-sensor extractors and the
temporal feature bank. Returns a pd.DataFrame of features per region.
"""

import time
from typing import Optional, Dict
import numpy as np
import pandas as pd

from features import (
    RGBFeatureExtractor,
    IRFeatureExtractor,
    RadarFeatureExtractor,
    SpatialFeatureExtractor,
    TemporalConfig,
    TemporalFeatureBank,
)


class FeatureExtractor:
    """
    Extract all features from a single frame.

    Usage:
        extractor = FeatureExtractor()
        df = extractor.extract_features(rgb, ir, radar, regions, air_temp=8.0)

    The returned DataFrame has one row per non-background region.
    """

    def __init__(self,
                 temporal_config: Optional[TemporalConfig] = None,
                 dry_reference: Optional[np.ndarray] = None):
        """
        Args:
            temporal_config:  Window sizes for temporal feature bank.
            dry_reference:    Optional H×W×3 uint8 RGB image of the same scene
                              under dry conditions — used to estimate wetness darkening.
                              Can be set later via set_dry_reference().
        """
        self.temporal_config = temporal_config or TemporalConfig()
        self.temporal_bank   = TemporalFeatureBank(self.temporal_config)
        self.dry_reference   = dry_reference

        self._rgb_ext    = RGBFeatureExtractor()
        self._ir_ext     = IRFeatureExtractor()
        self._radar_ext  = RadarFeatureExtractor()
        self._spatial_ext = SpatialFeatureExtractor()

    def set_dry_reference(self, dry_rgb: np.ndarray):
        """Set / update the dry-surface reference image for wetness detection."""
        self.dry_reference = dry_rgb

    # ── main entry point ───────────────────────────────────────────────────────

    def extract_features(self,
                         RGB_image: np.ndarray,
                         IR_image: Optional[np.ndarray],
                         radar_scan: Optional[np.ndarray],
                         regions: np.ndarray,
                         air_temp: Optional[float] = None,
                         dew_point: Optional[float] = None,
                         timestamp: Optional[float] = None) -> pd.DataFrame:
        """
        Extract all features for every region in the frame.

        Args:
            RGB_image:   H×W×3 uint8, RGB order
            IR_image:    H×W float32 Celsius, warped to RGB resolution (or None)
            radar_scan:  Radar data in capture_api format (or None)
            regions:     H×W int32 segmentation labels (0 = background)
            air_temp:    Ambient air temperature in °C (optional but recommended)
            dew_point:   Dew point in °C (optional)
            timestamp:   Unix timestamp — defaults to now

        Returns:
            pd.DataFrame with one row per region, all features as float columns.
        """
        if timestamp is None:
            timestamp = time.time()

        region_ids = np.unique(regions)
        region_ids = region_ids[region_ids != 0]

        if len(region_ids) == 0:
            return pd.DataFrame()

        rows:              list      = []
        features_by_region: Dict[int, dict] = {}

        for rid in region_ids:
            mask = (regions == rid)
            feat = {'region_id': int(rid), 'timestamp': timestamp}

            # ── RGB ──────────────────────────────────────────────────────────
            feat.update(self._rgb_ext.extract(
                RGB_image, mask, dry_reference=self.dry_reference
            ))

            # ── IR ───────────────────────────────────────────────────────────
            if IR_image is not None:
                # Build surrounding mask (15px dilation minus the region itself)
                import cv2
                kernel   = np.ones((15, 15), np.uint8)
                dilated  = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
                surround = dilated & ~mask

                feat.update(self._ir_ext.extract(
                    IR_image, mask,
                    air_temp=air_temp,
                    dew_point=dew_point,
                    surrounding_mask=surround if surround.any() else None,
                ))
            else:
                feat.update(self._ir_ext._zeros())

            # ── Radar ────────────────────────────────────────────────────────
            feat.update(self._radar_ext.extract(radar_scan, mask))

            # ── Spatial ──────────────────────────────────────────────────────
            feat.update(self._spatial_ext.extract(mask, image_shape=RGB_image.shape[:2]))

            # ── Temporal ─────────────────────────────────────────────────────
            feat.update(self.temporal_bank.get_temporal_features(
                rid, mask, timestamp, feat
            ))

            rows.append(feat)
            features_by_region[rid] = feat

        # Update temporal bank AFTER all regions are processed
        self.temporal_bank.update_history(timestamp, regions, features_by_region)

        return pd.DataFrame(rows)

    # ── convenience ───────────────────────────────────────────────────────────

    def feature_names(self) -> list:
        """Return the list of feature column names (excluding region_id/timestamp)."""
        dummy_rgb     = np.zeros((64, 64, 3), dtype=np.uint8)
        dummy_ir      = np.zeros((64, 64), dtype=np.float32)
        dummy_regions = np.ones((64, 64),  dtype=np.int32)
        df = self.extract_features(dummy_rgb, dummy_ir, None, dummy_regions)
        return [c for c in df.columns if c not in ('region_id', 'timestamp')]

    def reset_temporal(self):
        """Clear temporal history (e.g. when switching locations)."""
        self.temporal_bank = TemporalFeatureBank(self.temporal_config)


# ── example usage ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    extractor = FeatureExtractor()

    for frame_idx in range(5):
        rgb      = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ir       = np.random.randn(480, 640).astype(np.float32) * 5 + 10
        radar    = None
        regions  = np.random.randint(0, 6, (480, 640), dtype=np.int32)
        air_temp = 8.0

        df = extractor.extract_features(
            RGB_image=rgb, IR_image=ir, radar_scan=radar,
            regions=regions, air_temp=air_temp,
        )
        print(f'Frame {frame_idx}: {len(df)} regions, {len(df.columns)} features')

    print('\nAll feature names:')
    for name in extractor.feature_names():
        print(f'  {name}')
