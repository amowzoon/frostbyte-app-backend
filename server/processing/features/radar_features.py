"""
radar_features.py
mmWave radar features for ice detection.

Currently a structured placeholder — implement once radar data format is finalised.
The return dict keys are fixed so downstream code won't break when this is populated.
"""

import numpy as np
from typing import Optional


class RadarFeatureExtractor:
    """
    Extracts mmWave radar features per region.

    Expected radar_scan format (TBD — update __init__ docstring when confirmed):
      - Option A: H×W float32 reflection intensity map (same resolution as RGB)
      - Option B: dict with 'range_doppler', 'azimuth', 'intensity' arrays
      - Option C: list of point cloud (x, y, z, intensity) detections

    The public extract() method already returns the full feature dict with zeros
    so the rest of the pipeline doesn't need to branch on radar availability.
    """

    # Feature keys this extractor will eventually produce
    FEATURE_KEYS = (
        # Intensity features
        'radar_intensity_mean',
        'radar_intensity_std',
        'radar_intensity_max',
        'radar_intensity_snr',          # signal-to-noise ratio
        # Surface properties
        'radar_penetration_depth',       # estimated from multi-frequency response
        'radar_backscatter_coeff',       # normalised radar cross-section
        'radar_specular_fraction',       # fraction of specular (smooth) returns
        # Doppler / motion
        'radar_doppler_mean',            # mean radial velocity (water flow vs static ice)
        'radar_doppler_std',
        # Spatial
        'radar_coverage_fraction',       # fraction of mask pixels with valid radar return
        'radar_return_coherence',        # spatial coherence of returns (high = smooth surface)
    )

    def __init__(self, intensity_threshold: float = 0.1):
        self.intensity_threshold = intensity_threshold

    def extract(self, radar_scan: Optional[np.ndarray],
                mask: np.ndarray) -> dict:
        """
        Extract radar features for the masked region.

        Args:
            radar_scan:  Radar data in whatever format capture_api provides.
                         Pass None if radar not available.
            mask:        H×W boolean region mask (RGB resolution)
        Returns:
            Flat dict of float features (zeros if radar_scan is None).
        """
        if radar_scan is None or np.sum(mask) == 0:
            return self._zeros()

        # TODO: implement once radar data format is confirmed
        # Stub returns zeros with the correct keys so schema is stable
        return self._zeros()

    # ── future implementation hooks ────────────────────────────────────────────

    def _extract_intensity(self, radar_map: np.ndarray, mask: np.ndarray) -> dict:
        """Extract intensity statistics from a 2D radar intensity map."""
        vals = radar_map[mask]
        valid = vals[vals > self.intensity_threshold]
        if len(valid) == 0:
            return {k: 0.0 for k in ('radar_intensity_mean', 'radar_intensity_std',
                                      'radar_intensity_max', 'radar_intensity_snr')}
        noise = float(np.median(vals))
        return {
            'radar_intensity_mean': float(np.mean(valid)),
            'radar_intensity_std':  float(np.std(valid)),
            'radar_intensity_max':  float(np.max(valid)),
            'radar_intensity_snr':  float(np.mean(valid) / (noise + 1e-8)),
        }

    def _extract_doppler(self, doppler_map: np.ndarray, mask: np.ndarray) -> dict:
        """Extract Doppler velocity features — water flows, ice is static."""
        vals = doppler_map[mask]
        return {
            'radar_doppler_mean': float(np.mean(vals)),
            'radar_doppler_std':  float(np.std(vals)),
        }

    def _extract_coherence(self, intensity_map: np.ndarray, mask: np.ndarray) -> dict:
        """Spatial coherence of radar returns."""
        import cv2
        im = intensity_map.astype(np.float32)
        shifted = np.roll(im, 1, axis=1)
        num   = float(np.sum((im * shifted)[mask]))
        denom = float(np.sum((im ** 2)[mask]) + 1e-8)
        return {'radar_return_coherence': num / denom}

    def _zeros(self) -> dict:
        return {k: 0.0 for k in self.FEATURE_KEYS}
