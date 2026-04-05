"""
ir_features.py
IR / thermal camera features for ice detection.

The IR image passed in is always in Celsius (float32, warped to RGB resolution).
"""

import cv2
import numpy as np
from typing import Optional

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


class IRFeatureExtractor:

    def __init__(self, lbp_radius: int = 2, lbp_n_points: int = 16):
        self.lbp_radius   = lbp_radius
        self.lbp_n_points = lbp_n_points

    def extract(self, ir_celsius: np.ndarray, mask: np.ndarray,
                air_temp: Optional[float] = None,
                dew_point: Optional[float] = None,
                surrounding_mask: Optional[np.ndarray] = None) -> dict:
        """
        Extract all IR features for the masked region.

        Args:
            ir_celsius:       H×W float32, temperature in Celsius
            mask:             H×W boolean (should cover same region as RGB mask)
            air_temp:         Ambient air temperature (°C), optional
            dew_point:        Dew point (°C), optional — icing risk when surface ≈ dew_point
            surrounding_mask: Optional mask for the area surrounding the region,
                              used for thermal contrast features.
        Returns:
            Flat dict of float features.
        """
        if np.sum(mask) == 0:
            return self._zeros()

        temps = ir_celsius[mask].astype(np.float64)

        out = {}
        out.update(self._basic_stats(temps))
        out.update(self._freezing_proximity(temps))
        out.update(self._air_relative(temps, air_temp, dew_point))
        out.update(self._spatial_gradient(ir_celsius, mask))
        out.update(self._thermal_texture(ir_celsius, mask))
        out.update(self._surrounding_contrast(ir_celsius, mask, surrounding_mask, air_temp))
        return out

    # ── basic statistics ───────────────────────────────────────────────────────

    def _basic_stats(self, temps: np.ndarray) -> dict:
        return {
            'ir_mean_temp':   float(np.mean(temps)),
            'ir_median_temp': float(np.median(temps)),
            'ir_min_temp':    float(np.min(temps)),
            'ir_max_temp':    float(np.max(temps)),
            'ir_temp_range':  float(np.max(temps) - np.min(temps)),
            'ir_temp_std':    float(np.std(temps)),
            'ir_temp_var':    float(np.var(temps)),
            # Skewness: negative skew = left tail = cold spots in a warm region
            'ir_temp_skew':   float(self._skewness(temps)),
            'ir_temp_kurt':   float(self._kurtosis(temps)),
        }

    # ── freezing proximity ─────────────────────────────────────────────────────

    def _freezing_proximity(self, temps: np.ndarray) -> dict:
        return {
            # Fraction of pixels within ±2°C of freezing
            'ir_near_freezing':         float(np.mean(np.abs(temps) <= 2.0)),
            # Fraction at or below freezing
            'ir_at_or_below_freezing':  float(np.mean(temps <= 0.0)),
            # Fraction in the dangerous melt/refreeze zone (-5 to +3°C)
            'ir_in_danger_zone':        float(np.mean((temps >= -5.0) & (temps <= 3.0))),
            # Cold-pixel count ratio (below -1°C)
            'ir_cold_pixel_ratio':      float(np.mean(temps < -1.0)),
            # Distance of mean temp from freezing
            'ir_mean_dist_from_freeze': float(abs(np.mean(temps))),
        }

    # ── air-temperature-relative features ─────────────────────────────────────

    def _air_relative(self, temps: np.ndarray,
                      air_temp: Optional[float],
                      dew_point: Optional[float]) -> dict:
        out = {}

        if air_temp is not None:
            diff = temps - air_temp
            out['ir_diff_from_air_mean'] = float(np.mean(diff))
            out['ir_diff_from_air_min']  = float(np.min(diff))
            # Surface significantly colder than air → radiative cooling → icing risk
            out['ir_colder_than_air_fraction'] = float(np.mean(diff < -1.0))
        else:
            out.update({
                'ir_diff_from_air_mean': 0.0,
                'ir_diff_from_air_min':  0.0,
                'ir_colder_than_air_fraction': 0.0,
            })

        if dew_point is not None:
            # Surface near or below dew point → condensation / frost formation
            dp_diff = temps - dew_point
            out['ir_dist_from_dew_point']          = float(np.mean(dp_diff))
            out['ir_below_dew_point_fraction']      = float(np.mean(dp_diff <= 0.0))
        else:
            out['ir_dist_from_dew_point']      = 0.0
            out['ir_below_dew_point_fraction'] = 0.0

        return out

    # ── spatial gradient ───────────────────────────────────────────────────────

    def _spatial_gradient(self, ir: np.ndarray, mask: np.ndarray) -> dict:
        """
        Ice has very uniform temperature → low spatial gradient.
        Water pooling has higher gradient from evaporative cooling at edges.
        """
        ir_f = ir.astype(np.float32)
        gx   = cv2.Sobel(ir_f, cv2.CV_64F, 1, 0, ksize=3)
        gy   = cv2.Sobel(ir_f, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)

        masked_grad = grad[mask]
        return {
            'ir_gradient_mean': float(np.mean(masked_grad)),
            'ir_gradient_std':  float(np.std(masked_grad)),
            'ir_gradient_max':  float(np.max(masked_grad)),
            # Smoothness proxy: fraction of pixels with near-zero gradient
            'ir_gradient_smooth_fraction': float(np.mean(masked_grad < 0.5)),
        }

    # ── thermal texture ────────────────────────────────────────────────────────

    def _thermal_texture(self, ir: np.ndarray, mask: np.ndarray) -> dict:
        """
        Run GLCM and LBP on the IR image within the region.
        Ice → spatially uniform IR (low contrast, high homogeneity).
        """
        out = {}

        # Quantise to 0-255 uint8 for texture analysis
        # Clip to a realistic outdoor temp range first
        ir_clipped = np.clip(ir, -20.0, 50.0)
        ir_uint8   = ((ir_clipped + 20.0) / 70.0 * 255).astype(np.uint8)

        if _SKIMAGE:
            # LBP uniformity within region
            lbp = local_binary_pattern(ir_uint8, self.lbp_n_points,
                                       self.lbp_radius, method='uniform')
            n_uniform = np.sum(lbp[mask] <= self.lbp_n_points + 1)
            out['ir_lbp_uniformity'] = float(n_uniform / (np.sum(mask) + 1e-8))
            out['ir_lbp_energy']     = float(np.mean(lbp[mask] ** 2))

            # GLCM on bounding-box crop
            rows, cols = np.where(mask)
            if len(rows) > 0:
                r0, r1 = rows.min(), rows.max() + 1
                c0, c1 = cols.min(), cols.max() + 1
                patch  = (ir_uint8[r0:r1, c0:c1] // 8).astype(np.uint8)  # 32 levels
                try:
                    glcm = graycomatrix(patch, distances=[1, 3], angles=[0, np.pi/2],
                                        levels=32, symmetric=True, normed=True)
                    for prop in ('contrast', 'homogeneity', 'energy', 'correlation'):
                        vals = graycoprops(glcm, prop)
                        out[f'ir_glcm_{prop}'] = float(vals.mean())
                except Exception:
                    pass

        # Multi-scale Laplacian variance (thermal roughness)
        for sigma in (1, 3):
            blurred = cv2.GaussianBlur(ir_uint8, (0, 0), sigma)
            blurred = blurred.astype(np.float32)
            lap     = cv2.Laplacian(blurred, cv2.CV_32F)
            out[f'ir_thermal_roughness_s{sigma}'] = float(np.var(lap[mask]))

        return out

    # ── surrounding thermal contrast ───────────────────────────────────────────

    def _surrounding_contrast(self, ir: np.ndarray, mask: np.ndarray,
                               surrounding_mask: Optional[np.ndarray],
                               air_temp: Optional[float]) -> dict:
        """
        Compare region temperature to its surroundings.
        Relative cooling is more meaningful than absolute temperature.
        """
        region_mean = float(np.mean(ir[mask]))
        out = {}

        if surrounding_mask is not None and surrounding_mask.any():
            surr_mean = float(np.mean(ir[surrounding_mask]))
            out['ir_contrast_with_surround'] = region_mean - surr_mean
        else:
            # Fall back to 15px dilation to estimate surroundings
            kernel  = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            surround = dilated & ~mask
            if surround.any():
                out['ir_contrast_with_surround'] = region_mean - float(np.mean(ir[surround]))
            else:
                out['ir_contrast_with_surround'] = 0.0

        # Emissivity proxy:
        # Wet asphalt ε ≈ 0.96, dry asphalt ε ≈ 0.90 → apparent temp shift ~1-2°C
        # We approximate: high-emissivity surfaces appear slightly warmer than
        # their actual temp relative to surroundings.
        if air_temp is not None:
            # Normalised apparent emissivity proxy (0 = same as air, 1 = much warmer)
            out['ir_emissivity_proxy'] = float(
                np.clip((region_mean - air_temp + 5.0) / 10.0, 0.0, 1.0)
            )
        else:
            out['ir_emissivity_proxy'] = 0.0

        return out

    # ── stat helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        mu  = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return float(np.mean(((x - mu) / std) ** 3))

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        mu  = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return float(np.mean(((x - mu) / std) ** 4) - 3.0)

    def _zeros(self) -> dict:
        dummy_ir   = np.zeros((8, 8), dtype=np.float32)
        dummy_mask = np.ones((8, 8), dtype=bool)
        return {k: 0.0 for k in self.extract(dummy_ir, dummy_mask)}
