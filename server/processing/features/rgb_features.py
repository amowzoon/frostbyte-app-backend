"""
rgb_features.py
RGB-derived features for ice detection.

Covers:
  - Multi-scale roughness / smoothness
  - GLCM texture descriptors
  - Structure tensor isotropy
  - Wavelet energy bands
  - Fractal dimension estimate
  - Specularity / shininess
  - Wetness proxies (specular-diffuse ratio, darkening, saturation, reflection coherence)
  - Color distribution
"""

import cv2
import numpy as np
from typing import Optional


# ── optional deps (graceful degradation) ──────────────────────────────────────
try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

try:
    import pywt
    _PYWT = True
except ImportError:
    _PYWT = False


class RGBFeatureExtractor:
    """
    Extract all RGB-based features for a single image region.

    All public methods accept the full image + a boolean mask and return
    a flat dict of float features — no nested structures.
    """

    def __init__(self,
                 glcm_distances: tuple = (1, 3, 5),
                 glcm_angles: tuple = (0, np.pi/4, np.pi/2, 3*np.pi/4),
                 lbp_radius: int = 3,
                 lbp_n_points: int = 24,
                 wavelet: str = 'db2',
                 wavelet_levels: int = 3,
                 specular_percentile: float = 95.0,
                 highlight_dilation: int = 15):

        self.glcm_distances      = list(glcm_distances)
        self.glcm_angles         = list(glcm_angles)
        self.lbp_radius          = lbp_radius
        self.lbp_n_points        = lbp_n_points
        self.wavelet             = wavelet
        self.wavelet_levels      = wavelet_levels
        self.specular_percentile = specular_percentile
        self.highlight_dilation  = highlight_dilation

    # ── public entry point ─────────────────────────────────────────────────────

    def extract(self, rgb_image: np.ndarray, mask: np.ndarray,
                dry_reference: Optional[np.ndarray] = None) -> dict:
        """
        Extract all RGB features for the masked region.

        Args:
            rgb_image:      H×W×3 uint8, RGB channel order
            mask:           H×W boolean
            dry_reference:  Optional H×W×3 uint8 dry-surface reference image
                            for wetness darkening estimate (same location, dry day)
        Returns:
            Flat dict of float features.
        """
        if np.sum(mask) == 0:
            return self._zeros()

        gray        = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        masked_gray = gray[mask]
        masked_rgb  = rgb_image[mask]

        features = {}
        features.update(self._roughness(gray, mask, masked_gray))
        features.update(self._glcm(gray, mask))
        features.update(self._structure_tensor(gray, mask))
        features.update(self._wavelet_energy(gray, mask))
        features.update(self._fractal_dimension(gray, mask))
        features.update(self._specularity(gray, mask, masked_gray))
        features.update(self._wetness(gray, rgb_image, mask, masked_rgb, masked_gray, dry_reference))
        features.update(self._color(masked_rgb))
        features.update(self._edges(gray, mask))
        return features

    # ── roughness / smoothness ─────────────────────────────────────────────────

    def _roughness(self, gray: np.ndarray, mask: np.ndarray,
                   masked_gray: np.ndarray) -> dict:
        out = {}

        # Multi-scale Laplacian variance — captures roughness at different scales
        for sigma in (1, 2, 4, 8):
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            lap     = cv2.Laplacian(blurred, cv2.CV_64F)
            out[f'roughness_lap_s{sigma}'] = float(np.var(lap[mask]))

        # Local standard deviation (mean of per-pixel std in 7×7 window)
        gray_f  = gray.astype(np.float32)
        mean_sq = cv2.blur(gray_f ** 2, (7, 7))
        sq_mean = cv2.blur(gray_f,      (7, 7)) ** 2
        local_std = np.sqrt(np.maximum(mean_sq - sq_mean, 0))
        out['roughness_local_std_mean'] = float(np.mean(local_std[mask]))
        out['roughness_local_std_var']  = float(np.var(local_std[mask]))

        # Intensity variance within region
        out['roughness_intensity_var'] = float(np.var(masked_gray))

        # LBP energy (if skimage available)
        if _SKIMAGE:
            lbp = local_binary_pattern(gray, self.lbp_n_points, self.lbp_radius, method='uniform')
            out['roughness_lbp_energy'] = float(np.mean(lbp[mask] ** 2))
            # Uniformity: fraction of uniform patterns (smooth surfaces have more)
            n_uniform = np.sum(lbp[mask] <= self.lbp_n_points + 1)
            out['roughness_lbp_uniformity'] = float(n_uniform / np.sum(mask))

        return out

    # ── GLCM texture ───────────────────────────────────────────────────────────

    def _glcm(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        if not _SKIMAGE:
            return {}

        # Crop to bounding box of mask to speed up GLCM
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return {}
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        patch = gray[r0:r1, c0:c1]

        # Quantise to 64 levels for speed
        patch_q = (patch // 4).astype(np.uint8)

        try:
            glcm = graycomatrix(patch_q, distances=self.glcm_distances,
                                angles=self.glcm_angles, levels=64,
                                symmetric=True, normed=True)
        except Exception:
            return {}

        out = {}
        for prop in ('contrast', 'dissimilarity', 'homogeneity', 'energy',
                     'correlation', 'ASM'):
            try:
                vals = graycoprops(glcm, prop)   # shape: (n_dist, n_angle)
                out[f'glcm_{prop}_mean'] = float(vals.mean())
                out[f'glcm_{prop}_max']  = float(vals.max())
            except Exception:
                pass

        return out

    # ── structure tensor ───────────────────────────────────────────────────────

    def _structure_tensor(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        """
        Structure tensor isotropy:
          ratio of minor to major eigenvalue of the gradient covariance.
          1.0 = perfectly isotropic (ice-like), 0.0 = strongly directional.
        """
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        Ixx = cv2.GaussianBlur((gx * gx).astype(np.float32), (5, 5), 1.0)
        Iyy = cv2.GaussianBlur((gy * gy).astype(np.float32), (5, 5), 1.0)
        Ixy = cv2.GaussianBlur((gx * gy).astype(np.float32), (5, 5), 1.0)

        Jxx = float(np.mean(Ixx[mask]))
        Jyy = float(np.mean(Iyy[mask]))
        Jxy = float(np.mean(Ixy[mask]))

        # Eigenvalues of 2×2 tensor
        trace = Jxx + Jyy
        det   = Jxx * Jyy - Jxy ** 2
        disc  = max(0.0, (trace / 2) ** 2 - det)
        lam1  = trace / 2 + np.sqrt(disc)
        lam2  = trace / 2 - np.sqrt(disc)

        isotropy = float(lam2 / lam1) if lam1 > 1e-8 else 1.0  # ~1 = isotropic
        coherence = float((lam1 - lam2) / (lam1 + lam2 + 1e-8))  # ~1 = oriented

        return {
            'structure_isotropy':  isotropy,
            'structure_coherence': coherence,
            'structure_energy':    float(lam1 + lam2),
        }

    # ── wavelet energy ─────────────────────────────────────────────────────────

    def _wavelet_energy(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        if not _PYWT:
            return {}

        out  = {}
        data = gray.astype(np.float32)

        try:
            coeffs = pywt.wavedec2(data, self.wavelet, level=self.wavelet_levels)
            # coeffs[0] = approx; coeffs[1..] = (cH, cV, cD) per level
            total_energy = sum(np.sum(c ** 2) for c in
                               [coeffs[0]] + [item for subband in coeffs[1:] for item in subband])
            total_energy = max(total_energy, 1e-8)

            for lvl, subband in enumerate(coeffs[1:], start=1):
                cH, cV, cD = subband
                e_h = float(np.sum(cH ** 2) / total_energy)
                e_v = float(np.sum(cV ** 2) / total_energy)
                e_d = float(np.sum(cD ** 2) / total_energy)
                out[f'wavelet_energy_H_l{lvl}'] = e_h
                out[f'wavelet_energy_V_l{lvl}'] = e_v
                out[f'wavelet_energy_D_l{lvl}'] = e_d
                out[f'wavelet_energy_l{lvl}']   = e_h + e_v + e_d

            # High-frequency fraction (fine texture)
            hf_energy = sum(
                float(np.sum(item ** 2))
                for subband in coeffs[1:2]   # level-1 = finest
                for item in subband
            )
            out['wavelet_hf_fraction'] = float(hf_energy / total_energy)

        except Exception:
            pass

        return out

    # ── fractal dimension ──────────────────────────────────────────────────────

    def _fractal_dimension(self, gray: np.ndarray, mask: np.ndarray,
                           min_box: int = 2, max_box: int = 64) -> dict:
        """
        Box-counting fractal dimension on the masked region.
        Ice surfaces tend toward lower fractal dimension (smoother).
        """
        rows, cols = np.where(mask)
        if len(rows) < 16:
            return {'fractal_dimension': 0.0}

        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        patch  = gray[r0:r1, c0:c1].astype(np.float32)
        patch_mask = mask[r0:r1, c0:c1]

        # Threshold to binary using Otsu within patch
        patch_valid = patch.copy()
        patch_valid[~patch_mask] = 0
        _, binary = cv2.threshold(
            patch_valid.astype(np.uint8), 0, 1,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        sizes, counts = [], []
        box = min_box
        while box <= min(max_box, r1 - r0, c1 - c0):
            # Count non-empty boxes of size box×box
            n = 0
            for i in range(0, binary.shape[0], box):
                for j in range(0, binary.shape[1], box):
                    if binary[i:i+box, j:j+box].any():
                        n += 1
            if n > 0:
                sizes.append(box)
                counts.append(n)
            box *= 2

        if len(sizes) < 2:
            return {'fractal_dimension': 0.0}

        log_s = np.log(sizes)
        log_c = np.log(counts)
        # Fractal dim ≈ -slope of log(count) vs log(box_size)
        slope = float(np.polyfit(log_s, log_c, 1)[0])
        return {'fractal_dimension': float(-slope)}

    # ── specularity ────────────────────────────────────────────────────────────

    def _specularity(self, gray: np.ndarray, mask: np.ndarray,
                     masked_gray: np.ndarray) -> dict:
        out = {}

        if len(masked_gray) == 0:
            return {'shininess_ratio': 0.0, 'specular_lobe_peak': 0.0,
                    'specular_lobe_width': 0.0, 'highlight_density': 0.0,
                    'nearby_highlights': 0.0}

        mean_intensity = float(np.mean(masked_gray))

        # Threshold-based highlight fraction
        threshold = np.percentile(masked_gray, self.specular_percentile)
        bright    = masked_gray > threshold
        out['shininess_ratio'] = float(np.sum(bright) / len(masked_gray))

        # Specular lobe: fit Gaussian to the upper tail of intensity distribution
        hist, bin_edges = np.histogram(masked_gray, bins=64, range=(0, 255))
        hist_norm = hist / (hist.sum() + 1e-8)
        peak_bin  = int(np.argmax(hist_norm[32:]) + 32)  # look only in upper half
        peak_val  = float(bin_edges[peak_bin])
        # Width: std of pixels above the median
        above_med = masked_gray[masked_gray > np.median(masked_gray)]
        out['specular_lobe_peak']  = peak_val / 255.0
        out['specular_lobe_width'] = float(np.std(above_med)) / 255.0 if len(above_med) > 1 else 0.0

        # Specular-to-diffuse ratio (highlight intensity / mean)
        out['shininess_ratio']    = float(np.mean(masked_gray[bright]) / (mean_intensity + 1e-8)) \
                                    if bright.any() else 1.0

        # Spatial density of distinct highlight blobs
        highlight_map = ((gray > threshold) & mask).astype(np.uint8)
        n_blobs, _, stats, _ = cv2.connectedComponentsWithStats(highlight_map)
        n_blobs = max(0, n_blobs - 1)  # subtract background
        region_area = float(np.sum(mask))
        out['highlight_density'] = float(n_blobs / (region_area + 1e-8) * 1000)

        # Highlights in the immediate surrounding area
        kernel     = np.ones((self.highlight_dilation, self.highlight_dilation), np.uint8)
        dilated    = cv2.dilate(mask.astype(np.uint8), kernel)
        surround   = dilated.astype(bool) & ~mask
        if surround.any():
            thr_global = np.percentile(gray, self.specular_percentile)
            out['nearby_highlights'] = float(np.sum(gray[surround] > thr_global) / np.sum(surround))
        else:
            out['nearby_highlights'] = 0.0

        return out

    # ── wetness ────────────────────────────────────────────────────────────────

    def _wetness(self, gray: np.ndarray, rgb: np.ndarray,
                 mask: np.ndarray, masked_rgb: np.ndarray,
                 masked_gray: np.ndarray,
                 dry_reference: Optional[np.ndarray]) -> dict:
        out = {}

        # Saturation in HSV — wet surfaces have higher saturation
        hsv          = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        masked_sat   = hsv[:, :, 1][mask]
        out['wetness_saturation_mean'] = float(np.mean(masked_sat) / 255.0)
        out['wetness_saturation_std']  = float(np.std(masked_sat)  / 255.0)

        # Specular-to-diffuse ratio using channel difference
        # Wet surfaces: strong specular (high V) with low diffuse variance
        masked_v = hsv[:, :, 2][mask]
        diffuse  = float(np.median(masked_gray))
        specular = float(np.percentile(masked_gray, 95))
        out['wetness_spec_diffuse_ratio'] = float(specular / (diffuse + 1e-8))

        # Reflection coherence: spatial autocorrelation of the highlight mask
        thr           = float(np.percentile(masked_gray, self.specular_percentile))
        highlight_bin = ((gray > thr) & mask).astype(np.float32)
        if highlight_bin.any():
            # Normalised cross-correlation with a 1-pixel shifted version
            shifted   = np.roll(highlight_bin, 1, axis=1)
            numerator = float(np.sum(highlight_bin * shifted))
            denom     = float(np.sum(highlight_bin) + 1e-8)
            out['wetness_reflection_coherence'] = numerator / denom
        else:
            out['wetness_reflection_coherence'] = 0.0

        # Darkening relative to dry reference (if available)
        if dry_reference is not None:
            dry_gray        = cv2.cvtColor(dry_reference, cv2.COLOR_RGB2GRAY)
            mean_current    = float(np.mean(masked_gray))
            mean_dry        = float(np.mean(dry_gray[mask]))
            out['wetness_darkening'] = float((mean_dry - mean_current) / (mean_dry + 1e-8))
        else:
            out['wetness_darkening'] = 0.0   # unknown without reference

        # Variance in V channel — wet surfaces tend to have high V variance (shiny patches)
        out['wetness_v_variance'] = float(np.var(masked_v) / (255.0 ** 2))

        return out

    # ── color distribution ─────────────────────────────────────────────────────

    def _color(self, masked_rgb: np.ndarray) -> dict:
        if len(masked_rgb) == 0:
            return {k: 0.0 for k in (
                'color_mean_r', 'color_mean_g', 'color_mean_b',
                'color_std_r',  'color_std_g',  'color_std_b',
                'color_bluish_score', 'color_whitish_score',
                'color_brightness_mean', 'color_brightness_std',
            )}

        mean = np.mean(masked_rgb, axis=0) / 255.0
        std  = np.std(masked_rgb,  axis=0) / 255.0

        brightness = np.mean(masked_rgb, axis=1) / 255.0

        # Bluish: B channel dominates over average of R and G
        bluish = float(np.mean(
            masked_rgb[:, 2] > (masked_rgb[:, 0].astype(float) + masked_rgb[:, 1].astype(float)) / 2
        ))

        # Whitish: high brightness, low inter-channel std
        inter_ch_std = np.std(masked_rgb.astype(float), axis=1) / 255.0
        whitish      = float(np.mean((brightness > 0.75) & (inter_ch_std < 0.08)))

        return {
            'color_mean_r': float(mean[0]),
            'color_mean_g': float(mean[1]),
            'color_mean_b': float(mean[2]),
            'color_std_r':  float(std[0]),
            'color_std_g':  float(std[1]),
            'color_std_b':  float(std[2]),
            'color_bluish_score':     bluish,
            'color_whitish_score':    whitish,
            'color_brightness_mean':  float(np.mean(brightness)),
            'color_brightness_std':   float(np.std(brightness)),
        }

    # ── edges / boundaries ─────────────────────────────────────────────────────

    def _edges(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges[mask] > 0) / (np.sum(mask) + 1e-8))

        # Boundary sharpness: gradient magnitude at region boundary pixels
        eroded   = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        boundary = mask.astype(np.uint8) - eroded
        gx       = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy       = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        boundary_sharpness = float(np.mean(grad_mag[boundary > 0])) \
                             if boundary.any() else 0.0

        return {
            'edge_density':         edge_density,
            'boundary_sharpness':   boundary_sharpness / 255.0,
        }

    # ── zero-fill fallback ─────────────────────────────────────────────────────

    def _zeros(self) -> dict:
        """Return a zeroed feature dict for empty regions."""
        dummy_img  = np.zeros((8, 8, 3), dtype=np.uint8)
        dummy_mask = np.ones((8, 8), dtype=bool)
        return {k: 0.0 for k in self.extract(dummy_img, dummy_mask)}
