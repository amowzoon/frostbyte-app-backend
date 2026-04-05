"""
heuristic_model.py
Rule-Based Ice Detection with Tunable Weights.

Key change over original:
  - Hard temperature gate: surfaces above HARD_GATE_TEMP cannot be ice regardless
    of other features. Soft damping applied in the melt zone.
  - Feature keys updated to match the new extractor naming convention.
  - PresetWeights and explain/visualize API unchanged.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ── Hard physical gates ────────────────────────────────────────────────────────
# These override the weighted score entirely. No amount of shininess or texture
# should make a 25°C surface count as ice.
HARD_GATE_TEMP_MAX   = 10.0   # °C  — above this: impossible to be ice, score → 0
MELTING_ICE_TEMP_MAX =  4.0   # °C  — above this: apply damping (melting ice edge case)


@dataclass
class HeuristicWeights:
    """Tunable weights. Positive = evidence of ice. Negative = evidence against."""

    # ── Temperature (ir_ prefix matches new IRFeatureExtractor keys) ──────────
    ir_mean_temp_weight:              float = -0.15   # colder = more ice
    ir_near_freezing_weight:          float =  0.80   # % of pixels near 0°C
    ir_at_or_below_freezing_weight:   float =  0.60   # % of pixels ≤ 0°C
    ir_in_danger_zone_weight:         float =  0.40   # % in -5..+3°C
    ir_diff_from_air_mean_weight:     float = -0.05   # colder than air = more ice
    ir_colder_than_air_fraction_weight: float = 0.30  # fraction significantly colder
    ir_temp_var_weight:               float = -0.10   # low variance = uniform ice layer
    ir_gradient_mean_weight:          float = -0.20   # low gradient = smooth ice
    ir_gradient_smooth_fraction_weight: float = 0.25  # fraction with ~zero gradient
    ir_dist_from_dew_point_weight:    float = -0.15   # near/below dew point = frost risk
    ir_below_dew_point_fraction_weight: float = 0.35
    ir_emissivity_proxy_weight:       float =  0.10   # high emissivity ≈ wet/icy

    # ── Thermal texture ────────────────────────────────────────────────────────
    ir_glcm_homogeneity_weight:       float =  0.40   # high homogeneity = uniform ice
    ir_glcm_contrast_weight:          float = -0.20   # low contrast = smooth surface
    ir_lbp_uniformity_weight:         float =  0.30

    # ── Temporal temperature ───────────────────────────────────────────────────
    medium_temp_rate_weight:          float = -1.50   # cooling rate (negative = cooling)
    medium_temp_accel_weight:         float = -0.80
    in_cooling_plateau_weight:        float =  1.20   # plateau at 0°C = phase change!
    temp_trajectory_score_weight:     float =  1.00

    # ── Roughness / texture (rgb_ / roughness_ prefix) ────────────────────────
    roughness_lap_s1_weight:          float = -0.20   # smooth = ice
    roughness_lap_s4_weight:          float = -0.15   # coarser scale smoothness
    roughness_lbp_uniformity_weight:  float =  0.30   # uniform LBP = smooth surface
    glcm_homogeneity_mean_weight:     float =  0.40
    glcm_energy_mean_weight:          float =  0.30
    glcm_contrast_mean_weight:        float = -0.25
    structure_isotropy_weight:        float =  0.30   # isotropic = ice-like
    wavelet_hf_fraction_weight:       float = -0.25   # low HF energy = smooth

    # ── Specularity ────────────────────────────────────────────────────────────
    shininess_ratio_weight:           float =  0.60
    highlight_density_weight:         float =  0.30
    nearby_highlights_weight:         float =  0.40
    specular_lobe_peak_weight:        float =  0.25

    # ── Wetness ────────────────────────────────────────────────────────────────
    wetness_saturation_mean_weight:   float =  0.20
    wetness_spec_diffuse_ratio_weight: float = 0.35
    wetness_reflection_coherence_weight: float = 0.25
    wetness_darkening_weight:         float =  0.40   # darkened vs dry reference
    wetness_v_variance_weight:        float =  0.20

    # ── Temporal texture ───────────────────────────────────────────────────────
    texture_stability_1min_weight:    float =  0.50
    highlight_persistence_1min_weight: float = 0.40

    # ── Color ──────────────────────────────────────────────────────────────────
    color_bluish_score_weight:        float =  0.30
    color_whitish_score_weight:       float =  0.40
    color_brightness_mean_weight:     float =  0.10   # bright = more reflective

    # ── Spatial ────────────────────────────────────────────────────────────────
    spatial_area_fraction_weight:     float =  0.05   # larger = more confident
    spatial_compactness_weight:       float =  0.10
    spatial_boundary_smoothness_weight: float = 0.15  # smooth boundary = solid patch

    # ── Thresholds ─────────────────────────────────────────────────────────────
    ice_threshold:             float = 0.50
    uncertain_threshold_low:   float = 0.35
    uncertain_threshold_high:  float = 0.65

    # ── Temperature normalisation range ───────────────────────────────────────
    temp_range_min: float = -10.0
    temp_range_max: float =  30.0

    def to_dict(self)  -> dict:  return {k: v for k, v in self.__dict__.items()}
    @classmethod
    def from_dict(cls, d: dict) -> 'HeuristicWeights': return cls(**d)

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f'✓ Saved heuristic weights to {path}')

    @classmethod
    def load(cls, path: Path) -> 'HeuristicWeights':
        with open(path) as f:
            d = json.load(f)
        print(f'✓ Loaded heuristic weights from {path}')
        return cls.from_dict(d)


class HeuristicIceDetector:

    def __init__(self, weights: Optional[HeuristicWeights] = None):
        self.weights = weights or HeuristicWeights()

    # ── main predict ───────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        n   = len(features_df)
        raw = np.zeros(n)
        contributions: Dict[str, np.ndarray] = {}

        for idx, (_, row) in enumerate(features_df.iterrows()):

            # ── HARD GATE — physically impossible ─────────────────────────────
            mean_temp = row.get('ir_mean_temp', None)
            if mean_temp is not None and mean_temp > HARD_GATE_TEMP_MAX:
                raw[idx] = -999.0   # will map to ~0 after sigmoid
                continue

            score = 0.0
            row_contributions = {}

            # ── damping factor for the melting-ice zone ────────────────────────
            if mean_temp is not None and mean_temp > MELTING_ICE_TEMP_MAX:
                dampen = 1.0 - (mean_temp - MELTING_ICE_TEMP_MAX) / \
                               (HARD_GATE_TEMP_MAX - MELTING_ICE_TEMP_MAX)
            else:
                dampen = 1.0

            # ── score each feature ─────────────────────────────────────────────
            weights_dict = self.weights.to_dict()
            for feat_key, weight_key in self._feature_weight_pairs():
                if feat_key not in row or weight_key not in weights_dict:
                    continue
                val    = row[feat_key]
                weight = weights_dict[weight_key]

                # per-feature normalisation
                norm_val = self._normalise(feat_key, val)
                contrib  = float(norm_val * weight * dampen)
                score   += contrib
                row_contributions[feat_key] = contrib

            raw[idx] = score
            for k, v in row_contributions.items():
                contributions.setdefault(k, np.zeros(n))[idx] = v

        confidence = self._sigmoid(raw)
        # Hard-gated rows stay at 0
        confidence[raw <= -900] = 0.0

        return confidence, {
            'raw_scores':          raw,
            'feature_contributions': contributions,
            'top_features':        self._top_features(contributions),
        }

    def predict_binary(self, features_df: pd.DataFrame) -> np.ndarray:
        conf, _ = self.predict(features_df)
        return (conf > self.weights.ice_threshold).astype(int)

    def get_uncertain_regions(self, features_df: pd.DataFrame) -> np.ndarray:
        conf, _ = self.predict(features_df)
        return ((conf > self.weights.uncertain_threshold_low) &
                (conf < self.weights.uncertain_threshold_high)).astype(np.uint8)

    def explain_prediction(self, features_df: pd.DataFrame, region_idx: int = 0) -> str:
        conf, dbg = self.predict(features_df)
        rc  = conf[region_idx]
        rs  = dbg['raw_scores'][region_idx]

        contribs = {k: v[region_idx] for k, v in dbg['feature_contributions'].items()}
        top      = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)

        lines = [
            f'Region {region_idx} — Ice probability: {rc:.1%}',
            f'Raw score: {rs:.3f}',
            'Verdict: ' + (
                'LIKELY ICE ❄️'            if rc > self.weights.uncertain_threshold_high else
                'LIKELY NOT ICE ✓'         if rc < self.weights.uncertain_threshold_low  else
                'UNCERTAIN ⚠️  needs review'
            ),
            '',
            'Top contributing features:',
        ]
        for i, (feat, contrib) in enumerate(top[:12], 1):
            lines.append(f'  {i:2d}. {feat:45s} {contrib:+.3f}  '
                         f'{"→ ICE" if contrib > 0 else "→ NO ICE"}')
        return '\n'.join(lines)

    def visualize_weights(self) -> str:
        categories = {
            'Temperature':          [k for k in self.weights.to_dict() if k.startswith('ir_')
                                     and 'glcm' not in k and 'lbp' not in k
                                     and k.endswith('_weight')],
            'Thermal Texture':      [k for k in self.weights.to_dict() if 'ir_glcm' in k
                                     or 'ir_lbp' in k],
            'Temporal Temperature': ['medium_temp_rate_weight', 'medium_temp_accel_weight',
                                     'in_cooling_plateau_weight', 'temp_trajectory_score_weight'],
            'Roughness / Texture':  [k for k in self.weights.to_dict()
                                     if any(p in k for p in ('roughness_', 'glcm_', 'structure_',
                                                              'wavelet_'))
                                     and k.endswith('_weight')],
            'Specularity':          [k for k in self.weights.to_dict()
                                     if any(p in k for p in ('shininess', 'highlight', 'specular'))
                                     and k.endswith('_weight')],
            'Wetness':              [k for k in self.weights.to_dict()
                                     if 'wetness' in k and k.endswith('_weight')],
            'Temporal Texture':     ['texture_stability_1min_weight',
                                     'highlight_persistence_1min_weight'],
            'Color':                [k for k in self.weights.to_dict()
                                     if k.startswith('color_') and k.endswith('_weight')],
            'Spatial':              [k for k in self.weights.to_dict()
                                     if k.startswith('spatial_') and k.endswith('_weight')],
        }

        lines = ['=' * 65, 'HEURISTIC ICE DETECTOR WEIGHTS', '=' * 65]
        for cat, keys in categories.items():
            lines.append(f'\n{cat}:')
            for k in keys:
                w = getattr(self.weights, k, None)
                if w is None:
                    continue
                lines.append(f'  {k:50s}: {w:+.3f}  '
                              f'{"→ ICE" if w > 0 else "→ NO ICE"}')

        lines += [
            '',
            f'Hard gate:        surface temp > {HARD_GATE_TEMP_MAX}°C → impossible',
            f'Melt zone damping: {MELTING_ICE_TEMP_MAX}°C..{HARD_GATE_TEMP_MAX}°C → score dampened',
            f'Ice threshold:    {self.weights.ice_threshold}',
            f'Uncertain range:  {self.weights.uncertain_threshold_low}'
            f' – {self.weights.uncertain_threshold_high}',
            '=' * 65,
        ]
        return '\n'.join(lines)

    # ── internals ─────────────────────────────────────────────────────────────

    def _feature_weight_pairs(self) -> List[Tuple[str, str]]:
        """Map every feature key to its weight key."""
        w = self.weights.to_dict()
        pairs = []
        for weight_key in w:
            if not weight_key.endswith('_weight'):
                continue
            # Convention: feature key = weight key without trailing _weight
            feat_key = weight_key[:-len('_weight')]
            pairs.append((feat_key, weight_key))
        return pairs

    def _normalise(self, feat_key: str, val: float) -> float:
        """Per-feature normalisation so weights operate on a comparable scale."""
        # Temperature: normalise to [-1, 1]
        if feat_key == 'ir_mean_temp':
            return 2 * (val - self.weights.temp_range_min) / \
                   (self.weights.temp_range_max - self.weights.temp_range_min) - 1

        # Counts / areas: clip to [0, 1]
        if feat_key in ('spatial_area', 'spatial_area_fraction'):
            return float(np.clip(val / 100_000.0 if feat_key == 'spatial_area' else val, 0, 1))

        # Roughness values can be large — normalise
        if 'roughness_lap' in feat_key:
            return float(np.clip(val / 500.0, 0, 1))

        if 'boundary_sharpness' in feat_key or feat_key == 'ir_gradient_mean':
            return float(np.clip(val / 10.0, 0, 1))

        if 'ir_mean_dist_from_freeze' in feat_key:
            return float(np.clip(val / 15.0, 0, 1))

        if 'temp_rate' in feat_key:
            return float(np.clip(val / 0.01, -1, 1))

        if 'temp_accel' in feat_key:
            return float(np.clip(val / 0.001, -1, 1))

        if 'ir_diff_from_air' in feat_key:
            return float(np.clip(val / 20.0, -1, 1))

        # Wavelet / GLCM values typically in [0,1] already — pass through
        # Most fraction/ratio features are already [0,1]
        return float(np.clip(val, -1.0, 1.0))

    @staticmethod
    def _sigmoid(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-scale * x))

    def _top_features(self, contributions: Dict[str, np.ndarray],
                      top_n: int = 5) -> List[Tuple[str, float]]:
        avg = {k: float(np.mean(np.abs(v))) for k, v in contributions.items()}
        return sorted(avg.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ── Preset configurations ──────────────────────────────────────────────────────

class PresetWeights:

    @staticmethod
    def conservative() -> HeuristicWeights:
        w = HeuristicWeights()
        w.ice_threshold             = 0.70
        w.ir_near_freezing_weight   = 1.20
        w.in_cooling_plateau_weight = 1.50
        w.medium_temp_rate_weight   = -2.00
        return w

    @staticmethod
    def aggressive() -> HeuristicWeights:
        w = HeuristicWeights()
        w.ice_threshold           = 0.30
        w.ir_mean_temp_weight     = -0.25
        w.shininess_ratio_weight  = 0.90
        w.medium_temp_rate_weight = -1.00
        return w

    @staticmethod
    def texture_focused() -> HeuristicWeights:
        """Use when thermal data is unreliable."""
        w = HeuristicWeights()
        w.shininess_ratio_weight        = 1.00
        w.nearby_highlights_weight      = 0.80
        w.spatial_boundary_smoothness_weight = 0.60
        w.roughness_lap_s1_weight       = -0.50
        w.ir_mean_temp_weight           = -0.02
        return w

    @staticmethod
    def temperature_focused() -> HeuristicWeights:
        """Use when visual features are unreliable (nighttime, overcast)."""
        w = HeuristicWeights()
        w.ir_mean_temp_weight             = -0.30
        w.ir_near_freezing_weight         = 1.20
        w.medium_temp_rate_weight         = -2.50
        w.in_cooling_plateau_weight       = 2.00
        w.shininess_ratio_weight          = 0.05
        w.wetness_spec_diffuse_ratio_weight = 0.05
        return w


# ── example ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    detector = HeuristicIceDetector()
    print(detector.visualize_weights())

    df = pd.DataFrame([
        {   # Cold, near freezing, shiny — should be high ice probability
            'ir_mean_temp': 1.5, 'ir_near_freezing': 0.8,
            'ir_diff_from_air_mean': -10.0, 'medium_temp_rate': -0.008,
            'in_cooling_plateau': 0.9, 'shininess_ratio': 0.3,
            'roughness_lap_s1': 150.0, 'spatial_area_fraction': 0.05,
        },
        {   # Warm surface — hard gate should zero this out
            'ir_mean_temp': 25.0, 'ir_near_freezing': 0.0,
            'shininess_ratio': 0.8,   # doesn't matter — too warm
            'roughness_lap_s1': 50.0,
        },
        {   # Dry road, near-zero but no other ice signals
            'ir_mean_temp': 2.0, 'ir_near_freezing': 0.2,
            'ir_diff_from_air_mean': 0.0, 'medium_temp_rate': 0.001,
            'in_cooling_plateau': 0.0, 'shininess_ratio': 0.05,
            'roughness_lap_s1': 400.0, 'spatial_area_fraction': 0.10,
        },
    ])

    conf, dbg = detector.predict(df)
    for i, c in enumerate(conf):
        print(f'Region {i}: {c:.1%} ice probability')
    print()
    print(detector.explain_prediction(df, region_idx=0))
