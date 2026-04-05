from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np


@dataclass
class TemporalConfig:
    fps: float = 1.0
    short_term_window: int = 1
    medium_term_window: int = 1
    long_term_window: int = 1


class TemporalFeatureBank:
    """Maintains temporal history for each region."""

    def __init__(self, config: TemporalConfig):
        self.config = config
        self.previous_frame_regions  = None
        self.previous_frame_features = {}
        self.previous_timestamp      = None
        self.frame_history = deque(maxlen=int(config.long_term_window * config.fps))

    def update_history(self, timestamp: float, regions: np.ndarray,
                       features_dict: Dict[int, dict]):
        self.frame_history.append((timestamp, regions.copy(), features_dict.copy()))
        self.previous_frame_regions  = regions
        self.previous_frame_features = features_dict
        self.previous_timestamp      = timestamp

    def get_temporal_features(self, region_id: int, current_region_mask: np.ndarray,
                              current_time: float, current_features: dict) -> dict:
        if len(self.frame_history) < 2:
            return self._get_defaults()

        short_term  = self._filter_by_window(current_time, self.config.short_term_window)
        medium_term = self._filter_by_window(current_time, self.config.medium_term_window)

        short_series  = self._extract_weighted_history(current_region_mask, short_term,  current_features)
        medium_series = self._extract_weighted_history(current_region_mask, medium_term, current_features)

        out = {}

        if len(short_series) >= 2:
            out.update(self._compute_rate_features(short_series, 'short'))
            out['texture_stability_1min']     = self._compute_stability(short_series, 'roughness')
            out['highlight_persistence_1min'] = self._compute_stability(short_series, 'shininess_ratio')
        else:
            out.update({'short_temp_rate': 0.0, 'short_temp_accel': 0.0,
                        'texture_stability_1min': 0.0, 'highlight_persistence_1min': 0.0})

        if len(medium_series) >= 5:
            out.update(self._compute_rate_features(medium_series, 'medium'))
            out['in_cooling_plateau']    = self._detect_plateau(medium_series)
            out['temp_trajectory_score'] = self._compute_trajectory_score(medium_series)
        else:
            out.update({'medium_temp_rate': 0.0, 'medium_temp_accel': 0.0,
                        'in_cooling_plateau': 0.0, 'temp_trajectory_score': 0.0})

        out['time_since_last_ice'] = 1e6
        return out

    # ── internals ──────────────────────────────────────────────────────────────

    def _filter_by_window(self, current_time: float, window_seconds: int) -> List:
        cutoff = current_time - window_seconds
        return [(ts, r, f) for ts, r, f in self.frame_history if ts >= cutoff]

    def _extract_weighted_history(self, current_mask, history_window, current_features):
        series = []
        for timestamp, hist_regions, hist_features in history_window:
            overlaps = self._compute_overlaps(current_mask, hist_regions)
            if overlaps:
                series.append((timestamp, self._weight_features(overlaps, hist_features)))
        return series

    def _compute_overlaps(self, current_mask, historical_regions):
        current_area = np.sum(current_mask)
        if current_area == 0:
            return []
        overlaps = []
        for hid in np.unique(historical_regions):
            if hid == 0:
                continue
            intersection = np.sum(current_mask & (historical_regions == hid))
            if intersection > 0:
                overlaps.append((hid, intersection / current_area))
        return overlaps

    def _weight_features(self, overlaps, historical_features):
        total = sum(w for _, w in overlaps)
        normed = [(r, w / total) for r, w in overlaps]
        first_id = normed[0][0]
        if first_id not in historical_features:
            return {}
        keys = [k for k in historical_features[first_id]
                if k not in ('region_id', 'timestamp', 'persistent_region_id')]
        return {
            k: sum(w * historical_features[r].get(k, 0) for r, w in normed)
            for k in keys
        }

    def _compute_rate_features(self, series, prefix):
        times = [t for t, _ in series]
        temps = [f.get('mean_temp', 0) for _, f in series]
        dt = times[-1] - times[0]
        rate = (temps[-1] - temps[0]) / dt if dt else 0.0
        if len(temps) >= 3:
            mid = len(temps) // 2
            dt1 = times[mid] - times[0]
            dt2 = times[-1] - times[mid]
            r1  = (temps[mid] - temps[0])  / dt1 if dt1 else 0.0
            r2  = (temps[-1]  - temps[mid]) / dt2 if dt2 else 0.0
            accel = r2 - r1
        else:
            accel = 0.0
        return {f'{prefix}_temp_rate': rate, f'{prefix}_temp_accel': accel}

    def _compute_stability(self, series, key):
        vals = [f.get(key, 0) for _, f in series]
        return 1.0 / (1.0 + np.std(vals)) if len(vals) >= 2 else 0.0

    def _detect_plateau(self, series, threshold=0.5, temp_range=(-2.0, 2.0)):
        temps = [f.get('mean_temp', 0) for _, f in series]
        if len(temps) < 5:
            return 0.0
        near_zero  = np.mean([temp_range[0] <= t <= temp_range[1] for t in temps])
        is_plateau = np.mean(np.abs(np.diff(temps)) < threshold)
        return float(near_zero * is_plateau)

    def _compute_trajectory_score(self, series):
        temps = [f.get('mean_temp', 0) for _, f in series]
        if len(temps) < 5:
            return 0.0
        trend        = temps[-1] - temps[0]
        near_freezing = np.mean([abs(t) < 5.0 for t in temps])
        if trend < 0 and temps[-1] < 5.0:
            return float(near_freezing * min(1.0, abs(trend) / 10.0))
        if trend > 0 and temps[0] < 5.0:
            return float(near_freezing * 0.5)
        return 0.0

    def _get_defaults(self):
        return {
            'short_temp_rate': 0.0, 'short_temp_accel': 0.0,
            'medium_temp_rate': 0.0, 'medium_temp_accel': 0.0,
            'texture_stability_1min': 0.0, 'highlight_persistence_1min': 0.0,
            'in_cooling_plateau': 0.0, 'temp_trajectory_score': 0.0,
            'time_since_last_ice': 1e6,
        }
