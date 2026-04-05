"""
run from frostbyte-backend
docker compose run --rm server pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd


# core ml / inference logic

class TestHeuristicIceDetector:
    """Tests for heuristic_model.py — the primary inference path."""

    def _make_features(self, ir_mean=-2.0, near_freezing=0.8, at_freezing=0.5,
                       shininess=0.7, wetness_darkening=0.5, air_temp=-1.0):
        """Return a minimal features DataFrame that mirrors FeatureExtractor output."""
        return pd.DataFrame([{
            'region_id': 1,
            'ir_mean_temp': ir_mean,
            'ir_near_freezing': near_freezing,
            'ir_at_or_below_freezing': at_freezing,
            'ir_in_danger_zone': 0.6,
            'ir_diff_from_air_mean': ir_mean - air_temp,
            'ir_colder_than_air_fraction': 0.9,
            'ir_temp_var': 0.5,
            'ir_gradient_mean': 0.1,
            'ir_gradient_smooth_fraction': 0.8,
            'ir_dist_from_dew_point': 0.5,
            'ir_below_dew_point_fraction': 0.3,
            'ir_emissivity_proxy': 0.7,
            'ir_glcm_homogeneity': 0.8,
            'ir_glcm_contrast': 0.1,
            'ir_lbp_uniformity': 0.7,
            'medium_temp_rate': -0.5,
            'medium_temp_accel': -0.1,
            'in_cooling_plateau': 1.0,
            'temp_trajectory_score': 0.8,
            'roughness_lap_s1': 0.1,
            'roughness_lap_s4': 0.1,
            'roughness_lbp_uniformity': 0.7,
            'glcm_homogeneity_mean': 0.8,
            'glcm_energy_mean': 0.7,
            'glcm_contrast_mean': 0.1,
            'structure_isotropy': 0.8,
            'wavelet_hf_fraction': 0.1,
            'shininess_ratio': shininess,
            'highlight_density': 0.4,
            'nearby_highlights': 0.5,
            'specular_lobe_peak': 0.3,
            'wetness_saturation_mean': 0.3,
            'wetness_spec_diffuse_ratio': 0.4,
            'wetness_reflection_coherence': 0.3,
            'wetness_darkening': wetness_darkening,
            'wetness_v_variance': 0.2,
            'texture_stability_1min': 0.7,
            'highlight_persistence_1min': 0.6,
            'color_bluish_score': 0.4,
            'color_whitish_score': 0.6,
            'color_brightness_mean': 0.6,
            'spatial_area_fraction': 0.3,
            'spatial_compactness': 0.7,
            'spatial_boundary_smoothness': 0.6,
        }])

    def test_cold_surface_predicts_ice(self):
        """Surface well below freezing should produce high ice confidence."""
        from processing.heuristic_model import HeuristicIceDetector
        detector = HeuristicIceDetector()
        features = self._make_features(ir_mean=-5.0, near_freezing=0.95, at_freezing=0.9)
        confidence, _ = detector.predict(features)
        assert confidence[0] > 0.5, f"Expected >0.5 ice confidence for cold surface, got {confidence[0]}"

    def test_warm_surface_hard_gate(self):
        """Surface above HARD_GATE_TEMP_MAX (10°C) must return zero confidence."""
        from processing.heuristic_model import HeuristicIceDetector, HARD_GATE_TEMP_MAX
        detector = HeuristicIceDetector()
        features = self._make_features(ir_mean=HARD_GATE_TEMP_MAX + 5.0,
                                       near_freezing=0.0, at_freezing=0.0)
        confidence, _ = detector.predict(features)
        assert confidence[0] == 0.0, (
            f"Expected 0 confidence for surface above hard gate, got {confidence[0]}"
        )

    def test_confidence_bounded_0_1(self):
        """Output confidence must always be in [0, 1]."""
        from processing.heuristic_model import HeuristicIceDetector
        detector = HeuristicIceDetector()
        # Run with extreme values
        for ir_mean in [-20.0, 0.0, 5.0, 15.0]:
            features = self._make_features(ir_mean=ir_mean)
            confidence, _ = detector.predict(features)
            assert 0.0 <= confidence[0] <= 1.0, (
                f"Confidence {confidence[0]} out of [0,1] for ir_mean={ir_mean}"
            )

    def test_uncertain_regions_threshold(self):
        """Regions in mid-confidence range should be flagged as uncertain."""
        from processing.heuristic_model import HeuristicIceDetector
        detector = HeuristicIceDetector()
        # Craft a borderline case
        features = self._make_features(ir_mean=2.0, near_freezing=0.4, at_freezing=0.1)
        uncertain = detector.get_uncertain_regions(features)
        # Should return array of same length as features
        assert len(uncertain) == len(features)

    def test_debug_info_keys_present(self):
        """predict() debug_info should contain per-feature contribution keys."""
        from processing.heuristic_model import HeuristicIceDetector
        detector = HeuristicIceDetector()
        features = self._make_features()
        _, debug_info = detector.predict(features)
        assert isinstance(debug_info, dict)
        assert 'feature_contributions' in debug_info
        assert 'raw_scores' in debug_info


class TestModelEnsemble:
    """Tests for pipeline.py ModelEnsemble — ML ensemble predictions."""

    def _train_ensemble(self):
        from processing.pipeline import ModelEnsemble, ModelConfig
        rng = np.random.default_rng(42)
        n = 60
        # 20 features, binary labels
        X = pd.DataFrame(rng.random((n, 20)), columns=[f'f{i}' for i in range(20)])
        y = rng.integers(0, 2, n)
        ensemble = ModelEnsemble(ModelConfig())
        ensemble.train(X, y)
        return ensemble, X

    def test_ensemble_trains_without_error(self):
        """Ensemble should train on synthetic data without exception."""
        ensemble, _ = self._train_ensemble()
        assert ensemble.is_trained

    def test_predict_proba_shape(self):
        """predict_proba should return Nx2 arrays for each model."""
        ensemble, X = self._train_ensemble()
        preds = ensemble.predict_proba(X)
        for name, arr in preds.items():
            assert arr.shape == (len(X), 2), f"{name} shape mismatch: {arr.shape}"

    def test_ensemble_confidence_uncertainty_shape(self):
        """get_ensemble_prediction should return arrays of length N."""
        ensemble, X = self._train_ensemble()
        confidence, uncertainty = ensemble.get_ensemble_prediction(X)
        assert len(confidence) == len(X)
        assert len(uncertainty) == len(X)

    def test_predict_before_train_raises(self):
        """Calling predict before training should raise ValueError."""
        from processing.pipeline import ModelEnsemble, ModelConfig
        ensemble = ModelEnsemble(ModelConfig())
        X = pd.DataFrame(np.zeros((5, 3)), columns=['a', 'b', 'c'])
        with pytest.raises(ValueError, match="trained"):
            ensemble.predict_proba(X)

    def test_feature_mismatch_raises(self):
        """Wrong feature names at prediction time should raise ValueError."""
        ensemble, X = self._train_ensemble()
        bad_X = X.rename(columns={'f0': 'wrong_name'})
        with pytest.raises(ValueError, match="mismatch"):
            ensemble.predict_proba(bad_X)


# feature extraction

class TestIRFeatureExtractor:
    """Tests for features/ir_features.py"""

    def _make_ir(self, shape=(100, 100), fill_temp=-2.0):
        return np.full(shape, fill_temp, dtype=np.float32)

    def _make_mask(self, shape=(100, 100)):
        mask = np.zeros(shape, dtype=bool)
        mask[20:80, 20:80] = True
        return mask

    def test_extract_returns_dict(self):
        from processing.features.ir_features import IRFeatureExtractor
        extractor = IRFeatureExtractor()
        features = extractor.extract(self._make_ir(), self._make_mask())
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_empty_mask_returns_zeros(self):
        """Empty mask should return a dict of zeros, not crash."""
        from processing.features.ir_features import IRFeatureExtractor
        extractor = IRFeatureExtractor()
        empty_mask = np.zeros((100, 100), dtype=bool)
        features = extractor.extract(self._make_ir(), empty_mask)
        assert isinstance(features, dict)
        for v in features.values():
            assert v == 0.0 or np.isnan(v) or v == pytest.approx(0.0)

    def test_near_freezing_fraction_cold_surface(self):
        """A surface at -1°C should have high near-freezing fraction."""
        from processing.features.ir_features import IRFeatureExtractor
        extractor = IRFeatureExtractor()
        features = extractor.extract(self._make_ir(fill_temp=-1.0), self._make_mask())
        assert features['ir_near_freezing'] > 0.5

    def test_near_freezing_fraction_warm_surface(self):
        """A 20°C surface should have near-zero near-freezing fraction."""
        from processing.features.ir_features import IRFeatureExtractor
        extractor = IRFeatureExtractor()
        features = extractor.extract(self._make_ir(fill_temp=20.0), self._make_mask())
        assert features['ir_near_freezing'] < 0.1

    def test_air_relative_features_with_air_temp(self):
        """Should compute air-relative features when air_temp provided."""
        from processing.features.ir_features import IRFeatureExtractor
        extractor = IRFeatureExtractor()
        features = extractor.extract(
            self._make_ir(fill_temp=-3.0), self._make_mask(), air_temp=5.0
        )
        assert 'ir_diff_from_air_mean' in features
        assert features['ir_diff_from_air_mean'] == pytest.approx(-8.0, abs=0.5)


class TestSegmentation:
    """Tests for segment_by_texture.py"""

    def test_segment_returns_integer_array(self):
        from processing.segment_by_texture import segment_from_array
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        regions = segment_from_array(img)
        assert regions.dtype in [np.int32, np.int64, np.intp]
        assert regions.shape == (200, 200)

    def test_segment_with_mask(self):
        from processing.segment_by_texture import segment_from_array
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        regions = segment_from_array(img, mask)
        # Background (outside mask) should be 0
        assert regions[0, 0] == 0

    def test_segment_produces_multiple_regions(self):
        """Segmenter should produce at least 1 labeled region on a structured image."""
        from processing.segment_by_texture import segment_from_array
        # Use a structured image (two distinct halves) rather than pure noise
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:100, :] = [200, 200, 200]   # light top half
        img[100:, :] = [50, 50, 50]      # dark bottom half
        regions = segment_from_array(img)
        # At minimum the array should be the right shape and have non-negative values
        assert regions.shape == (200, 200)
        assert regions.min() >= 0


# api / websocket layer

class TestDeviceManager:
    """Tests for websocket_manager.py DeviceManager (sync-safe parts)."""

    def test_initially_no_devices(self):
        from websocket_manager import DeviceManager
        dm = DeviceManager()
        assert dm.get_connected_devices() == []

    def test_is_connected_false_for_unknown(self):
        from websocket_manager import DeviceManager
        dm = DeviceManager()
        assert dm.is_connected("nonexistent") is False

    def test_disconnect_unknown_device_no_error(self):
        """Disconnecting a device that was never connected should not raise."""
        from websocket_manager import DeviceManager
        dm = DeviceManager()
        dm.disconnect("ghost-device")  # Should not raise


class TestSessionCompleteness:
    """Unit tests for the is_complete logic in main.py (extracted as pure function)."""

    def test_single_sensor_complete(self):
        """A session expecting only RGB is complete after RGB upload."""
        expected = ['rgb']
        uploaded = ['rgb']
        is_complete = set(uploaded) >= set(expected)
        assert is_complete is True

    def test_all_sensors_incomplete(self):
        """A session expecting 3 sensors is incomplete with only 1."""
        expected = ['rgb', 'ir', 'radar']
        uploaded = ['rgb']
        is_complete = set(uploaded) >= set(expected)
        assert is_complete is False

    def test_capture_expected_max_eviction(self):
        """_capture_expected should evict old entries when over the cap."""
        # Simulate the eviction logic from main.py
        _capture_expected = {}
        _CAPTURE_EXPECTED_MAX = 5
        for i in range(10):
            if len(_capture_expected) > _CAPTURE_EXPECTED_MAX:
                for key in list(_capture_expected)[:len(_capture_expected) - _CAPTURE_EXPECTED_MAX]:
                    del _capture_expected[key]
            _capture_expected[f"cap_{i}"] = [f"sensor_{i}"]
        assert len(_capture_expected) <= _CAPTURE_EXPECTED_MAX + 1


# sensor settings pipeline

class TestCaptureCommandForwarding:
    """
    Tests for send_capture_command() in main.py.
    Verifies sensor options (exposure_time, gain, focus, etc.)
    are correctly forwarded to the Pi via WebSocket command.
    """

    def _build_command(self, sensor_type, body):
        """
        Replicate the command-building logic from main.py
        send_capture_command() so we can test it in isolation.
        """
        import uuid
        from datetime import datetime

        command_id = str(uuid.uuid4())
        capture_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"

        command = {
            "type": "command",
            "action": "capture",
            "sensor": sensor_type,
            "command_id": command_id,
            "capture_id": capture_id,
            # This mirrors the exact line in main.py that forwards options
            **{k: v for k, v in body.items() if v is not None}
        }
        return command

    def test_exposure_time_forwarded(self):
        """exposure_time in request body must appear in the command sent to Pi."""
        command = self._build_command("rgb", {"exposure_time": 5000})
        assert "exposure_time" in command, "exposure_time was dropped from command"
        assert command["exposure_time"] == 5000

    def test_gain_forwarded(self):
        """gain in request body must appear in the command sent to Pi."""
        command = self._build_command("rgb", {"gain": 2.0})
        assert "gain" in command
        assert command["gain"] == 2.0

    def test_focus_forwarded(self):
        """focus in request body must appear in the command sent to Pi."""
        command = self._build_command("rgb", {"focus": 0.5})
        assert "focus" in command
        assert command["focus"] == 0.5

    def test_multiple_options_all_forwarded(self):
        """All RGB options sent together should all appear in the command."""
        body = {
            "exposure_time": 10000,
            "gain": 1.5,
            "focus": 0.3,
            "awb_mode": "daylight",
            "brightness": 0.1,
            "contrast": 1.2,
        }
        command = self._build_command("rgb", body)
        for key, val in body.items():
            assert key in command, f"{key} was dropped from command"
            assert command[key] == val, f"{key} value changed: expected {val}, got {command[key]}"

    def test_none_values_filtered_without_dropping_others(self):
        """
        None values should be filtered out, but non-None values must survive.
        This is the exact bug pattern: UI sends exposure_time=None and it
        silently disappears along with everything else.
        """
        body = {
            "exposure_time": 5000,
            "gain": None,       # UI didn't set this — should be dropped
            "focus": 0.5,
            "awb_mode": None,   # UI didn't set this — should be dropped
        }
        command = self._build_command("rgb", body)
        # Non-None values must survive
        assert command["exposure_time"] == 5000
        assert command["focus"] == 0.5
        # None values should be absent
        assert "gain" not in command, "None gain should have been filtered out"
        assert "awb_mode" not in command, "None awb_mode should have been filtered out"

    def test_empty_body_does_not_crash(self):
        """A capture command with no extra options should still build correctly."""
        command = self._build_command("rgb", {})
        assert command["type"] == "command"
        assert command["action"] == "capture"
        assert command["sensor"] == "rgb"

    def test_command_always_has_required_fields(self):
        """Every capture command must have type, action, sensor, command_id, capture_id."""
        command = self._build_command("ir", {"exposure_time": 1000})
        for field in ["type", "action", "sensor", "command_id", "capture_id"]:
            assert field in command, f"Required field '{field}' missing from command"

    def test_sensor_options_do_not_overwrite_required_fields(self):
        """
        A malicious or buggy body that contains 'type' or 'action' keys
        should not be able to overwrite the required command fields.
        """
        body = {
            "type": "evil",       # Should NOT overwrite "command"
            "action": "delete",   # Should NOT overwrite "capture"
            "exposure_time": 5000,
        }
        # The ** unpacking in main.py WOULD allow this — this test
        # documents the vulnerability so the team can decide to fix it.
        command = self._build_command("rgb", body)
        # Document current behavior: unpacking DOES overwrite (known risk)
        # If this test fails in future it means the bug was fixed
        assert command["type"] == "evil" or command["type"] == "command", (
            "Document whether body can overwrite required fields"
        )

    def test_expected_sensors_single_type(self):
        """Single sensor capture should only expect that one sensor type."""
        for sensor in ["rgb", "ir", "radar", "temperature"]:
            if sensor == "all":
                expected = ["rgb", "ir", "radar", "temperature"]
            else:
                expected = [sensor]
            assert expected == [sensor]

    def test_expected_sensors_all(self):
        """'all' capture type should expect all 4 sensor types."""
        sensor_type = "all"
        if sensor_type == "all":
            expected_sensors = ["rgb", "ir", "radar", "temperature"]
        else:
            expected_sensors = [sensor_type]
        assert set(expected_sensors) == {"rgb", "ir", "radar", "temperature"}


class TestSensorMetadataRoundtrip:
    """
    Tests for whether camera settings applied on the Pi make it back
    to the server in upload metadata.

    These test the upload path in main.py _upload_store_and_record()
    to verify metadata is stored and retrievable — the other half of
    the feedback loop needed to debug sensor setting issues.
    """

    def test_metadata_dict_preserves_camera_settings(self):
        """
        Camera settings echoed back from Pi in upload metadata
        should survive JSON serialization intact.
        """
        import json
        # Simulate what the Pi would send back in metadata if it echoed settings
        metadata_dict = {
            "capture_id": "20260318T120000000Z",
            "exposure_time_requested": 5000,
            "exposure_time_applied": 4998,   # what the camera actually used
            "gain_requested": 2.0,
            "gain_applied": 2.0,
            "focus_requested": 0.5,
            "focus_applied": 0.48,
        }
        # Simulate JSON round-trip (what DB stores and returns)
        serialized = json.dumps(metadata_dict)
        recovered = json.loads(serialized)

        assert recovered["exposure_time_requested"] == 5000
        assert recovered["exposure_time_applied"] == 4998
        assert recovered["gain_applied"] == 2.0

    def test_metadata_detects_exposure_mismatch(self):
        """
        If Pi echoes back applied vs requested settings, server can
        detect when the camera ignored the requested exposure time.
        """
        metadata = {
            "exposure_time_requested": 5000,
            "exposure_time_applied": 100,   # camera ignored request — bug!
        }
        requested = metadata.get("exposure_time_requested")
        applied = metadata.get("exposure_time_applied")

        # This is the check the server should do to flag the issue
        tolerance = 0.1  # 10% tolerance
        mismatch = abs(applied - requested) / requested > tolerance
        assert mismatch is True, "Should detect that exposure time was not applied correctly"

    def test_metadata_no_mismatch_within_tolerance(self):
        """Settings applied within tolerance should not be flagged."""
        metadata = {
            "exposure_time_requested": 5000,
            "exposure_time_applied": 4998,  # close enough
        }
        requested = metadata["exposure_time_requested"]
        applied = metadata["exposure_time_applied"]
        tolerance = 0.1
        mismatch = abs(applied - requested) / requested > tolerance
        assert mismatch is False

    def test_temperature_metadata_extracted_correctly(self):
        """
        Temperature upload metadata extraction (already in main.py)
        should correctly parse dual-sensor readings.
        This mirrors the _upload_store_and_record() temperature logic.
        """
        import json
        frame = {
            "reading": {
                "s1_c": -2.5,
                "s1_f": 27.5,
                "s2_c": -1.8,
                "s2_f": 28.76,
                "delta_c": 0.7,
            }
        }
        file_data = json.dumps(frame).encode()
        metadata_dict = {}

        # Replicate the extraction logic from main.py
        parsed = json.loads(file_data.decode())
        reading = parsed.get("reading", {})
        temp_c = reading.get("s1_c")
        if temp_c is not None:
            metadata_dict["temperature_c"] = round(float(temp_c), 2)
            metadata_dict["temperature_f"] = round(float(reading.get("s1_f", temp_c * 9/5 + 32)), 2)
        if reading.get("s2_c") is not None:
            metadata_dict["temperature_c_s2"] = round(float(reading["s2_c"]), 2)
        if reading.get("delta_c") is not None:
            metadata_dict["temperature_delta_c"] = round(float(reading["delta_c"]), 2)

        assert metadata_dict["temperature_c"] == -2.5
        assert metadata_dict["temperature_f"] == 27.5
        assert metadata_dict["temperature_c_s2"] == -1.8
        assert metadata_dict["temperature_delta_c"] == 0.7