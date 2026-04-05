"""
pipeline.py
Ice Detection Inference Pipeline
Generates confidence maps for labeling assistance
"""

import numpy as np
import pandas as pd
import cv2
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from segment_by_texture import segment_from_array, SegmentationConfig
from feature_extraction import FeatureExtractor, TemporalConfig
from heuristic_model import HeuristicWeights, HeuristicIceDetector, PresetWeights

@dataclass
class ModelConfig:
    """Configuration for model ensemble"""
    use_logistic: bool = True
    use_naive_bayes: bool = True
    use_random_forest: bool = True
    
    # Smoothing parameters
    apply_gaussian_smoothing: bool = True
    gaussian_sigma: float = 2.0
    apply_median_smoothing: bool = True
    median_size: int = 5
    
    # Confidence thresholds
    uncertain_threshold_low: float = 0.35   # Below this = likely not ice
    uncertain_threshold_high: float = 0.65  # Above this = likely ice
    
    # Model paths
    model_dir: Path = Path("models")
    scaler_path: Optional[Path] = None
    logistic_path: Optional[Path] = None
    naive_bayes_path: Optional[Path] = None
    random_forest_path: Optional[Path] = None


class ModelEnsemble:
    """Manages ensemble of ice detection models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.logistic: Optional[LogisticRegression] = None
        self.naive_bayes: Optional[GaussianNB] = None
        self.random_forest: Optional[RandomForestClassifier] = None
        self.feature_names: Optional[List[str]] = None
        
        self.is_trained = False
    
    def train(self, features_df: pd.DataFrame, labels: np.ndarray):
        """
        Train all models in the ensemble.
        
        Args:
            features_df: DataFrame with features (from FeatureExtractor)
            labels: Binary labels (1=ice, 0=not ice)
        """
        # Remove non-feature columns
        exclude_cols = ['region_id', 'timestamp']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols].values
        self.feature_names = feature_cols
        
        # Handle inf/nan values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        if self.config.use_logistic:
            self.logistic = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            )
            self.logistic.fit(X_scaled, labels)
        
        if self.config.use_naive_bayes:
            self.naive_bayes = GaussianNB()
            self.naive_bayes.fit(X_scaled, labels)
        
        if self.config.use_random_forest:
            self.random_forest = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            self.random_forest.fit(X_scaled, labels)
        
        self.is_trained = True
        print(f"✓ Trained {self._num_active_models()} models on {len(labels)} samples")
    
    def predict_proba(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get probability predictions from all models.
        
        Returns:
            Dictionary mapping model_name -> probabilities (Nx2 array)
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
            
        # feature validation
        feature_cols = [col for col in features_df.columns if col not in ['region_id', 'timestamp']]
        if set(feature_cols) != set(self.feature_names):
            missing = set(self.feature_names) - set(feature_cols)
            extra = set(feature_cols) - set(self.feature_names)
            raise ValueError(f"Feature mismatch! Missing: {missing}, Extra: {extra}")
        
        # Extract and scale features
        exclude_cols = ['region_id', 'timestamp']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        
        if self.config.use_logistic and self.logistic is not None:
            predictions['logistic'] = self.logistic.predict_proba(X_scaled)
        
        if self.config.use_naive_bayes and self.naive_bayes is not None:
            predictions['naive_bayes'] = self.naive_bayes.predict_proba(X_scaled)
        
        if self.config.use_random_forest and self.random_forest is not None:
            predictions['random_forest'] = self.random_forest.predict_proba(X_scaled)
        
        return predictions
    
    def get_ensemble_prediction(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ensemble prediction (average of all models).
        
        Returns:
            confidence: Probability of ice (0-1)
            uncertainty: Disagreement between models (0-1)
        """
        predictions = self.predict_proba(features_df)
        
        if not predictions:
            raise ValueError("No trained models available")
        
        # Extract ice probabilities (class 1)
        ice_probs = np.array([pred[:, 1] for pred in predictions.values()])
        
        # Average prediction
        confidence = np.mean(ice_probs, axis=0)
        
        # Uncertainty = standard deviation across models
        uncertainty = np.std(ice_probs, axis=0)
        
        return confidence, uncertainty
    
    def save(self, directory: Optional[Path] = None):
        """Save all trained models"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        save_dir = directory or self.config.model_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        with open(save_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        with open(save_dir / "feature_names.pkl", 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save models
        if self.logistic is not None:
            with open(save_dir / "logistic.pkl", 'wb') as f:
                pickle.dump(self.logistic, f)
        
        if self.naive_bayes is not None:
            with open(save_dir / "naive_bayes.pkl", 'wb') as f:
                pickle.dump(self.naive_bayes, f)
        
        if self.random_forest is not None:
            with open(save_dir / "random_forest.pkl", 'wb') as f:
                pickle.dump(self.random_forest, f)
        
        print(f"✓ Models saved to {save_dir}")
    
    def load(self, directory: Optional[Path] = None):
        """Load trained models"""
        load_dir = directory or self.config.model_dir
        
        if not load_dir.exists():
            raise ValueError(f"Model directory {load_dir} does not exist")
        
        # Load scaler
        with open(load_dir / "scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        with open(load_dir / "feature_names.pkl", 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Load models
        if self.config.use_logistic and (load_dir / "logistic.pkl").exists():
            with open(load_dir / "logistic.pkl", 'rb') as f:
                self.logistic = pickle.load(f)
        
        if self.config.use_naive_bayes and (load_dir / "naive_bayes.pkl").exists():
            with open(load_dir / "naive_bayes.pkl", 'rb') as f:
                self.naive_bayes = pickle.load(f)
        
        if self.config.use_random_forest and (load_dir / "random_forest.pkl").exists():
            with open(load_dir / "random_forest.pkl", 'rb') as f:
                self.random_forest = pickle.load(f)
        
        self.is_trained = True
        print(f"✓ Loaded {self._num_active_models()} models from {load_dir}")
    
    def _num_active_models(self) -> int:
        """Count number of active models"""
        count = 0
        if self.logistic is not None: count += 1
        if self.naive_bayes is not None: count += 1
        if self.random_forest is not None: count += 1
        return count
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from Random Forest (if available)"""
        if self.random_forest is None or self.feature_names is None:
            return None
        
        importances = self.random_forest.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df


class InferencePipeline:
    """Complete inference pipeline for ice detection"""
    
    def __init__(self,
                 model_config: Optional[ModelConfig] = None,
                 segmentation_config: Optional[SegmentationConfig] = None,
                 temporal_config: Optional[TemporalConfig] = None):
        
        self.model_config = model_config or ModelConfig()
        self.segmentation_config = segmentation_config or SegmentationConfig()
        self.temporal_config = temporal_config or TemporalConfig()
        
        self.ensemble = ModelEnsemble(self.model_config)
        self.feature_extractor = FeatureExtractor(self.temporal_config)
    
    def run_inference(self,
                     rgb_image: np.ndarray,
                     ir_image: np.ndarray,
                     radar_scan: Optional[np.ndarray] = None,
                     mask: Optional[np.ndarray] = None,
                     air_temp: Optional[float] = None,
                     timestamp: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Run complete inference pipeline.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            ir_image: Thermal IR image (H, W)
            radar_scan: Optional radar data
            mask: Optional ROI mask
            air_temp: Current air temperature
            timestamp: Current timestamp
        
        Returns:
            Dictionary containing:
                - 'confidence_map': Per-pixel confidence (0-1)
                - 'uncertainty_map': Per-pixel uncertainty (0-1)
                - 'overlay_rgb': Confidence overlay on RGB image
                - 'overlay_heatmap': Standalone heatmap
                - 'regions': Region segmentation
                - 'uncertain_regions': Mask of uncertain regions for labeling
        """
        # Step 1: Segment image into regions
        regions = segment_from_array(rgb_image, mask, self.segmentation_config)
        
        # Step 2: Extract features per region
        features_df = self.feature_extractor.extract_features(
            RGB_image=rgb_image,
            IR_image=ir_image,
            radar_scan=radar_scan,
            regions=regions,
            air_temp=air_temp,
            timestamp=timestamp
        )
        
        # Step 3: Get predictions if model is trained
        if not self.ensemble.is_trained:
            print("⚠ No trained models available - returning empty confidence map")
            return self._create_empty_output(rgb_image.shape[:2], regions)
        
        # Get ensemble predictions
        confidence, uncertainty = self.ensemble.get_ensemble_prediction(features_df)
        
        # Step 4: Map confidence back to pixel space
        confidence_map = self._regions_to_pixels(regions, features_df['region_id'].values, confidence)
        uncertainty_map = self._regions_to_pixels(regions, features_df['region_id'].values, uncertainty)
        
        # Step 5: Apply spatial smoothing
        if self.model_config.apply_gaussian_smoothing:
            confidence_map = gaussian_filter(confidence_map, sigma=self.model_config.gaussian_sigma)
            uncertainty_map = gaussian_filter(uncertainty_map, sigma=self.model_config.gaussian_sigma)
        
        if self.model_config.apply_median_smoothing:
            confidence_map = median_filter(confidence_map, size=self.model_config.median_size)
        
        # Step 6: Create visualizations
        overlay_rgb = self._create_overlay(rgb_image, confidence_map)
        overlay_heatmap = self._create_heatmap(confidence_map)
        uncertain_regions = self._identify_uncertain_regions(
            confidence_map, 
            uncertainty_map,
            regions
        )
        
        return {
            'confidence_map': confidence_map,
            'uncertainty_map': uncertainty_map,
            'overlay_rgb': overlay_rgb,
            'overlay_heatmap': overlay_heatmap,
            'regions': regions,
            'uncertain_regions': uncertain_regions,
            'features_df': features_df,  # For debugging
            'per_region_confidence': confidence,
            'per_region_uncertainty': uncertainty
        }
    
    def _regions_to_pixels(self, regions: np.ndarray, region_ids: np.ndarray, 
                          values: np.ndarray) -> np.ndarray:
        """Map per-region values back to pixel space"""
        pixel_map = np.zeros(regions.shape, dtype=np.float32)
        
        for region_id, value in zip(region_ids, values):
            pixel_map[regions == region_id] = value
        
        return pixel_map
    
    def _create_overlay(self, rgb_image: np.ndarray, confidence_map: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
        """Create confidence overlay on RGB image"""
        # Normalize RGB to 0-1
        rgb_norm = rgb_image.astype(np.float32) / 255.0
        
        # Create heatmap (blue=low, red=high)
        cmap = plt.cm.RdYlBu_r
        heatmap = cmap(confidence_map)[:, :, :3]  # Drop alpha channel
        
        # Blend
        overlay = (1 - alpha) * rgb_norm + alpha * heatmap
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        
        return overlay
    
    def _create_heatmap(self, confidence_map: np.ndarray) -> np.ndarray:
        """Create standalone heatmap visualization"""
        cmap = plt.cm.RdYlBu_r
        heatmap = cmap(confidence_map)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap
    
    def _identify_uncertain_regions(self, confidence_map: np.ndarray,
                                   uncertainty_map: np.ndarray,
                                   regions: np.ndarray) -> np.ndarray:
        """
        Identify uncertain regions for targeted labeling.
        
        Returns binary mask where 1 = should be labeled
        """
        # Regions with mid-range confidence
        mid_confidence = (confidence_map > self.model_config.uncertain_threshold_low) & \
                        (confidence_map < self.model_config.uncertain_threshold_high)
        
        # Regions with high model disagreement
        high_uncertainty = uncertainty_map > np.percentile(uncertainty_map[regions > 0], 75)
        
        # Combine: uncertain if either condition
        uncertain_mask = mid_confidence | high_uncertainty
        
        # Only consider regions within ROI
        uncertain_mask = uncertain_mask & (regions > 0)
        
        return uncertain_mask.astype(np.uint8)
    
    def _create_empty_output(self, shape: Tuple[int, int], regions: np.ndarray) -> Dict:
        """Create empty output when no model is trained"""
        empty_map = np.zeros(shape, dtype=np.float32)
        rgb_dummy = np.zeros((*shape, 3), dtype=np.uint8)
        
        return {
            'confidence_map': empty_map,
            'uncertainty_map': empty_map,
            'overlay_rgb': rgb_dummy,
            'overlay_heatmap': rgb_dummy,
            'regions': regions,
            'uncertain_regions': np.ones(shape, dtype=np.uint8),  # Label everything
            'features_df': pd.DataFrame(),
            'per_region_confidence': np.array([]),
            'per_region_uncertainty': np.array([])
        }
    
    def train_models(self, features_df: pd.DataFrame, labels: np.ndarray):
        """Train the model ensemble"""
        self.ensemble.train(features_df, labels)
    
    def save_models(self, directory: Optional[Path] = None):
        """Save trained models"""
        self.ensemble.save(directory)
    
    def load_models(self, directory: Optional[Path] = None):
        """Load trained models"""
        self.ensemble.load(directory)
        
class HeuristicInferencePipeline:
    """Inference pipeline using rule-based heuristic model (no ML training required)"""
    
    def __init__(self,
                 segmentation_config: Optional[SegmentationConfig] = None,
                 temporal_config: Optional[TemporalConfig] = None,
                 heuristic_weights: Optional[HeuristicWeights] = None,
                 smoothing_sigma: float = 2.0,
                 smoothing_median_size: int = 5):
        
        self.segmentation_config = segmentation_config or SegmentationConfig()
        self.temporal_config = temporal_config or TemporalConfig()
        
        self.feature_extractor = FeatureExtractor(self.temporal_config)
        self.heuristic_detector = HeuristicIceDetector(heuristic_weights)
        
        # Smoothing parameters
        self.smoothing_sigma = smoothing_sigma
        self.smoothing_median_size = smoothing_median_size
    
    def run_inference(self,
                     rgb_image: np.ndarray,
                     ir_image: np.ndarray,
                     radar_scan: Optional[np.ndarray] = None,
                     mask: Optional[np.ndarray] = None,
                     air_temp: Optional[float] = None,
                     timestamp: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Run heuristic inference pipeline.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            ir_image: Thermal IR image (H, W)
            radar_scan: Optional radar data
            mask: Optional ROI mask
            air_temp: Current air temperature
            timestamp: Current timestamp
        
        Returns:
            Dictionary containing inference results
        """
        # Step 1: Segment image into regions
        regions = segment_from_array(rgb_image, mask, self.segmentation_config)
        
        # Step 2: Extract features per region
        features_df = self.feature_extractor.extract_features(
            RGB_image=rgb_image,
            IR_image=ir_image,
            radar_scan=radar_scan,
            regions=regions,
            air_temp=air_temp,
            timestamp=timestamp
        )
        
        # Step 3: Get heuristic predictions
        confidence, debug_info = self.heuristic_detector.predict(features_df)
        uncertainty = np.zeros_like(confidence)  # Heuristic model has no uncertainty
        
        # Step 4: Map confidence back to pixel space
        confidence_map = self._regions_to_pixels(regions, features_df['region_id'].values, confidence)
        uncertainty_map = self._regions_to_pixels(regions, features_df['region_id'].values, uncertainty)
        
        # Step 5: Apply spatial smoothing
        confidence_map = gaussian_filter(confidence_map, sigma=self.smoothing_sigma)
        confidence_map = median_filter(confidence_map, size=self.smoothing_median_size)
        
        # Step 6: Create visualizations
        overlay_rgb = self._create_overlay(rgb_image, confidence_map)
        overlay_heatmap = self._create_heatmap(confidence_map)
        uncertain_regions = self.heuristic_detector.get_uncertain_regions(features_df)
        uncertain_regions_map = self._regions_to_pixels(
            regions, features_df['region_id'].values, uncertain_regions
        )
        
        return {
            'confidence_map': confidence_map,
            'uncertainty_map': uncertainty_map,
            'overlay_rgb': overlay_rgb,
            'overlay_heatmap': overlay_heatmap,
            'regions': regions,
            'uncertain_regions': uncertain_regions_map,
            'features_df': features_df,
            'per_region_confidence': confidence,
            'per_region_uncertainty': uncertainty,
            'debug_info': debug_info  # Heuristic-specific: feature contributions
        }
    
    def _regions_to_pixels(self, regions: np.ndarray, region_ids: np.ndarray, 
                          values: np.ndarray) -> np.ndarray:
        """Map per-region values back to pixel space"""
        pixel_map = np.zeros(regions.shape, dtype=np.float32)
        
        for region_id, value in zip(region_ids, values):
            pixel_map[regions == region_id] = value
        
        return pixel_map
    
    def _create_overlay(self, rgb_image: np.ndarray, confidence_map: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
        """Create confidence overlay on RGB image"""
        rgb_norm = rgb_image.astype(np.float32) / 255.0
        
        cmap = plt.cm.RdYlBu_r
        heatmap = cmap(confidence_map)[:, :, :3]
        
        overlay = (1 - alpha) * rgb_norm + alpha * heatmap
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        
        return overlay
    
    def _create_heatmap(self, confidence_map: np.ndarray) -> np.ndarray:
        """Create standalone heatmap visualization"""
        cmap = plt.cm.RdYlBu_r
        heatmap = cmap(confidence_map)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap
    
    def explain_region(self, features_df: pd.DataFrame, region_idx: int) -> str:
        """Get human-readable explanation for a region's prediction"""
        return self.heuristic_detector.explain_prediction(features_df, region_idx)
    
    def get_weights_summary(self) -> str:
        """Get current heuristic weights"""
        return self.heuristic_detector.visualize_weights()
    
    def save_weights(self, path: Path):
        """Save current heuristic weights"""
        self.heuristic_detector.weights.save(path)
    
    def load_weights(self, path: Path):
        """Load heuristic weights"""
        self.heuristic_detector.weights = HeuristicWeights.load(path)
