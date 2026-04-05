"""
segment_by_texture.py
Region Segmentation Module
Creates texture-based region segmentation using LBP + Watershed
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import rgb2gray
from skimage.filters import sobel, gaussian
from skimage.feature import local_binary_pattern, peak_local_max
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SegmentationConfig:
    """Configuration for LBP watershed segmentation"""
    P: int = 8                      # Number of circularly symmetric neighbor points
    R: int = 1                      # Radius of circle
    sigma: float = 1.0              # Gaussian smoothing for LBP energy
    min_distance: int = 200          # Minimum distance between segment markers
    threshold_rel: float = 0.1      # Relative threshold for peak detection (0-1)
    
    def __str__(self):
        return (f"SegmentationConfig(P={self.P}, R={self.R}, sigma={self.sigma}, "
                f"min_distance={self.min_distance}, threshold_rel={self.threshold_rel})")


class RegionSegmenter:
    """Handles texture-based region segmentation"""
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
    
    def segment_image(self, 
                     image: np.ndarray, 
                     mask: Optional[np.ndarray] = None,
                     return_visualization: bool = False) -> np.ndarray:
        """
        Segment an image into texture-based regions.
        
        Args:
            image: Input image (RGB or BGR, will be converted to grayscale internally)
            mask: Optional binary mask (255=process, 0=ignore). If None, process entire image.
            return_visualization: If True, return (segments, visualization_dict)
        
        Returns:
            segments: Integer array where each pixel value represents its region ID (0=background)
            If return_visualization=True: tuple of (segments, viz_dict)
        """
        # Convert to RGB if BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
        
        # Convert to grayscale
        gray = rgb2gray(img_rgb) if len(img_rgb.shape) == 3 else img_rgb
        
        # Create mask if not provided
        if mask is None:
            binary_mask = np.ones(gray.shape, dtype=np.uint8)
        else:
            binary_mask = (mask > 0).astype(np.uint8)
        
        # Compute gradient for watershed
        gradient = sobel(gray)
        
        # Perform LBP-based watershed
        segments = self._watershed_lbp(
            gray=gray,
            binary_mask=binary_mask,
            gradient=gradient,
            P=self.config.P,
            R=self.config.R,
            sigma=self.config.sigma,
            min_distance=self.config.min_distance,
            threshold_rel=self.config.threshold_rel
        )
        
        if return_visualization:
            viz_dict = self._create_visualization(img_rgb, segments, binary_mask)
            return segments, viz_dict
        
        return segments
    
    def _watershed_lbp(self, 
                      gray: np.ndarray, 
                      binary_mask: np.ndarray, 
                      gradient: np.ndarray,
                      P: int, 
                      R: int, 
                      sigma: float, 
                      min_distance: int, 
                      threshold_rel: float) -> np.ndarray:
        """
        Perform LBP-based watershed segmentation.
        
        The algorithm:
        1. Compute Local Binary Pattern (LBP) to capture texture
        2. Smooth LBP energy map
        3. Find local maxima as markers
        4. Watershed on gradient using markers
        """
        if gray.dtype != np.uint8:
       	    gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray
	# Compute LBP texture energy
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        lbp_energy = gaussian(lbp.astype(float), sigma=sigma)
        
        # ── DIAGNOSTICS ──────────────────────────────────────────────────
        import logging
        log = logging.getLogger("app.processing")
        log.info(f"gray shape={gray.shape} dtype={gray.dtype} min={gray.min()} max={gray.max()}")
        log.info(f"binary_mask shape={binary_mask.shape} dtype={binary_mask.dtype} nonzero={np.count_nonzero(binary_mask)} total={binary_mask.size}")
        log.info(f"lbp_energy min={lbp_energy.min():.4f} max={lbp_energy.max():.4f} mean={lbp_energy.mean():.4f}")
        # ─────────────────────────────────────────────────────────────────
        
        # Find peaks in LBP energy as segment markers
        coords = peak_local_max(
            lbp_energy,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            labels=binary_mask
        )
        
        log.info(f"peak_local_max found {len(coords)} coords with min_distance={min_distance} threshold_rel={threshold_rel}")

        # Create marker image
        markers = np.zeros(gray.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        
        # Watershed segmentation
        segments = watershed(gradient, markers, mask=binary_mask)
        log.info(f"watershed result: {segments.max()} regions")
        return segments
    
    def _create_visualization(self, 
                            img_rgb: np.ndarray, 
                            segments: np.ndarray,
                            mask: np.ndarray) -> dict:
        """Create visualization outputs"""
        overlay = mark_boundaries(img_rgb, segments, color=(1, 0, 0), mode='thick')
        colored_segments = self._random_colors(segments)
        num_regions = segments.max()
        
        return {
            'overlay': overlay,
            'colored_segments': colored_segments,
            'num_regions': num_regions,
            'original': img_rgb,
            'segments': segments,
            'mask': mask
        }
    
    @staticmethod
    def _random_colors(labels: np.ndarray) -> np.ndarray:
        """Generate random colors for each segment"""
        n = labels.max() + 1
        colors = np.random.rand(n, 3)
        colors[0] = 0  # Background stays black
        return colors[labels]
    
    def visualize(self, 
                 image: np.ndarray, 
                 segments: np.ndarray,
                 save_path: Optional[str] = None,
                 show: bool = True) -> plt.Figure:
        """
        Visualize segmentation results.
        
        Args:
            image: Original RGB image
            segments: Segmentation output
            save_path: Optional path to save figure
            show: Whether to display the figure
        
        Returns:
            matplotlib Figure object
        """
        overlay = mark_boundaries(image, segments, color=(1, 0, 0), mode='thick')
        colored_segments = self._random_colors(segments)
        num_regions = segments.max()
        
        fig = plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original Image')
        
        plt.subplot(1, 3, 2)
        plt.imshow(overlay)
        plt.axis('off')
        plt.title(f'Boundary Overlay\n{num_regions} regions')
        
        plt.subplot(1, 3, 3)
        plt.imshow(colored_segments)
        plt.axis('off')
        plt.title(f'Colored Segments\n{self.config}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def get_region_info(self, segments: np.ndarray) -> dict:
        """Get information about segmented regions"""
        unique_regions = np.unique(segments)
        unique_regions = unique_regions[unique_regions != 0]  # Exclude background
        
        region_info = {
            'num_regions': len(unique_regions),
            'region_ids': unique_regions.tolist(),
            'region_sizes': {}
        }
        
        for region_id in unique_regions:
            region_info['region_sizes'][int(region_id)] = int(np.sum(segments == region_id))
        
        return region_info


# -------------------------------
# Convenience functions
# -------------------------------

def segment_image(image_path: str, 
                 mask_path: Optional[str] = None,
                 config: Optional[SegmentationConfig] = None,
                 visualize: bool = True) -> np.ndarray:
    """
    One-line convenience function to segment an image.
    
    Args:
        image_path: Path to input image
        mask_path: Optional path to ROI mask
        config: Optional segmentation configuration
        visualize: Whether to show visualization
    
    Returns:
        segments: Region segmentation array
    
    Example:
        >>> segments = segment_image('input.jpg', 'mask.png')
        >>> segments = segment_image('input.jpg', config=SegmentationConfig(min_distance=50))
    """
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Load mask if provided
    mask = None
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask from {mask_path}, processing entire image")
    
    # Create segmenter
    segmenter = RegionSegmenter(config)
    
    # Segment
    segments = segmenter.segment_image(img_rgb, mask)
    
    # Visualize if requested
    if visualize:
        segmenter.visualize(img_rgb, segments)
    
    # Print info
    info = segmenter.get_region_info(segments)
    print(f"Segmentation complete: {info['num_regions']} regions found")
    
    return segments


def segment_from_array(image: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      config: Optional[SegmentationConfig] = None) -> np.ndarray:
    """
    Segment from numpy arrays (for video processing or when images are already loaded).
    
    Args:
        image: Input image array (RGB or BGR)
        mask: Optional binary mask array
        config: Optional segmentation configuration
    
    Returns:
        segments: Region segmentation array
    
    Example:
        >>> import cv2
        >>> frame = cv2.imread('frame.jpg')
        >>> segments = segment_from_array(frame)
    """
    segmenter = RegionSegmenter(config)
    return segmenter.segment_image(image, mask)

"""
# -------------------------------
# Example usage and testing
# -------------------------------

if __name__ == "__main__":
    # Example 1: Simple one-liner
    print("Example 1: Basic segmentation")
    segments = segment_image('porchview.jpeg', 'porch_mask.png')
    print(f"Segments shape: {segments.shape}")
    print(f"Unique regions: {np.unique(segments)}")
    
    # Example 2: Custom configuration
    print("\nExample 2: Custom configuration")
    custom_config = SegmentationConfig(
        min_distance=50,      # More segments (smaller regions)
        threshold_rel=0.05,   # More sensitive to texture
        sigma=2.0             # More smoothing
    )
    segments_fine = segment_image('porchview.jpeg', 'porch_mask.png', 
                                  config=custom_config, visualize=True)
    
    # Example 3: Using the class directly for more control
    print("\nExample 3: Using class directly")
    segmenter = RegionSegmenter(config=SegmentationConfig(min_distance=90))
    
    img = cv2.imread('porchview.jpeg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('porch_mask.png', cv2.IMREAD_GRAYSCALE)
    
    segments, viz = segmenter.segment_image(img_rgb, mask, return_visualization=True)
    
    print(f"Found {viz['num_regions']} regions")
    print(f"Overlay shape: {viz['overlay'].shape}")
    
    # Get detailed region info
    info = segmenter.get_region_info(segments)
    print(f"\nRegion sizes:")
    for region_id, size in info['region_sizes'].items():
        print(f"  Region {region_id}: {size} pixels")
    
    # Example 4: Video processing workflow
    print("\nExample 4: Video frame processing")
    cap = cv2.VideoCapture('video.mp4')  # Your video file
    
    # Process first frame
    ret, frame = cap.read()
    if ret:
        segments = segment_from_array(frame, config=SegmentationConfig(min_distance=70))
        print(f"Frame segmented into {segments.max()} regions")
    
    cap.release()
    
    # Example 5: Parameter sweep (like your original code)
    print("\nExample 5: Parameter sweep")
    min_distances = [30, 60, 90, 120]
    threshold_rels = [0.05, 0.1, 0.15]
    
    img = cv2.imread('porchview.jpeg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('porch_mask.png', cv2.IMREAD_GRAYSCALE)
    
    fig, axes = plt.subplots(len(threshold_rels), len(min_distances), 
                            figsize=(16, 10))
    
    for i, thr in enumerate(threshold_rels):
        for j, min_dist in enumerate(min_distances):
            config = SegmentationConfig(
                min_distance=min_dist,
                threshold_rel=thr
            )
            segmenter = RegionSegmenter(config)
            segments = segmenter.segment_image(img_rgb, mask)
            
            overlay = mark_boundaries(img_rgb, segments, color=(1, 0, 0), mode='thick')
            
            axes[i, j].imshow(overlay)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'dist={min_dist}, thr={thr}\n{segments.max()} regions',
                                fontsize=8)
    
    plt.tight_layout()
    plt.savefig('parameter_sweep.png', dpi=150)
    plt.show()
"""
