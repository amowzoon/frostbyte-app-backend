"""
radar_projection.py
Project mmWave Range-Azimuth heatmap onto road surface
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class RadarConfig:
    """Radar mounting and field-of-view configuration"""
    
    # Mounting geometry
    height: float = 2.0                    # Height above road (meters)
    tilt_angle_deg: float = 45.0           # Downward tilt angle (degrees)
    
    # Range bins
    range_min: float = 0.5                 # Minimum detection range (meters)
    range_max: float = 10.0                # Maximum detection range (meters)
    num_range_bins: int = 64               # Number of range bins
    
    # Azimuth bins (horizontal sweep)
    azimuth_min_deg: float = -60.0         # Leftmost angle (degrees)
    azimuth_max_deg: float = 60.0          # Rightmost angle (degrees)
    num_azimuth_bins: int = 128            # Number of azimuth bins
    
    # Projection settings
    road_width: float = 8.0                # Width of road to project (meters)
    road_length: float = 12.0              # Length of road to project (meters)
    projection_resolution: float = 0.05     # Pixels per meter
    
    def __post_init__(self):
        self.tilt_angle_rad = np.deg2rad(self.tilt_angle_deg)
        self.azimuth_min_rad = np.deg2rad(self.azimuth_min_deg)
        self.azimuth_max_rad = np.deg2rad(self.azimuth_max_deg)


class RadarProjector:
    """Projects mmWave Range-Azimuth heatmap onto road surface"""
    
    def __init__(self, config: RadarConfig):
        self.config = config
        
        # Pre-compute coordinate grids
        self._build_coordinate_grids()
    
    def _build_coordinate_grids(self):
        """Pre-compute range and azimuth coordinate grids"""
        # Range bins (1D array)
        self.ranges = np.linspace(
            self.config.range_min,
            self.config.range_max,
            self.config.num_range_bins
        )
        
        # Azimuth bins (1D array)
        self.azimuths = np.linspace(
            self.config.azimuth_min_rad,
            self.config.azimuth_max_rad,
            self.config.num_azimuth_bins
        )
        
        # 2D meshgrids for vectorized computation
        self.R, self.PHI = np.meshgrid(self.ranges, self.azimuths, indexing='ij')
        # Shape: (num_range_bins, num_azimuth_bins)
    
    def project_to_road(self, 
                       heatmap: np.ndarray,
                       mask_invalid: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project Range-Azimuth heatmap onto road surface.
        
        Args:
            heatmap: Range-Azimuth intensity map, shape (num_range_bins, num_azimuth_bins)
            mask_invalid: If True, set invalid projections to NaN
        
        Returns:
            x_road: X coordinates on road (meters), shape (num_range, num_azimuth)
            y_road: Y coordinates on road (meters), shape (num_range, num_azimuth)
            intensity: Projected intensity values, shape (num_range, num_azimuth)
        
        Coordinate system:
            - Origin at radar position projected onto road
            - X: Left (-) to Right (+)
            - Y: Radar (0) to Forward (+)
        """
        if heatmap.shape != (self.config.num_range_bins, self.config.num_azimuth_bins):
            raise ValueError(
                f"Heatmap shape {heatmap.shape} doesn't match config "
                f"({self.config.num_range_bins}, {self.config.num_azimuth_bins})"
            )
        
        # Fixed elevation angle (tilt)
        psi = self.config.tilt_angle_rad
        
        # Step 1: Spherical (range, azimuth, elevation) → Cartesian (x, y, z) in radar frame
        # Using physics/robotics convention: x=forward, y=left, z=up
        # But we'll use x=right, y=forward for road projection
        
        x_radar = self.R * np.cos(psi) * np.sin(self.PHI)   # Horizontal (left-right)
        y_radar = self.R * np.cos(psi) * np.cos(self.PHI)   # Horizontal (forward)
        z_radar = -self.R * np.sin(psi)                     # Vertical (down is negative)
        
        # Step 2: Intersect with road plane (z = -height)
        # Beam equation: (0,0,0) + t*(x_radar, y_radar, z_radar)
        # Road plane: z = -h
        # Solve: 0 + t*z_radar = -h  =>  t = -h/z_radar
        
        h = self.config.height
        
        # Compute intersection parameter t
        # Avoid division by zero: if z_radar ≈ 0, beam is parallel to road (invalid)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -h / z_radar
        
        # Intersection points on road
        x_road = t * x_radar
        y_road = t * y_radar
        
        # Step 3: Mask invalid projections
        if mask_invalid:
            # Invalid cases:
            # 1. t < 0: Beam points upward, never hits road
            # 2. t is infinite/NaN: Beam parallel to road
            # 3. y_road < 0: Intersection behind radar (shouldn't happen with proper tilt)
            
            valid_mask = (t > 0) & np.isfinite(t) & (y_road > 0)
            
            x_road = np.where(valid_mask, x_road, np.nan)
            y_road = np.where(valid_mask, y_road, np.nan)
        
        return x_road, y_road, heatmap
    
    def create_road_grid(self, 
                        x_road: np.ndarray,
                        y_road: np.ndarray,
                        intensity: np.ndarray,
                        method: str = 'nearest') -> np.ndarray:
        """
        Rasterize scattered radar points onto a regular road grid.
        
        Args:
            x_road: X coordinates (meters)
            y_road: Y coordinates (meters)
            intensity: Intensity values
            method: Interpolation method ('nearest', 'linear', 'max')
        
        Returns:
            road_grid: Regular grid image, shape (height, width)
        
        Grid coordinate system:
            - Row 0 = farthest forward (max Y)
            - Row -1 = closest to radar (min Y)
            - Col 0 = leftmost (min X)
            - Col -1 = rightmost (max X)
        """
        # Define regular grid
        width_pixels = int(self.config.road_width / self.config.projection_resolution)
        length_pixels = int(self.config.road_length / self.config.projection_resolution)
        
        # Grid coordinates (meters)
        x_grid = np.linspace(-self.config.road_width/2, self.config.road_width/2, width_pixels)
        y_grid = np.linspace(0, self.config.road_length, length_pixels)
        
        # Initialize output grid
        road_grid = np.zeros((length_pixels, width_pixels), dtype=np.float32)
        
        # Flatten inputs and remove NaNs
        x_flat = x_road.flatten()
        y_flat = y_road.flatten()
        i_flat = intensity.flatten()
        
        valid = ~(np.isnan(x_flat) | np.isnan(y_flat))
        x_flat = x_flat[valid]
        y_flat = y_flat[valid]
        i_flat = i_flat[valid]
        
        if len(x_flat) == 0:
            return road_grid  # No valid points
        
        # Map to grid indices
        # x: [-width/2, +width/2] → [0, width_pixels]
        # y: [0, length] → [length_pixels, 0] (flip Y so forward is up in image)
        
        col_indices = ((x_flat + self.config.road_width/2) / self.config.projection_resolution).astype(int)
        row_indices = (length_pixels - (y_flat / self.config.projection_resolution).astype(int))
        
        # Clip to valid range
        col_indices = np.clip(col_indices, 0, width_pixels - 1)
        row_indices = np.clip(row_indices, 0, length_pixels - 1)
        
        # Rasterize
        if method == 'nearest':
            # Simple: assign each point to nearest grid cell
            road_grid[row_indices, col_indices] = i_flat
        
        elif method == 'max':
            # Max pooling: take maximum intensity in each cell
            for r, c, val in zip(row_indices, col_indices, i_flat):
                road_grid[r, c] = max(road_grid[r, c], val)
        
        elif method == 'linear':
            # Proper interpolation (slower but smoother)
            from scipy.interpolate import griddata
            points = np.column_stack([x_flat, y_flat])
            grid_x, grid_y = np.meshgrid(x_grid, y_grid)
            road_grid = griddata(points, i_flat, (grid_x, grid_y), method='linear', fill_value=0)
        
        return road_grid
    
    def visualize_coverage(self, ax=None, show_radar_position: bool = True):
        """
        Visualize the radar's coverage area on the road.
        
        Returns:
            matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 10))
        
        # Create dummy heatmap to see coverage
        dummy_heatmap = np.ones((self.config.num_range_bins, self.config.num_azimuth_bins))
        
        x_road, y_road, _ = self.project_to_road(dummy_heatmap)
        
        # Plot valid projection points
        valid = ~np.isnan(x_road)
        ax.scatter(x_road[valid], y_road[valid], c='blue', s=1, alpha=0.3, label='Coverage')
        
        # Plot road boundaries
        half_width = self.config.road_width / 2
        ax.plot([-half_width, -half_width], [0, self.config.road_length], 'k--', label='Road edge')
        ax.plot([half_width, half_width], [0, self.config.road_length], 'k--')
        
        # Show radar position
        if show_radar_position:
            ax.plot(0, 0, 'r*', markersize=15, label='Radar position')
            ax.arrow(0, 0, 0, 1, head_width=0.3, head_length=0.2, fc='red', ec='red')
        
        # Annotate
        ax.set_xlabel('X (meters) - Left (-) to Right (+)')
        ax.set_ylabel('Y (meters) - Radar (0) to Forward (+)')
        ax.set_title(f'mmWave Radar Coverage\nHeight={self.config.height}m, Tilt={self.config.tilt_angle_deg}°')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        return ax.get_figure()


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example 1: Visualize coverage
    print("=" * 60)
    print("EXAMPLE 1: Visualize Radar Coverage")
    print("=" * 60)
    
    config = RadarConfig(
        height=2.0,              # 2m above road
        tilt_angle_deg=45.0,     # 45° downward
        range_max=10.0,          # 10m max range
        azimuth_min_deg=-60.0,   # ±60° horizontal FOV
        azimuth_max_deg=60.0
    )
    
    projector = RadarProjector(config)
    
    fig = projector.visualize_coverage()
    plt.savefig('radar_coverage.png', dpi=150, bbox_inches='tight')
    print("✓ Saved radar_coverage.png")
    plt.show()
    
    # Example 2: Project synthetic radar data
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Project Synthetic Radar Heatmap")
    print("=" * 60)
    
    # Create synthetic radar heatmap (simulate ice patch at range 5-7m, azimuth -20° to +20°)
    heatmap = np.random.rand(config.num_range_bins, config.num_azimuth_bins) * 0.2
    
    # Add "ice patch" - high intensity at specific location
    range_idx_ice = int(config.num_range_bins * (6.0 - config.range_min) / (config.range_max - config.range_min))
    azimuth_idx_ice = config.num_azimuth_bins // 2
    
    heatmap[range_idx_ice-5:range_idx_ice+5, azimuth_idx_ice-15:azimuth_idx_ice+15] = 0.9
    
    # Project to road
    x_road, y_road, intensity = projector.project_to_road(heatmap)
    
    # Create regular grid
    road_grid = projector.create_road_grid(x_road, y_road, intensity, method='max')
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original radar heatmap (polar)
    ax1 = axes[0]
    extent_polar = [
        config.azimuth_min_deg, config.azimuth_max_deg,
        config.range_max, config.range_min
    ]
    im1 = ax1.imshow(heatmap, aspect='auto', extent=extent_polar, cmap='hot')
    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('Range (meters)')
    ax1.set_title('Radar Range-Azimuth Heatmap')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # Projected road grid (Cartesian)
    ax2 = axes[1]
    extent_cart = [
        -config.road_width/2, config.road_width/2,
        0, config.road_length
    ]
    im2 = ax2.imshow(road_grid, aspect='auto', extent=extent_cart, cmap='hot', origin='upper')
    ax2.set_xlabel('X (meters) - Left to Right')
    ax2.set_ylabel('Y (meters) - Forward Distance')
    ax2.set_title('Projected onto Road Surface')
    ax2.plot(0, 0, 'c*', markersize=15, label='Radar')
    ax2.legend()
    plt.colorbar(im2, ax=ax2, label='Intensity')
    
    plt.tight_layout()
    plt.savefig('radar_projection_example.png', dpi=150)
    print("✓ Saved radar_projection_example.png")
    plt.show()
    
    # Example 3: Project onto camera view
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Overlay on Camera Image")
    print("=" * 60)
    
    # Simulate camera image
    camera_img = np.zeros((480, 640, 3), dtype=np.uint8)
    camera_img[:, :] = [50, 100, 50]  # Dark green background (road)
    
    # Simple camera projection (assuming camera co-located with radar, looking forward)
    # Camera intrinsics (example values)
    f_x, f_y = 500, 500  # Focal length (pixels)
    c_x, c_y = 320, 400  # Principal point
    
    # Project road coordinates to image
    valid = ~np.isnan(x_road)
    x_valid = x_road[valid]
    y_valid = y_road[valid]
    i_valid = intensity[valid]
    
    # Simple perspective projection (pinhole camera model)
    u = (f_x * (x_valid / y_valid) + c_x).astype(int)
    v = (f_y * (-config.height / y_valid) + c_y).astype(int)  # -height because road is below camera
    
    # Draw on image
    in_bounds = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    u_draw = u[in_bounds]
    v_draw = v[in_bounds]
    i_draw = i_valid[in_bounds]
    
    # Color map: blue (low) to red (high)
    for uu, vv, ii in zip(u_draw, v_draw, i_draw):
        color = (0, int(255 * (1 - ii)), int(255 * ii))  # BGR: blue to red
        cv2.circle(camera_img, (uu, vv), 3, color, -1)
    
    cv2.imwrite('radar_overlay_on_camera.png', camera_img)
    print("✓ Saved radar_overlay_on_camera.png")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("PROJECTION STATISTICS")
    print("=" * 60)
    print(f"Total radar bins: {heatmap.size}")
    print(f"Valid road projections: {np.sum(~np.isnan(x_road))}")
    print(f"Coverage area: {np.sum(~np.isnan(x_road)) * config.projection_resolution**2:.2f} m²")
    print(f"X range: [{np.nanmin(x_road):.2f}, {np.nanmax(x_road):.2f}] m")
    print(f"Y range: [{np.nanmin(y_road):.2f}, {np.nanmax(y_road):.2f}] m")
