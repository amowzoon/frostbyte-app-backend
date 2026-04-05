import os
import cv2
import numpy as np

CALIBRATION_DIR = os.environ.get("CALIBRATION_DIR", "/app/calibration")
HOMOGRAPHY_PATH = os.path.join(CALIBRATION_DIR, "ir_rgb_homography.npy")


class IRRGBMapper:
    def __init__(self, theta_deg, h_mount, gamma_baseline):
        """
        Initialize IR to RGB mapper for road surface viewing

        Args:
            theta_deg: Camera tilt angle from vertical (degrees)
            h_mount: Mounting height above road (meters)
            gamma_baseline: Horizontal distance between camera centers (meters)
        """
        self.theta = theta_deg
        self.h = h_mount
        self.gamma = gamma_baseline

        # IR camera parameters (FLIR Lepton 3.1R)
        self.ir_full_width = 160
        self.ir_full_height = 120
        self.ir_h_fov_full = 95  # degrees
        self.ir_diag_fov = 119  # degrees
        self.ir_pixel_size = 12e-6  # meters (12 μm)

        # RGB camera parameters (Arducam Pi V3 - IMX708)
        self.rgb_width = 4608
        self.rgb_height = 2592
        self.rgb_h_fov = 66  # degrees
        self.rgb_pixel_size = 1.4e-6  # meters (1.4 μm)

        # Calculate IR V-FoV from diagonal FoV
        self.ir_v_fov_full = self._calculate_ir_v_fov()

        # Calculate RGB V-FoV
        self.rgb_v_fov = self._calculate_rgb_v_fov()

        # Calculate cropped IR dimensions to match RGB FoV
        self.ir_cropped_width = int(self.ir_full_width * self.rgb_h_fov / self.ir_h_fov_full)
        self.ir_cropped_height = int(self.ir_full_height * self.rgb_v_fov / self.ir_v_fov_full)

        # Calculate crop offsets (center crop)
        self.ir_crop_left = (self.ir_full_width - self.ir_cropped_width) // 2
        self.ir_crop_right = self.ir_crop_left + self.ir_cropped_width
        self.ir_crop_top = (self.ir_full_height - self.ir_cropped_height) // 2
        self.ir_crop_bottom = self.ir_crop_top + self.ir_cropped_height

        # Compute intrinsics for cropped IR
        self.K_ir = self._compute_ir_intrinsics()
        self.K_rgb = self._compute_rgb_intrinsics()

        # Compute geometric (ground-plane) homography
        self.H = self._compute_homography()

        # ── Empirical calibration ─────────────────────────────────────────────
        # If a saved homography exists from the calibration UI, use it instead.
        # It maps from full IR native pixels (160×120) directly to RGB pixels,
        # so crop_ir_image() is bypassed when this is active.
        self.H_empirical = None
        self._load_empirical_calibration()
        # ─────────────────────────────────────────────────────────────────────

        # Pre-compute mapping for efficiency (uses whichever H is active)
        self.map_x, self.map_y = self._create_mapping()

        # Print computed parameters for verification
        self._print_camera_info()

    # ── Empirical calibration ─────────────────────────────────────────────────

    def _load_empirical_calibration(self):
        """Load saved homography from calibration UI, if present."""
        if os.path.exists(HOMOGRAPHY_PATH):
            try:
                self.H_empirical = np.load(HOMOGRAPHY_PATH)
                print(f"[IRRGBMapper] Loaded empirical calibration: {HOMOGRAPHY_PATH}")
            except Exception as e:
                print(f"[IRRGBMapper] Failed to load empirical calibration: {e}")
                self.H_empirical = None
        else:
            print(f"[IRRGBMapper] No empirical calibration found, using geometric model")

    def reload_calibration(self):
        """
        Hot-reload the empirical calibration without reinstantiating the mapper.
        Call this after saving a new calibration from the UI.
        """
        self._load_empirical_calibration()
        self.map_x, self.map_y = self._create_mapping()

    @property
    def using_empirical(self) -> bool:
        """True if the empirical (calibrated) homography is active."""
        return self.H_empirical is not None

    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_ir_v_fov(self):
        """Calculate IR vertical FoV from diagonal FoV and aspect ratio"""
        sensor_w = self.ir_full_width * self.ir_pixel_size
        sensor_h = self.ir_full_height * self.ir_pixel_size
        sensor_diag = np.sqrt(sensor_w**2 + sensor_h**2)
        f = (sensor_diag / 2) / np.tan(np.radians(self.ir_diag_fov / 2))
        v_fov = 2 * np.degrees(np.arctan(sensor_h / (2 * f)))
        return v_fov

    def _calculate_rgb_v_fov(self):
        """Calculate RGB vertical FoV from horizontal FoV and aspect ratio"""
        sensor_w = self.rgb_width * self.rgb_pixel_size
        sensor_h = self.rgb_height * self.rgb_pixel_size
        f = (sensor_w / 2) / np.tan(np.radians(self.rgb_h_fov / 2))
        v_fov = 2 * np.degrees(np.arctan(sensor_h / (2 * f)))
        return v_fov

    def _compute_ir_intrinsics(self):
        """Compute IR camera intrinsic matrix for cropped region"""
        sensor_width = self.ir_cropped_width * self.ir_pixel_size
        fx_mm = (sensor_width / 2) / np.tan(np.radians(self.rgb_h_fov / 2))
        fx_px = fx_mm / self.ir_pixel_size
        fy_px = fx_px
        cx = self.ir_cropped_width / 2.0
        cy = self.ir_cropped_height / 2.0
        K = np.array([[fx_px, 0, cx], [0, fy_px, cy], [0, 0, 1]])
        return K

    def _compute_rgb_intrinsics(self):
        """Compute RGB camera intrinsic matrix"""
        sensor_width = self.rgb_width * self.rgb_pixel_size
        fx_mm = (sensor_width / 2) / np.tan(np.radians(self.rgb_h_fov / 2))
        fx_px = fx_mm / self.rgb_pixel_size
        fy_px = fx_px
        cx = self.rgb_width / 2.0
        cy = self.rgb_height / 2.0
        K = np.array([[fx_px, 0, cx], [0, fy_px, cy], [0, 0, 1]])
        return K

    def _compute_homography(self):
        """Compute homography from IR to RGB for ground plane"""
        theta_rad = np.radians(self.theta)
        n = np.array([[0], [-np.sin(theta_rad)], [np.cos(theta_rad)]])
        d = self.h / np.cos(theta_rad)
        R = np.eye(3)
        t = np.array([[self.gamma], [0], [0]])
        H = self.K_rgb @ (R - (t @ n.T) / d) @ np.linalg.inv(self.K_ir)
        H = H / H[2, 2]
        return H

    def _create_mapping(self):
        """Pre-compute pixel mapping from IR to RGB"""
        if self.using_empirical:
            # Empirical H maps from full IR native space (160×120)
            ir_y, ir_x = np.mgrid[0:self.ir_full_height, 0:self.ir_full_width].astype(np.float32)
            ir_coords = np.stack([ir_x, ir_y], axis=-1).reshape(-1, 2)
            rgb_coords = cv2.perspectiveTransform(
                ir_coords.reshape(-1, 1, 2), self.H_empirical
            ).reshape(self.ir_full_height, self.ir_full_width, 2)
        else:
            # Geometric H maps from cropped IR space
            ir_y, ir_x = np.mgrid[0:self.ir_cropped_height, 0:self.ir_cropped_width].astype(np.float32)
            ir_coords = np.stack([ir_x, ir_y], axis=-1).reshape(-1, 2)
            rgb_coords = cv2.perspectiveTransform(
                ir_coords.reshape(-1, 1, 2), self.H
            ).reshape(self.ir_cropped_height, self.ir_cropped_width, 2)

        map_x = rgb_coords[:, :, 0]
        map_y = rgb_coords[:, :, 1]
        return map_x, map_y

    def _print_camera_info(self):
        """Print computed camera parameters for verification"""
        print("=" * 70)
        print("IR Camera (FLIR Lepton 3.1R) - FULL SENSOR")
        print("=" * 70)
        print(f"Full Resolution: {self.ir_full_height} × {self.ir_full_width}")
        print(f"Full FoV: H={self.ir_h_fov_full:.1f}° × V={self.ir_v_fov_full:.1f}°")
        print()

        print("=" * 70)
        print("IR Camera (FLIR Lepton 3.1R) - CROPPED TO MATCH RGB FoV")
        print("=" * 70)
        print(f"Cropped Resolution: {self.ir_cropped_height} × {self.ir_cropped_width}")
        print(f"Crop region: rows [{self.ir_crop_top}:{self.ir_crop_bottom}], cols [{self.ir_crop_left}:{self.ir_crop_right}]")
        print(f"Pixel size: {self.ir_pixel_size * 1e6:.1f} μm × {self.ir_pixel_size * 1e6:.1f} μm")
        print(f"Cropped sensor: {self.ir_cropped_width * self.ir_pixel_size * 1e3:.3f} mm × {self.ir_cropped_height * self.ir_pixel_size * 1e3:.3f} mm")
        print(f"Focal length: {self.K_ir[0, 0] * self.ir_pixel_size * 1e3:.3f} mm")
        print(f"  fx = {self.K_ir[0, 0]:.2f} pixels")
        print(f"  fy = {self.K_ir[1, 1]:.2f} pixels")
        print(f"Principal point: ({self.K_ir[0, 2]:.1f}, {self.K_ir[1, 2]:.1f})")
        print(f"Matched FoV: H={self.rgb_h_fov:.1f}° × V={self.rgb_v_fov:.1f}°")
        print()

        print("=" * 70)
        print("RGB Camera (Arducam Pi V3 - IMX708)")
        print("=" * 70)
        print(f"Resolution: {self.rgb_height} × {self.rgb_width}")
        print(f"Pixel size: {self.rgb_pixel_size * 1e6:.1f} μm × {self.rgb_pixel_size * 1e6:.1f} μm")
        print(f"Sensor size: {self.rgb_width * self.rgb_pixel_size * 1e3:.3f} mm × {self.rgb_height * self.rgb_pixel_size * 1e3:.3f} mm")
        print(f"Focal length: {self.K_rgb[0, 0] * self.rgb_pixel_size * 1e3:.3f} mm")
        print(f"  fx = {self.K_rgb[0, 0]:.2f} pixels")
        print(f"  fy = {self.K_rgb[1, 1]:.2f} pixels")
        print(f"Principal point: ({self.K_rgb[0, 2]:.1f}, {self.K_rgb[1, 2]:.1f})")
        print(f"Field of View: H={self.rgb_h_fov:.1f}° × V={self.rgb_v_fov:.1f}°")
        print()

        print("=" * 70)
        print("FoV Matching Summary")
        print("=" * 70)
        print(f"✓ Horizontal FoV matched: {self.rgb_h_fov:.1f}°")
        print(f"✓ Vertical FoV matched: {self.rgb_v_fov:.1f}°")
        print(f"  Cropped {self.ir_full_width - self.ir_cropped_width} pixels horizontally ({(1 - self.ir_cropped_width/self.ir_full_width)*100:.1f}% removed)")
        print(f"  Cropped {self.ir_full_height - self.ir_cropped_height} pixels vertically ({(1 - self.ir_cropped_height/self.ir_full_height)*100:.1f}% removed)")
        print()

        print("=" * 70)
        print("Resolution Ratio")
        print("=" * 70)
        print(f"Each IR pixel maps to approximately:")
        print(f"  {self.rgb_width / self.ir_cropped_width:.1f} × {self.rgb_height / self.ir_cropped_height:.1f} RGB pixels")
        print(f"  Total: ~{(self.rgb_width / self.ir_cropped_width) * (self.rgb_height / self.ir_cropped_height):.0f} RGB pixels per IR pixel")
        print("=" * 70)
        print()

        # ── Calibration status line ───────────────────────────────────────────
        if self.using_empirical:
            print("  Mapping mode : EMPIRICAL  (loaded from calibration UI)")
        else:
            print("  Mapping mode : GEOMETRIC  (ground-plane homography)")
        print("=" * 70)
        print()

    def crop_ir_image(self, ir_full):
        """
        Crop IR image to match RGB FoV.
        When using empirical calibration this is a no-op — the full native
        image is passed directly to warp_ir_to_rgb.

        Args:
            ir_full: Full IR image (120 × 160)

        Returns:
            Cropped IR image (geometric mode) or the original array (empirical mode)
        """
        if self.using_empirical:
            return ir_full  # empirical H already accounts for the full sensor
        return ir_full[self.ir_crop_top:self.ir_crop_bottom,
                       self.ir_crop_left:self.ir_crop_right]

    def warp_ir_to_rgb(self, ir_input):
        """
        Warp IR image to RGB coordinate space.

        Uses the empirical homography (calibration UI) when available,
        otherwise falls back to the geometric ground-plane homography.
        BORDER_REPLICATE ensures every RGB pixel gets a valid temperature
        rather than a zero-padding artifact.

        Args:
            ir_input: IR image — full 120×160 in empirical mode,
                      cropped in geometric mode.

        Returns:
            IR image warped to RGB space (rgb_height × rgb_width), float32 Celsius.
        """
        H_active = self.H_empirical if self.using_empirical else self.H

        return cv2.warpPerspective(
            ir_input,
            H_active,
            (self.rgb_width, self.rgb_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE   # no more 0°C padding
        )

    def aggregate_by_regions(self, ir_cropped, segmented_rgb):
        """
        Aggregate IR measurements by RGB texture regions.

        Args:
            ir_cropped: Output of crop_ir_image()
            segmented_rgb: Segmented RGB image with region labels (2592 × 4608)

        Returns:
            Dictionary mapping region_id -> thermal statistics
        """
        ir_warped = self.warp_ir_to_rgb(ir_cropped)

        region_stats = {}
        unique_regions = np.unique(segmented_rgb)

        for region_id in unique_regions:
            if region_id == 0:
                continue
            region_mask = (segmented_rgb == region_id)
            ir_values   = ir_warped[region_mask]
            valid_ir    = ir_values[ir_values > 0]
            if len(valid_ir) == 0:
                continue
            region_stats[region_id] = {
                'mean':     float(np.mean(valid_ir)),
                'median':   float(np.median(valid_ir)),
                'std':      float(np.std(valid_ir)),
                'min':      float(np.min(valid_ir)),
                'max':      float(np.max(valid_ir)),
                'count':    int(len(valid_ir)),
                'coverage': float(len(valid_ir) / np.sum(region_mask))
            }

        return region_stats

    def get_ir_pixel_in_rgb_space(self, ir_y, ir_x):
        """
        Get RGB coordinates for a specific IR pixel.

        Args:
            ir_y, ir_x: Coordinates in the IR image (cropped or full,
                        depending on mode — use crop_ir_image() output coords).

        Returns:
            (rgb_y, rgb_x) or (None, None) if outside bounds
        """
        h = self.ir_full_height if self.using_empirical else self.ir_cropped_height
        w = self.ir_full_width  if self.using_empirical else self.ir_cropped_width

        if 0 <= ir_y < h and 0 <= ir_x < w:
            rgb_x = self.map_x[ir_y, ir_x]
            rgb_y = self.map_y[ir_y, ir_x]
            if 0 <= rgb_x < self.rgb_width and 0 <= rgb_y < self.rgb_height:
                return int(rgb_y), int(rgb_x)

        return None, None

    def visualize_mapping(self, ir_cropped, rgb_image, alpha=0.4):
        """
        Create visualization showing IR overlay on RGB.

        Args:
            ir_cropped: Output of crop_ir_image()
            rgb_image: RGB image
            alpha: Transparency of IR overlay (0-1)

        Returns:
            Visualization image with IR overlay
        """
        ir_warped  = self.warp_ir_to_rgb(ir_cropped)
        ir_norm    = cv2.normalize(ir_warped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ir_colored = cv2.applyColorMap(ir_norm, cv2.COLORMAP_JET)
        mask       = (ir_warped > 0).astype(np.uint8) * 255
        mask_3ch   = cv2.merge([mask, mask, mask])
        rgb_vis    = rgb_image.copy()
        rgb_vis    = np.where(mask_3ch > 0,
                              cv2.addWeighted(rgb_vis, 1 - alpha, ir_colored, alpha, 0),
                              rgb_vis)
        return rgb_vis

    def get_coverage_stats(self, ir_cropped):
        """
        Get statistics about IR coverage in RGB space.

        Args:
            ir_cropped: Output of crop_ir_image()

        Returns:
            Dictionary with coverage statistics
        """
        ir_warped        = self.warp_ir_to_rgb(ir_cropped)
        total_rgb_pixels = self.rgb_width * self.rgb_height
        valid_ir_pixels  = np.sum(ir_warped > 0)

        return {
            'total_rgb_pixels':    total_rgb_pixels,
            'ir_covered_pixels':   int(valid_ir_pixels),
            'coverage_percentage': float(valid_ir_pixels / total_rgb_pixels * 100),
            'ir_resolution':       (self.ir_cropped_height, self.ir_cropped_width),
            'rgb_resolution':      (self.rgb_height, self.rgb_width),
            'mode':                'empirical' if self.using_empirical else 'geometric',
        }


# Usage Example
if __name__ == "__main__":
    theta = 30
    h = 1.5
    gamma = 0.05

    mapper = IRRGBMapper(theta, h, gamma)
    print(f"Using empirical calibration: {mapper.using_empirical}")

    ir_full = np.random.rand(120, 160).astype(np.float32) * 100
    rgb_image = np.random.randint(0, 255, (2592, 4608, 3), dtype=np.uint8)
    segmented_rgb = np.random.randint(0, 10, (2592, 4608), dtype=np.int32)

    ir_cropped    = mapper.crop_ir_image(ir_full)
    thermal_stats = mapper.aggregate_by_regions(ir_cropped, segmented_rgb)

    print("\nThermal Statistics by Region:")
    for region_id, stats in thermal_stats.items():
        print(f"Region {region_id}: mean={stats['mean']:.2f}°C  std={stats['std']:.2f}°C  coverage={stats['coverage']*100:.1f}%")

    coverage = mapper.get_coverage_stats(ir_cropped)
    print(f"\nIR Coverage: {coverage['coverage_percentage']:.1f}% ({coverage['mode']} mode)")
