"""
spatial_features.py
Geometric and spatial features of a segmented region.
"""

import cv2
import numpy as np


class SpatialFeatureExtractor:

    def extract(self, mask: np.ndarray,
                image_shape: tuple = None) -> dict:
        """
        Args:
            mask:         H×W boolean region mask
            image_shape:  (H, W) of the full image, used for relative position features
        Returns:
            Flat dict of float features.
        """
        area = int(np.sum(mask))
        if area == 0:
            return self._zeros()

        H, W = mask.shape
        img_H, img_W = image_shape if image_shape else (H, W)

        out = {}
        out['spatial_area']          = float(area)
        out['spatial_area_fraction'] = float(area / (img_H * img_W))

        # Bounding box
        rows, cols = np.where(mask)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        bb_h = r1 - r0 + 1
        bb_w = c1 - c0 + 1
        out['spatial_bbox_h']            = float(bb_h)
        out['spatial_bbox_w']            = float(bb_w)
        out['spatial_aspect_ratio']      = float(bb_w / (bb_h + 1e-8))
        out['spatial_bbox_fill']         = float(area / (bb_h * bb_w + 1e-8))

        # Centroid and relative position in frame
        cy = float(np.mean(rows)) / img_H
        cx = float(np.mean(cols)) / img_W
        out['spatial_centroid_y_rel'] = cy
        out['spatial_centroid_x_rel'] = cx
        # Distance from image centre (0 = centre, 1 = corner)
        out['spatial_dist_from_center'] = float(np.sqrt((cy - 0.5)**2 + (cx - 0.5)**2))

        # Contour-based features
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, True))

            # Compactness / circularity: 1 = perfect circle, <1 = elongated/jagged
            out['spatial_compactness'] = float(
                4 * np.pi * area / (perimeter ** 2 + 1e-8)
            )

            # Convex hull fill ratio: low = concave / irregular boundary (puddle-like)
            hull        = cv2.convexHull(contour)
            hull_area   = float(cv2.contourArea(hull))
            out['spatial_convexity']  = float(area / (hull_area + 1e-8))
            out['spatial_solidity']   = out['spatial_convexity']   # alias

            # Elongation from fitted ellipse (if enough points)
            if len(contour) >= 5:
                _, (ma, Ma), _ = cv2.fitEllipse(contour)
                out['spatial_elongation'] = float(ma / (Ma + 1e-8))
            else:
                out['spatial_elongation'] = 1.0

            # Boundary smoothness: ratio of convex hull perimeter to actual perimeter
            hull_perimeter = float(cv2.arcLength(hull, True))
            out['spatial_boundary_smoothness'] = float(
                hull_perimeter / (perimeter + 1e-8)
            )

            # Number of contour fragments (1 = solid patch, >1 = broken/cracked)
            out['spatial_n_fragments'] = float(len(contours))

        else:
            out.update({
                'spatial_compactness':         0.0,
                'spatial_convexity':           0.0,
                'spatial_solidity':            0.0,
                'spatial_elongation':          1.0,
                'spatial_boundary_smoothness': 0.0,
                'spatial_n_fragments':         0.0,
            })

        # Perimeter-to-area ratio (high = lacy / cracked edges)
        perimeter = out.get('spatial_compactness', 0)
        out['spatial_perimeter_area_ratio'] = float(
            perimeter / (area + 1e-8)
        ) if perimeter else 0.0

        # Vertical position within frame (drainage flows downward)
        out['spatial_vertical_position'] = cy   # 0=top, 1=bottom

        return out

    def _zeros(self) -> dict:
        dummy = np.zeros((8, 8), dtype=bool)
        dummy[2:6, 2:6] = True
        return {k: 0.0 for k in self.extract(dummy, (8, 8))}
