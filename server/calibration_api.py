"""
IR-RGB Homography Calibration API
Receives point correspondences from the calibration UI,
computes findHomography, saves the matrix to disk.
"""

import json
import logging
import os
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional

log = logging.getLogger("app.calibration")

calibration_router = APIRouter(prefix="/api/calibration", tags=["calibration"])

# Where the homography matrix is persisted
CALIBRATION_DIR  = os.environ.get("CALIBRATION_DIR", "/app/calibration")
HOMOGRAPHY_PATH  = os.path.join(CALIBRATION_DIR, "ir_rgb_homography.npy")
METADATA_PATH    = os.path.join(CALIBRATION_DIR, "ir_rgb_homography_meta.json")

os.makedirs(CALIBRATION_DIR, exist_ok=True)


# ── Models ────────────────────────────────────────────────────────────────────

class PointPair(BaseModel):
    ir_x:  float
    ir_y:  float
    rgb_x: float
    rgb_y: float


class ComputeRequest(BaseModel):
    session_id:         str
    pairs:              List[PointPair]
    ir_natural_width:   int
    ir_natural_height:  int
    rgb_natural_width:  int
    rgb_natural_height: int


class SaveRequest(BaseModel):
    homography:  List[List[float]]   # 3x3 matrix as nested list
    pair_count:  int
    session_id:  str
    pairs:       List[PointPair]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_minio_image(s3_path: str) -> np.ndarray:
    """Load an image from MinIO into a numpy array."""
    # Import here to avoid circular imports
    from main import minio_client, BUCKET_NAME
    response = minio_client.get_object(BUCKET_NAME, s3_path)
    data = response.read()
    response.close()
    response.release_conn()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {s3_path}")
    return img


def _build_preview(ir_img_bgr: np.ndarray, rgb_img_bgr: np.ndarray, H: np.ndarray) -> bytes:
    """
    Warp the IR image onto the RGB using H, blend, return JPEG bytes.

    H maps from native IR sensor space (160×120) to native RGB space (4608×2592).
    So we must:
      1. Resize the IR JPEG down to native 160×120 before warping.
      2. Warp into full RGB resolution using H as-is.
      3. Downscale the result to a preview size for the response.
    """
    IR_NATIVE_W, IR_NATIVE_H = 160, 120
    MAX_SIDE = 1200

    # Step 1: resize IR JPEG → native sensor resolution so H applies correctly
    ir_native = cv2.resize(ir_img_bgr, (IR_NATIVE_W, IR_NATIVE_H), interpolation=cv2.INTER_AREA)

    # Step 2: warp IR native → full RGB space using H unchanged
    rh, rw = rgb_img_bgr.shape[:2]
    ir_warped_full = cv2.warpPerspective(
        ir_native, H, (rw, rh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Step 3: downscale both to preview size
    scale = min(MAX_SIDE / rw, MAX_SIDE / rh, 1.0)
    pw    = int(rw * scale)
    ph    = int(rh * scale)

    rgb_small = cv2.resize(rgb_img_bgr,   (pw, ph), interpolation=cv2.INTER_AREA)
    ir_warped = cv2.resize(ir_warped_full, (pw, ph), interpolation=cv2.INTER_AREA)

    # Colorise the IR (apply inferno colormap)
    ir_gray   = cv2.cvtColor(ir_warped, cv2.COLOR_BGR2GRAY)
    ir_color  = cv2.applyColorMap(ir_gray, cv2.COLORMAP_INFERNO)

    # Blend
    valid_mask = (ir_gray > 0).astype(np.uint8)
    valid_3ch  = cv2.merge([valid_mask * 255] * 3)
    blended    = cv2.addWeighted(rgb_small, 0.5, ir_color, 0.5, 0)
    preview    = np.where(valid_3ch > 0, blended, rgb_small)

    _, buf = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


# ── Routes ────────────────────────────────────────────────────────────────────

@calibration_router.post("/compute")
async def compute_homography(req: ComputeRequest):
    """
    Given ≥4 IR→RGB point correspondences, compute findHomography (RANSAC),
    generate a preview overlay, return the 3×3 matrix and reprojection error.

    Points are in the natural (native) pixel space of each image —
    the UI sends raw image coordinates, not display-scaled coordinates.
    """
    if len(req.pairs) < 4:
        raise HTTPException(400, f"Need at least 4 point pairs, got {len(req.pairs)}")

    src_pts = np.float32([[p.ir_x,  p.ir_y]  for p in req.pairs])
    dst_pts = np.float32([[p.rgb_x, p.rgb_y] for p in req.pairs])

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=8.0)

    if H is None:
        raise HTTPException(422, "findHomography failed — points may be collinear or degenerate")

    # Reprojection error on inliers
    inlier_mask = mask.ravel().astype(bool)
    n_inliers   = inlier_mask.sum()

    projected   = cv2.perspectiveTransform(src_pts[inlier_mask].reshape(-1, 1, 2), H)
    projected   = projected.reshape(-1, 2)
    target      = dst_pts[inlier_mask]
    errors      = np.linalg.norm(projected - target, axis=1)
    mean_error  = float(errors.mean()) if len(errors) else 0.0

    log.info(
        f"Homography computed: {len(req.pairs)} pairs, {n_inliers} inliers, "
        f"reprojection error={mean_error:.2f}px"
    )

    # Build preview by fetching actual images from MinIO
    preview_url = None
    try:
        from main import minio_client, BUCKET_NAME
        import psycopg2
        from main import get_db

        # Fetch the IR and RGB s3 paths for this session
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            SELECT sensor_type, s3_path FROM sensor_data
            WHERE session_id = %s AND sensor_type IN ('ir', 'rgb')
        """, (req.session_id,))
        rows = {r[0]: r[1] for r in cur.fetchall()}
        cur.close()
        conn.close()

        if 'ir' in rows and 'rgb' in rows:
            ir_bgr  = _load_minio_image(rows['ir'])
            rgb_bgr = _load_minio_image(rows['rgb'])

            preview_bytes = _build_preview(ir_bgr, rgb_bgr, H)

            # Save preview to MinIO under calibration/
            preview_path = f"calibration/preview_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg"
            minio_client.put_object(
                BUCKET_NAME, preview_path,
                BytesIO(preview_bytes), len(preview_bytes),
                content_type="image/jpeg"
            )
            preview_url = f"/api/images/{preview_path}"
            log.info(f"Preview saved: {preview_path}")

    except Exception as e:
        log.warning(f"Preview generation failed (non-fatal): {e}")

    return {
        "status":             "ok",
        "homography":         H.tolist(),
        "n_pairs":            len(req.pairs),
        "n_inliers":          int(n_inliers),
        "reprojection_error": mean_error,
        "preview_url":        preview_url,
    }


@calibration_router.post("/save")
async def save_calibration(req: SaveRequest):
    """
    Persist the homography matrix and metadata to disk.
    IRRGBMapper will load this on next mapping call.
    """
    H = np.array(req.homography, dtype=np.float64)
    if H.shape != (3, 3):
        raise HTTPException(400, f"Homography must be 3×3, got {H.shape}")

    np.save(HOMOGRAPHY_PATH, H)

    meta = {
        "saved_at":   datetime.utcnow().isoformat(),
        "session_id": req.session_id,
        "pair_count": req.pair_count,
        "pairs": [
            {"ir_x": p.ir_x, "ir_y": p.ir_y, "rgb_x": p.rgb_x, "rgb_y": p.rgb_y}
            for p in req.pairs
        ],
        "homography": H.tolist(),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Calibration saved: {HOMOGRAPHY_PATH} ({req.pair_count} pairs)")

    return {
        "status":     "ok",
        "path":       HOMOGRAPHY_PATH,
        "pair_count": req.pair_count,
    }


@calibration_router.get("/homography")
async def get_homography():
    """
    Return the current saved homography matrix, if one exists.
    Used by IRRGBMapper and by the UI to show calibration status.
    """
    if not os.path.exists(HOMOGRAPHY_PATH):
        return {"exists": False}

    H = np.load(HOMOGRAPHY_PATH)
    meta = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            meta = json.load(f)

    return {
        "exists":     True,
        "homography": H.tolist(),
        "pair_count": meta.get("pair_count"),
        "saved_at":   meta.get("saved_at"),
        "session_id": meta.get("session_id"),
    }


@calibration_router.delete("/homography")
async def delete_calibration():
    """Delete the saved calibration, reverting to geometric model."""
    removed = []
    for path in [HOMOGRAPHY_PATH, METADATA_PATH]:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    return {"status": "ok", "removed": removed}
