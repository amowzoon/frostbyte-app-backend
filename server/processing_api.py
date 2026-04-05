"""
processing_api.py
FastAPI router for the ice detection processing pipeline.
Mount this into main.py with: app.include_router(processing_router)
"""

import sys
import os
import json
import logging
import numpy as np
import cv2
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

# road isolation imports
import glob
import torch
from torch.utils.data import Dataset, DataLoader

try:
    sys.path.append(str(Path(__file__).parent / "processing" / "MobileSAM"))
    from mobile_sam import build_sam, sam_model_registry
    from processing.road_isolation_backend.road_isolation_dataloader import AcclimateWeatherDataset
    from processing.road_isolation_backend.segmentation_head import SegmentationHead
    from processing.road_isolation_backend.inference_road_isolation import road_filter_inference
    ROAD_ISOLATION_AVAILABLE = True
except Exception as e:
    print(f"Warning: Road isolation modules not available: {e}")
    ROAD_ISOLATION_AVAILABLE = False

import torch.nn as nn
import torch.nn.functional as F

# Add processing directory to path
sys.path.insert(0, str(Path(__file__).parent / "processing"))

log = logging.getLogger("app.processing")

processing_router = APIRouter(prefix="/api/processing", tags=["processing"])

# Absolute path to this file's directory — used for model paths
_HERE = Path(__file__).parent


# ─── Lazy imports ─────────────────────────────────────────────────────────────
def _import_processing():
    try:
        from ir_to_rgb_mapping import IRRGBMapper
        from radar_projection import RadarProjector
        from segment_by_texture import RegionSegmenter, SegmentationConfig
        from feature_extraction import FeatureExtractor, TemporalConfig
        from pipeline import HeuristicInferencePipeline, InferencePipeline, ModelConfig
        from heuristic_model import HeuristicWeights, HeuristicIceDetector, PresetWeights
        return {
            'IRRGBMapper': IRRGBMapper,
            'RadarProjector': RadarProjector,
            'RegionSegmenter': RegionSegmenter,
            'SegmentationConfig': SegmentationConfig,
            'FeatureExtractor': FeatureExtractor,
            'TemporalConfig': TemporalConfig,
            'HeuristicInferencePipeline': HeuristicInferencePipeline,
            'InferencePipeline': InferencePipeline,
            'ModelConfig': ModelConfig,
            'HeuristicWeights': HeuristicWeights,
        }
    except ImportError as e:
        log.warning(f"Processing modules not fully available: {e}")
        return None


# ─── In-memory session cache ──────────────────────────────────────────────────
_session_cache: Dict[str, Dict[str, Any]] = {}


def _cache_set(session_id: str, key: str, value: Any):
    if session_id not in _session_cache:
        _session_cache[session_id] = {}
    _session_cache[session_id][key] = value


def _cache_get(session_id: str, key: str) -> Optional[Any]:
    return _session_cache.get(session_id, {}).get(key)

def _cache_clear(session_id: str):
    _session_cache.pop(session_id, None)

# ─── Pydantic models ──────────────────────────────────────────────────────────

class GeometryConfig(BaseModel):
    baseline: float = 0.12
    height: float = 1.5
    angle: float = 30.0

class SegmentationParams(BaseModel):
    min_distance: int = 90
    threshold_rel: float = 0.10
    lbp_r: int = 1
    lbp_p: int = 8

class EnsembleParams(BaseModel):
    use_logistic: bool = True
    use_naive_bayes: bool = True
    use_random_forest: bool = True
    gaussian_sigma: float = 2.0
    median_size: int = 5

class PipelineConfig(BaseModel):
    baseline: float = 0.12
    height: float = 1.5
    angle: float = 30.0
    airTemp: float = 5.0
    segmentation: SegmentationParams = SegmentationParams()
    model: str = "heuristic"
    weights: Optional[Dict[str, float]] = None
    ensemble: EnsembleParams = EnsembleParams()

class StepRequest(BaseModel):
    session_id: str
    config: PipelineConfig = PipelineConfig()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_minio_and_db():
    try:
        import main as app_main
        return app_main.minio_client, app_main.get_db, app_main.BUCKET_NAME
    except Exception as e:
        raise HTTPException(500, f"Cannot access app context: {e}")


def _fetch_sensor_data(session_id: str):
    _, get_db, _ = _get_minio_and_db()
    with get_db() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT sensor_type, s3_path, s3_path_raw, metadata
            FROM sensor_data
            WHERE session_id = %s
        """, (session_id,))
        rows = cur.fetchall()
        cur.close()

    log.info(f"_fetch_sensor_data({session_id!r}) -> {len(rows)} sensors: {[r[0] for r in rows]}")
    return {row[0]: {'s3_path': row[1], 's3_path_raw': row[2], 'metadata': row[3]} for row in rows}

def _load_ir_from_s3(s3_path: str, target_shape: tuple) -> np.ndarray:
    """
    Load IR data from S3.
    
    For .npy files: returns native-resolution float32 Celsius array (do NOT resize - 
    the IRRGBMapper needs native 120x160 to crop and warp correctly).
    
    For JPEG files: returns resized float32 Celsius array scaled to target_shape
    (lossy fallback — accurate temps require .npy).
    
    target_shape is only used for the JPEG fallback path.
    """
    minio_client, _, bucket = _get_minio_and_db()
    resp = minio_client.get_object(bucket, s3_path)
    data = resp.read()
    resp.close()
    resp.release_conn()

    if s3_path.endswith('.npy'):
        arr = np.load(BytesIO(data))
        
        # Raw radiometric uint16 from Lepton: values are Kelvin * 100
        # e.g. 29815 = 298.15K = 25°C. Real Celsius values are always < 1000.
        if arr.dtype == np.uint16 or arr.max() > 1000:
            ir_celsius = (arr.astype(np.float32) - 27315.0) / 100.0
        else:
            ir_celsius = arr.astype(np.float32)
        
        # Return at NATIVE resolution — IRRGBMapper.crop_ir_image() expects 120x160
        # Resizing happens after mapping in run_mapping()
        return ir_celsius

    else:
        # JPEG fallback: false-color image, no accurate temp data possible
        # Resize to target_shape so downstream code that bypasses the mapper still works
        h, w = target_shape[:2]
        buf = np.frombuffer(data, dtype=np.uint8)
        ir_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if ir_bgr is None:
            raise ValueError(f"Could not decode IR image from {s3_path}")
        ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Map 0-255 -> rough Celsius range. This is visualization-only quality.
        ir_celsius = ir_gray / 255.0 * 60.0 - 20.0
        return cv2.resize(ir_celsius, (w, h), interpolation=cv2.INTER_LINEAR)

def _load_image_from_s3(s3_path: str) -> np.ndarray:
    minio_client, _, bucket = _get_minio_and_db()
    resp = minio_client.get_object(bucket, s3_path)
    data = resp.read()
    resp.close()
    resp.release_conn()
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image from {s3_path}")
    return img


def _load_json_from_s3(s3_path: str) -> dict:
    minio_client, _, bucket = _get_minio_and_db()
    resp = minio_client.get_object(bucket, s3_path)
    data = json.loads(resp.read().decode('utf-8'))
    resp.close()
    resp.release_conn()
    return data


def _save_image_to_minio(img_bgr: np.ndarray, s3_path: str, quality: int = 90) -> str:
    minio_client, _, bucket = _get_minio_and_db()
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    data = buf.tobytes()
    minio_client.put_object(
        bucket_name=bucket,
        object_name=s3_path,
        data=BytesIO(data),
        length=len(data),
        content_type='image/jpeg'
    )
    return s3_path


def _s3_url(s3_path: str) -> str:
    return f"/api/images/{s3_path}"

def _ensure_rgb_bgr(session_id: str, sensor_data: dict) -> np.ndarray:
    # Always load fresh from S3 — never trust a cached value as a run's starting point
    if 'rgb' not in sensor_data:
        raise HTTPException(400, "RGB sensor data not available for this session")
    rgb_bgr = _load_image_from_s3(sensor_data['rgb']['s3_path'])
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    _cache_set(session_id, 'rgb_bgr', rgb_bgr)
    _cache_set(session_id, 'rgb', rgb)
    return rgb_bgr

def _ensure_rgb(session_id: str, sensor_data: dict) -> np.ndarray:
    """Return cached RGB array, deriving from BGR if needed."""
    rgb = _cache_get(session_id, 'rgb')
    if rgb is not None:
        return rgb
    rgb_bgr = _ensure_rgb_bgr(session_id, sensor_data)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    _cache_set(session_id, 'rgb', rgb)
    return rgb


def _ensure_ir_raw(session_id: str, sensor_data: dict, target_shape: tuple) -> Optional[np.ndarray]:
    ir_raw = _cache_get(session_id, 'ir_raw')
    if ir_raw is not None:
        # If mapping was skipped, ir_raw may still be native resolution — resize to match RGB
        h, w = target_shape[:2]
        if ir_raw.shape[0] != h or ir_raw.shape[1] != w:
            log.warning(f"[{session_id}] ir_raw shape {ir_raw.shape} != target {target_shape[:2]}, resizing")
            ir_raw = cv2.resize(ir_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            _cache_set(session_id, 'ir_raw', ir_raw)
        return ir_raw

    if 'ir' not in sensor_data:
        log.warning(f"[{session_id}] No IR sensor data available")
        return None

    ir_info = sensor_data['ir']
    ir_s3 = ir_info.get('s3_path_raw') or ir_info['s3_path']

    try:
        ir_native = _load_ir_from_s3(ir_s3, target_shape)
        h, w = target_shape[:2]
        # Resize to RGB resolution if needed (mapping was skipped)
        if ir_native.shape[0] != h or ir_native.shape[1] != w:
            ir_native = cv2.resize(ir_native, (w, h), interpolation=cv2.INTER_LINEAR)
        log.info(f"[{session_id}] IR raw loaded: shape={ir_native.shape} "
                 f"min={ir_native.min():.1f} max={ir_native.max():.1f} mean={ir_native.mean():.1f}C")
        _cache_set(session_id, 'ir_raw', ir_native)
        return ir_native
    except Exception as e:
        log.error(f"[{session_id}] IR raw load failed from {ir_s3}: {e}")
        return None

def _clean_features_df(features_df):
    """
    Ensure features_df has exactly one 'region_id' column and it's an integer.
    Handles the case where the extractor returns region_id as the index,
    as a column, or both (which causes a duplicate crash in inference).
    """
    import pandas as pd

    # If region_id is the index but not a column, promote it
    if features_df.index.name == 'region_id' and 'region_id' not in features_df.columns:
        features_df = features_df.reset_index()
    # If it's both the index and a column, just reset without keeping the index
    elif features_df.index.name == 'region_id' and 'region_id' in features_df.columns:
        features_df = features_df.reset_index(drop=True)
    # If neither index nor column, create a sequential one
    elif 'region_id' not in features_df.columns:
        features_df = features_df.reset_index().rename(columns={'index': 'region_id'})

    features_df['region_id'] = features_df['region_id'].astype(int)
    return features_df


# ─── ROAD ISOLATION ───────────────────────────────────────────────────────────

@processing_router.post("/road")
def run_road_isolation(req: StepRequest):
    try:
        sensor_data = _fetch_sensor_data(req.session_id)
        rgb_bgr = _ensure_rgb_bgr(req.session_id, sensor_data)
        rgb = _ensure_rgb(req.session_id, sensor_data)

        # Use absolute path so it works regardless of working directory
        checkpoint = str(_HERE / "processing" / "saved_models" / "road_segmentation_model_inference_float16.pth")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        road_mask, overlay = road_filter_inference(checkpoint, rgb, device=device, target_size=(1024, 1024))

        _cache_set(req.session_id, 'road_mask', road_mask)

        # Visualization
        mask_vis = rgb_bgr.copy()
        mask_overlay = np.zeros_like(rgb_bgr)
        mask_overlay[road_mask > 0] = [0, 200, 80]
        mask_vis = cv2.addWeighted(mask_vis, 0.65, mask_overlay, 0.35, 0)
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_vis, contours, -1, (0, 255, 100), 2)

        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        mask_path = f"processing/{req.session_id}/road_mask_{ts}.jpg"
        _save_image_to_minio(mask_vis, mask_path)

        road_pixels = int(np.sum(road_mask > 0))

        return {
            "status": "success",
            "mask_url": _s3_url(mask_path),
            "road_pixels": road_pixels,
            "road_fraction": round(road_pixels / road_mask.size, 3)
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Road isolation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Road isolation failed: {e}")


# ─── IR + RADAR MAPPING ───────────────────────────────────────────────────────

@processing_router.post("/mapping")
def run_mapping(req: StepRequest):
    mods = _import_processing()
    try:
        sensor_data = _fetch_sensor_data(req.session_id)
        cfg = req.config

        rgb_bgr = _ensure_rgb_bgr(req.session_id, sensor_data)
        h, w = rgb_bgr.shape[:2]

        result_paths = {}
        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')

        # ── IR mapping ────────────────────────────────────────────────────────
        if 'ir' in sensor_data:
            try:
                ir_info = sensor_data['ir']
                ir_s3_raw = ir_info.get('s3_path_raw') or ir_info['s3_path']

                # Load at native resolution (120x160) — mapper requires this
                ir_native = _load_ir_from_s3(ir_s3_raw, rgb_bgr.shape)
                log.info(f"IR native loaded: shape={ir_native.shape} "
                         f"min={ir_native.min():.1f} max={ir_native.max():.1f} "
                         f"mean={ir_native.mean():.1f}C")

                if mods and 'IRRGBMapper' in mods:
                    IRRGBMapper = mods['IRRGBMapper']
                    mapper = IRRGBMapper(
                        theta_deg=cfg.angle,
                        h_mount=cfg.height,
                        gamma_baseline=cfg.baseline
                    )
                    # Crop IR to match RGB FoV, then warp into RGB pixel space
                    ir_cropped = mapper.crop_ir_image(ir_native)
                    ir_warped  = mapper.warp_ir_to_rgb(ir_cropped)  # float32 Celsius, zeros where no data
                else:
                    # No mapper: simple resize
                    ir_warped = cv2.resize(ir_native, (w, h), interpolation=cv2.INTER_LINEAR)

                # Resize to exactly (h, w) if warp produced different dims
                if ir_warped.shape[:2] != (h, w):
                    ir_warped = cv2.resize(ir_warped, (w, h), interpolation=cv2.INTER_LINEAR)

                # Build valid-pixel mask — warpPerspective pads with zeros,
                # real temps are never exactly 0.0 so this is a clean separator
                valid_mask = (ir_warped != 0.0).astype(np.uint8)

                # Cache the warped Celsius array for feature extraction
                _cache_set(req.session_id, 'ir_raw', ir_warped)
                log.info(f"IR warped cached: shape={ir_warped.shape} "
                         f"min={ir_warped.min():.1f} max={ir_warped.max():.1f} "
                         f"mean={ir_warped.mean():.1f}C "
                         f"valid_px={valid_mask.sum()} / {valid_mask.size}")

                # Normalize using only valid pixel range for best colormap contrast
                valid_temps = ir_warped[valid_mask == 1]
                if len(valid_temps) > 0:
                    display_min = float(valid_temps.min())
                    display_max = float(valid_temps.max())
                else:
                    display_min, display_max = 15.0, 40.0

                ir_norm    = np.clip(
                    (ir_warped - display_min) / max(display_max - display_min, 1.0), 0, 1
                )
                ir_uint8   = (ir_norm * 255).astype(np.uint8)
                ir_colored = cv2.applyColorMap(ir_uint8, cv2.COLORMAP_INFERNO)

                # Blend IR over RGB only where valid IR pixels exist
                valid_3ch = cv2.merge([valid_mask * 255] * 3)
                blended   = cv2.addWeighted(rgb_bgr, 0.45, ir_colored, 0.55, 0)
                ir_vis    = np.where(valid_3ch > 0, blended, rgb_bgr)

                ir_path = f"processing/{req.session_id}/ir_mapped_{ts}.jpg"
                _save_image_to_minio(ir_vis, ir_path)
                result_paths['ir_mapped_url'] = _s3_url(ir_path)

            except Exception as e:
                log.warning(f"IR mapping failed: {e}", exc_info=True)
                result_paths['ir_mapped_url'] = None
        else:
            result_paths['ir_mapped_url'] = None

        # ── Radar mapping ─────────────────────────────────────────────────────
        if 'radar' in sensor_data:
            try:
                radar_json = _load_json_from_s3(sensor_data['radar']['s3_path'])
                _cache_set(req.session_id, 'radar_data', radar_json)
                radar_vis  = _generate_radar_rgb_overlay(radar_json, rgb_bgr)
                radar_path = f"processing/{req.session_id}/radar_mapped_{ts}.jpg"
                _save_image_to_minio(radar_vis, radar_path)
                result_paths['radar_mapped_url'] = _s3_url(radar_path)
            except Exception as e:
                log.warning(f"Radar mapping failed: {e}")
                result_paths['radar_mapped_url'] = None
        else:
            result_paths['radar_mapped_url'] = None

        return {"status": "success", **result_paths}

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Mapping failed: {e}", exc_info=True)
        raise HTTPException(500, f"Mapping failed: {e}")

def _generate_radar_rgb_overlay(radar_json: dict, rgb_bgr: np.ndarray) -> np.ndarray:
    import scipy.interpolate as spi

    TX, RX, BINS, ABINS = 2, 4, 256, 64
    RES = 0.04360212053571429
    VA = TX * RX

    azimuth_data = radar_json['azimuth_static']
    a = np.array([azimuth_data[i] + 1j * azimuth_data[i+1]
                  for i in range(0, len(azimuth_data), 2)])
    a = np.reshape(a, (BINS, VA))
    a = np.fft.fft(a, ABINS)
    a = np.abs(a)
    a = np.fft.fftshift(a, axes=(1,))

    range_depth = BINS * RES
    range_width = range_depth / 2
    grid_res = 200

    t = np.linspace(-np.pi/2, np.pi/2, ABINS)
    r = np.arange(BINS) * RES
    x = np.array([r]).T * np.sin(t)
    y = np.array([r]).T * np.cos(t)

    xi = np.linspace(-range_width, range_width, grid_res)
    yi = np.linspace(0, range_depth, grid_res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi = spi.griddata((x.ravel(), y.ravel()), a.ravel(), (xi_grid, yi_grid), method='linear')
    zi = np.nan_to_num(zi)
    zi_norm = (zi / zi.max() * 255).astype(np.uint8) if zi.max() > 0 else zi.astype(np.uint8)
    radar_color = cv2.applyColorMap(zi_norm, cv2.COLORMAP_JET)

    h2, w2 = rgb_bgr.shape[:2]
    radar_resized = cv2.resize(radar_color, (w2, h2))
    alpha_mask = (zi_norm > 30).astype(np.float32)
    alpha_mask = cv2.resize(alpha_mask, (w2, h2))

    result = rgb_bgr.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = (rgb_bgr[:, :, c] * (1 - alpha_mask * 0.6)
                           + radar_resized[:, :, c] * alpha_mask * 0.6)
    return result.astype(np.uint8)


# ─── SEGMENTATION ─────────────────────────────────────────────────────────────

@processing_router.post("/segment")
def run_segmentation(req: StepRequest):
    mods = _import_processing()
    if not mods:
        raise HTTPException(503, "Processing modules not available. Check server logs.")

    try:
        RegionSegmenter = mods['RegionSegmenter']
        SegmentationConfig = mods['SegmentationConfig']

        sensor_data = _fetch_sensor_data(req.session_id)
        rgb_bgr = _ensure_rgb_bgr(req.session_id, sensor_data)
        rgb = _ensure_rgb(req.session_id, sensor_data)

        # road_mask is optional — segmentation still works without it
        road_mask = _cache_get(req.session_id, 'road_mask')
        if road_mask is None:
            log.warning(f"[{req.session_id}] No road mask cached; segmenting full image. "
                        "Run Road Isolation first for best results.")

        seg_cfg = req.config.segmentation
        config = SegmentationConfig(
            P=seg_cfg.lbp_p,
            R=seg_cfg.lbp_r,
            min_distance=seg_cfg.min_distance,
            threshold_rel=seg_cfg.threshold_rel
        )

        segmenter = RegionSegmenter(config)
        segments = segmenter.segment_image(rgb, mask=road_mask)
        _cache_set(req.session_id, 'segments', segments)

        from skimage.segmentation import mark_boundaries
        num_regions = int(segments.max())

        colors = np.random.randint(40, 220, (num_regions + 1, 3))
        colors[0] = [20, 20, 30]
        colored = colors[segments].astype(np.uint8)
        colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        seg_vis = cv2.addWeighted(rgb_bgr, 0.55, colored_bgr, 0.45, 0)

        seg_labeled = seg_vis.copy()
        for region_id in range(1, num_regions + 1):
            region_pixels = np.argwhere(segments == region_id)
            if len(region_pixels) > 50:
                centroid = region_pixels.mean(axis=0).astype(int)
                cv2.putText(seg_labeled, str(region_id),
                            (centroid[1], centroid[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                            cv2.LINE_AA)

        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        seg_path = f"processing/{req.session_id}/segments_{ts}.jpg"
        _save_image_to_minio(seg_labeled, seg_path)

        return {
            "status": "success",
            "segment_overlay_url": _s3_url(seg_path),
            "num_regions": num_regions,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Segmentation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Segmentation failed: {e}")


# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────

@processing_router.post("/features")
def run_feature_extraction(req: StepRequest):
    mods = _import_processing()
    if not mods:
        raise HTTPException(503, "Processing modules not available")

    try:
        FeatureExtractor = mods['FeatureExtractor']
        TemporalConfig = mods['TemporalConfig']

        segments = _cache_get(req.session_id, 'segments')
        if segments is None:
            raise HTTPException(400, "Segmentation results not found. Run Segmentation first.")

        sensor_data = _fetch_sensor_data(req.session_id)
        rgb = _ensure_rgb(req.session_id, sensor_data)

        # Use the raw IR array, not the visualization blend
        ir_raw = _ensure_ir_raw(req.session_id, sensor_data, rgb.shape)
        log.info(f"IR raw going into extractor: {ir_raw if ir_raw is None else f'shape={ir_raw.shape} min={ir_raw.min():.1f} max={ir_raw.max():.1f}'}")
        radar_data = _cache_get(req.session_id, 'radar_data')

        extractor = FeatureExtractor(TemporalConfig())
        features_df = extractor.extract_features(
            RGB_image=rgb,
            IR_image=ir_raw,
            radar_scan=None,
            regions=segments,
            air_temp=req.config.airTemp,
            timestamp=datetime.utcnow().timestamp()
        )

        features_df = _clean_features_df(features_df)
        _cache_set(req.session_id, 'features_df', features_df)

        region_features = {}
        for _, row in features_df.iterrows():
            rid = int(row['region_id'])
            region_features[rid] = {
                k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                for k, v in row.items()
                if k != 'region_id' and not isinstance(v, (list, np.ndarray))
            }

        return {
            "status": "success",
            "num_regions": len(features_df),
            "region_features": region_features,
            "feature_names": list(features_df.columns)
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Feature extraction failed: {e}", exc_info=True)
        raise HTTPException(500, f"Feature extraction failed: {e}")


# ─── INFERENCE ────────────────────────────────────────────────────────────────

@processing_router.post("/inference")
def run_inference(req: StepRequest):
    mods = _import_processing()
    if not mods:
        raise HTTPException(503, "Processing modules not available")

    try:
        features_df = _cache_get(req.session_id, 'features_df')
        segments    = _cache_get(req.session_id, 'segments')

        if features_df is None:
            raise HTTPException(400, "Feature data not found. Run Feature Extraction first.")
        if segments is None:
            raise HTTPException(400, "Segmentation data not found. Run Segmentation first.")

        sensor_data = _fetch_sensor_data(req.session_id)
        rgb_bgr = _ensure_rgb_bgr(req.session_id, sensor_data)
        rgb = _ensure_rgb(req.session_id, sensor_data)

        cfg = req.config
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter, median_filter

        if cfg.model == 'heuristic':
            HeuristicWeights = mods['HeuristicWeights']
            try:
                from heuristic_model import HeuristicIceDetector
            except ImportError:
                raise HTTPException(503, "heuristic_model module not available")

            weights_dict = cfg.weights or {}
            valid_fields = set(HeuristicWeights.__dataclass_fields__.keys())
            weights = HeuristicWeights(**{
                k: v for k, v in weights_dict.items() if k in valid_fields
            }) if weights_dict else HeuristicWeights()

            detector = HeuristicIceDetector(weights)
            confidence_per_region, debug_info = detector.predict(features_df)
            uncertainty_per_region = np.zeros_like(confidence_per_region)

        else:  # ensemble
            ModelConfig = mods['ModelConfig']
            ens = cfg.ensemble
            model_cfg = ModelConfig(
                use_logistic=ens.use_logistic,
                use_naive_bayes=ens.use_naive_bayes,
                use_random_forest=ens.use_random_forest,
                gaussian_sigma=ens.gaussian_sigma,
                median_size=ens.median_size,
                # Use absolute path
                model_dir=_HERE / "processing" / "saved_models"
            )
            try:
                from pipeline import ModelEnsemble
                ensemble_model = ModelEnsemble(model_cfg)
                ensemble_model.load(model_cfg.model_dir)
                confidence_per_region, uncertainty_per_region = ensemble_model.get_ensemble_prediction(features_df)
            except Exception as e:
                raise HTTPException(
                    400,
                    f"Could not load ensemble models: {e}. "
                    "Ensure trained model files exist in processing/saved_models/"
                )

        # Map per-region confidence to pixel space
        region_ids = features_df['region_id'].values
        confidence_map  = np.zeros(segments.shape, dtype=np.float32)
        uncertainty_map = np.zeros(segments.shape, dtype=np.float32)
        for rid, conf, unc in zip(region_ids, confidence_per_region, uncertainty_per_region):
            confidence_map[segments == rid]  = conf
            uncertainty_map[segments == rid] = unc

        # Smooth
        confidence_map = gaussian_filter(confidence_map, sigma=2.0)
        confidence_map = median_filter(confidence_map, size=5)
        confidence_map = np.clip(confidence_map, 0, 1)

        # Visualizations
        cmap = plt.cm.RdYlBu_r
        heatmap_rgba = cmap(confidence_map)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

        rgb_norm = rgb.astype(np.float32) / 255.0
        overlay_arr = (0.5 * rgb_norm + 0.5 * heatmap_rgba[:, :, :3])
        overlay_rgb = np.clip(overlay_arr * 255, 0, 255).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        heat_path    = f"processing/{req.session_id}/heatmap_{ts}.jpg"
        overlay_path = f"processing/{req.session_id}/overlay_{ts}.jpg"
        _save_image_to_minio(heatmap_bgr, heat_path)
        _save_image_to_minio(overlay_bgr, overlay_path)

        _cache_set(req.session_id, 'confidence_map', confidence_map)

        ice_threshold = weights.ice_threshold
        mean_conf = float(confidence_map[segments > 0].mean()) if np.any(segments > 0) else 0.0
        ice_region_count = int(np.sum(confidence_per_region > ice_threshold))

        # ── Fire ice alert if confidence is high enough ────────────────────────
        alert_fired = False
        if mean_conf > 0.5:
            try:
                _, get_db_fn, _ = _get_minio_and_db()
                with get_db_fn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT d.latitude, d.longitude, cs.device_id
                        FROM capture_sessions cs
                        JOIN devices d ON cs.device_id = d.id
                        WHERE cs.id = %s
                    """, (req.session_id,))
                    loc = cur.fetchone()
                    cur.close()

                if loc and loc[0] is not None and loc[1] is not None:
                    import httpx as _httpx
                    _httpx.post(
                        "http://localhost:8000/api/app/alerts",
                        json={
                            "session_id": req.session_id,
                            "device_id": loc[2],
                            "latitude": loc[0],
                            "longitude": loc[1],
                            "confidence": mean_conf,
                        },
                        timeout=5.0
                    )
                    alert_fired = True
                    log.info(f"Ice alert fired for session {req.session_id} confidence={mean_conf:.2f}")
            except Exception as e:
                log.warning(f"Alert firing failed (non-fatal): {e}")

        return {
            "status": "success",
            "heatmap_url": _s3_url(heat_path),
            "overlay_rgb_url": _s3_url(overlay_path),
            "confidence_map_url": _s3_url(heat_path),
            "mean_confidence": mean_conf,
            "ice_region_count": ice_region_count,
            "total_regions": len(confidence_per_region),
            "model_used": cfg.model,
            "alert_fired": alert_fired
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(500, f"Inference failed: {e}")


# ─── REGION AT POINT ──────────────────────────────────────────────────────────

@processing_router.get("/region_at")
def get_region_at_point(session_id: str, x: int, y: int):
    segments = _cache_get(session_id, 'segments')
    if segments is None:
        raise HTTPException(400, "No segmentation data cached for this session")

    if y < 0 or y >= segments.shape[0] or x < 0 or x >= segments.shape[1]:
        raise HTTPException(400, "Coordinates out of bounds")

    region_id = int(segments[y, x])
    features_df = _cache_get(session_id, 'features_df')
    features = {}
    if features_df is not None and region_id > 0:
        rows = features_df[features_df['region_id'] == region_id]
        if not rows.empty:
            row = rows.iloc[0]
            features = {
                k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                for k, v in row.items()
                if not isinstance(v, (list, np.ndarray))
            }

    conf_map = _cache_get(session_id, 'confidence_map')
    if conf_map is not None and 0 <= y < conf_map.shape[0] and 0 <= x < conf_map.shape[1]:
        features['_confidence'] = round(float(conf_map[y, x]), 3)

    return {
        "region_id": region_id,
        "x": x,
        "y": y,
        "features": features
    }

# ------------------ Reset Endpoint
@processing_router.post("/reset")
def reset_session_cache(body: dict):
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    _cache_clear(session_id)
    return {"status": "ok", "session_id": session_id}