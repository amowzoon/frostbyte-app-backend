from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict
from contextlib import contextmanager
from minio import Minio
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from datetime import datetime
import uuid
import json
import logging
import asyncio
from io import BytesIO

from websocket_manager import device_manager
from processing_api import processing_router
from calibration_api import calibration_router
from alert_api import alert_router

# FOR DASHBOARD 
from dashboard import dashboard_websocket_handler, broadcast_dashboard_event, logs_websocket_handler, setup_websocket_logging, set_websocket_log_loop, set_shutdown_callback
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
import os

# Data visualizer
import numpy as np
import scipy.interpolate as spi
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from io import BytesIO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app.server")

# ---------------------------------------------------------------------------
# Capture expected-sensors tracking
# ---------------------------------------------------------------------------
# When the UI sends a capture command we record which sensors the session
# should expect.  The upload handler uses this when it creates the session
# row so that ``is_complete`` fires correctly for single-sensor captures.
_capture_expected: Dict[str, list] = {}
_CAPTURE_EXPECTED_MAX = 500

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Frostbyte Backend")
app.include_router(processing_router)
app.include_router(calibration_router)
app.include_router(alert_router)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Setup WebSocket logging (FOR DASHBOARD)
setup_websocket_logging()

# ---------------------------------------------------------------------------
# MinIO client
# ---------------------------------------------------------------------------
minio_client = Minio(
    "minio:9000",
    access_key="frosty",          # UPDATE if you changed this
    secret_key="fr0stbyte",       # UPDATE if you changed this
    secure=False
)

BUCKET_NAME = "sensor-data"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_db_pool: Optional[ThreadedConnectionPool] = None


def init_db_pool():
    """Create the connection pool.  Called once during startup."""
    global _db_pool
    _db_pool = ThreadedConnectionPool(
        minconn=2,
        maxconn=20,
        host="postgres",
        database="frostbyte",
        user="frosty",            # UPDATE if you changed this
        password="fr0stbyte"      # UPDATE if you changed this
    )
    log.info("Database connection pool created (min=2, max=20)")


@contextmanager
def get_db():
    """Get a connection from the pool.

    Usage::

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(...)
            conn.commit()
            cur.close()

    The connection is returned to the pool when the block exits.
    On unhandled exceptions the transaction is rolled back first.
    """
    conn = _db_pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _db_pool.putconn(conn)


@app.on_event("startup")
def startup():
    # Create MinIO bucket
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        log.info(f"Created MinIO bucket: {BUCKET_NAME}")
    
    # Initialize database connection pool
    init_db_pool()
    
    # Create PostgreSQL tables
    with get_db() as conn:
        cur = conn.cursor()
        
        # Devices table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100),
                registered_at TIMESTAMP DEFAULT NOW(),
                last_seen TIMESTAMP,
                metadata JSONB
            )
        """)
        
        # Capture sessions table (groups sensor captures together)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS capture_sessions (
                id UUID PRIMARY KEY,
                device_id VARCHAR(50) REFERENCES devices(id),
                capture_id VARCHAR(50) UNIQUE NOT NULL,
                captured_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                label VARCHAR(255),
                is_complete BOOLEAN DEFAULT FALSE,
                expected_sensors VARCHAR[] DEFAULT ARRAY['rgb', 'radar', 'ir']
            )
        """)
        
        # Individual sensor data table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id UUID PRIMARY KEY,
                session_id UUID REFERENCES capture_sessions(id) ON DELETE CASCADE,
                sensor_type VARCHAR(50) NOT NULL,
                s3_path VARCHAR(255) NOT NULL,
                metadata JSONB,
                uploaded_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(session_id, sensor_type)
            )
        """)
        
        # Indexes for faster lookups
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensor_data_session 
            ON sensor_data(session_id)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensor_data_type 
            ON sensor_data(sensor_type)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_capture_sessions_capture_id
            ON capture_sessions(capture_id)
        """)

        # Add s3_path_raw column if it doesn't exist (safe to run on existing DB)
        cur.execute("""
            ALTER TABLE sensor_data ADD COLUMN IF NOT EXISTS s3_path_raw VARCHAR(255)
        """)
        

        # ---------------------------------------------------------------------------
        # Alert app tables
        # ---------------------------------------------------------------------------
        cur.execute("""
            ALTER TABLE devices ADD COLUMN IF NOT EXISTS latitude  FLOAT
        """)
        cur.execute("""
            ALTER TABLE devices ADD COLUMN IF NOT EXISTS longitude FLOAT
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS app_users (
                id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email          VARCHAR(255) UNIQUE NOT NULL,
                password_hash  VARCHAR(255) NOT NULL,
                push_token     VARCHAR(255),
                alert_radius_m INTEGER DEFAULT 500,
                latitude       FLOAT,
                longitude      FLOAT,
                created_at     TIMESTAMP DEFAULT NOW(),
                last_seen      TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ice_alerts (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id  UUID REFERENCES capture_sessions(id),
                device_id   VARCHAR(50) REFERENCES devices(id),
                latitude    FLOAT NOT NULL,
                longitude   FLOAT NOT NULL,
                confidence  FLOAT NOT NULL,
                created_at  TIMESTAMP DEFAULT NOW(),
                expires_at  TIMESTAMP NOT NULL,
                active      BOOLEAN DEFAULT TRUE
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ice_alerts_location
            ON ice_alerts(latitude, longitude)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ice_alerts_active
            ON ice_alerts(active, expires_at)
        """)
        # is_test column — separates pseudo/test alerts from real captures
        # Safe to run on existing DB — ignored if column already exists
        cur.execute("""
            ALTER TABLE ice_alerts ADD COLUMN IF NOT EXISTS is_test BOOLEAN DEFAULT FALSE
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_app_users_location
            ON app_users(latitude, longitude)
        """)

        conn.commit()
        cur.close()
    log.info("Database tables initialized")


@app.on_event("shutdown")
def shutdown():
    global _db_pool
    if _db_pool is not None:
        _db_pool.closeall()
        log.info("Database connection pool closed")


@app.on_event("startup")
async def startup_async():
    """Async startup: bind the running event loop to the WebSocket log handler."""
    set_websocket_log_loop(asyncio.get_running_loop())
    set_shutdown_callback(shutdown_all_sensors)


async def shutdown_all_sensors():
    """Send uninitialize commands to all connected devices.

    Called by the dashboard grace-period timer when all dashboards
    have disconnected and none reconnected within the grace window.
    """
    devices = device_manager.get_connected_devices()
    if not devices:
        log.info("No connected devices — nothing to shut down")
        return

    for dev_id in devices:
        log.info(f"Sending sensor shutdown to {dev_id}")
        for sensor_path in [
            "/api/camera/uninitialize",
            "/api/ir/uninitialize",
            "/api/temperature/uninitialize",
        ]:
            try:
                result = await device_manager.send_query(
                    dev_id, method="POST", path=sensor_path, timeout=10.0
                )
                log.info(f"  {sensor_path}: {result.get('data', {}).get('message', 'ok')}")
            except Exception as e:
                log.warning(f"  {sensor_path}: {e}")
        # Radar: just stop, don't full-teardown (preserves config for fast re-init)
        try:
            result = await device_manager.send_query(
                dev_id, method="GET", path="/api/radar/status", timeout=5.0
            )
            radar_data = result.get("data", {})
            if radar_data.get("initialized"):
                log.info(f"  Radar is initialized on {dev_id} — leaving running (no auto-shutdown)")
        except Exception as e:
            log.warning(f"  Radar status check: {e}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Device Management
# ---------------------------------------------------------------------------
@app.get("/api/devices")
def list_devices():
    """List all registered devices and their connection status"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, registered_at, last_seen, metadata FROM devices ORDER BY registered_at")
        rows = cur.fetchall()
        cur.close()
    
    devices = []
    for row in rows:
        devices.append({
            "id": row[0],
            "name": row[1],
            "registered_at": row[2].isoformat() if row[2] else None,
            "last_seen": row[3].isoformat() if row[3] else None,
            "metadata": row[4],
            "connected": device_manager.is_connected(row[0])
        })
    
    return {"devices": devices}


@app.get("/api/devices/connected")
def list_connected_devices():
    """List currently connected devices"""
    return {"devices": device_manager.get_connected_devices()}


def register_device(device_id: str, metadata: dict = None):
    """Register or update a device in the database"""
    with get_db() as conn:
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO devices (id, last_seen, metadata)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                last_seen = EXCLUDED.last_seen,
                metadata = COALESCE(EXCLUDED.metadata, devices.metadata)
        """, (device_id, datetime.utcnow(), json.dumps(metadata) if metadata else None))
        
        conn.commit()
        cur.close()
    log.info(f"Device registered/updated: {device_id}")


def update_device_last_seen(device_id: str):
    """Update device last_seen timestamp"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE devices SET last_seen = %s WHERE id = %s", (datetime.utcnow(), device_id))
        conn.commit()
        cur.close()


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/device/{device_id}")
async def device_websocket(websocket: WebSocket, device_id: str):
    """WebSocket endpoint for device communication"""
    await device_manager.connect(device_id, websocket)
    
    # Register device in database
    await asyncio.to_thread(register_device, device_id)
    
    try:
        while True:
            # Wait for messages from device
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                await asyncio.to_thread(update_device_last_seen, device_id)
            
            elif msg_type == "command_complete":
                log.info(f"Command complete from {device_id}: {data}")
                await asyncio.to_thread(update_device_last_seen, device_id)
                
                # Dashboard: Broadcast sensor capture complete
                result = data.get("result", {})
                await broadcast_dashboard_event({
                    "type": "checkpoint",
                    "device_id": device_id,
                    "capture_id": result.get("capture_id"),
                    "stage": "sensor_capture_complete",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "sensor": result.get("sensor_type"),
                        "status": result.get("status"),
                        "metadata": result.get("metadata"),
                        "error": result.get("error") if result.get("status") == "failed" else None
                    }
                })
            
            elif msg_type == "query_response":
                device_manager.resolve_query(data.get("query_id"), data)

            elif msg_type == "status":
                log.info(f"Status from {device_id}: {data}")
                await asyncio.to_thread(update_device_last_seen, device_id)
            
            elif msg_type == "register":
                # Device sending its metadata
                await asyncio.to_thread(register_device, device_id, data.get("metadata"))
                await websocket.send_json({"type": "registered", "device_id": device_id})
            
            else:
                log.warning(f"Unknown message type from {device_id}: {msg_type}")
                
    except WebSocketDisconnect:
        device_manager.disconnect(device_id)
    except Exception as e:
        log.error(f"WebSocket error for {device_id}: {e}")
        device_manager.disconnect(device_id)

# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@app.websocket("/ws/dashboard")
async def dashboard_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for dashboard monitoring"""
    await dashboard_websocket_handler(websocket)

@app.websocket("/ws/logs")
async def logs_websocket_endpoint(websocket: WebSocket):
    await logs_websocket_handler(websocket)

@app.get("/")
async def root():
    return RedirectResponse(url="/capture")

@app.get("/capture")
async def serve_capture():
    return FileResponse("capture.html")

@app.get("/dashboard")
async def serve_dashboard():
    """Serve the dashboard HTML page"""
    return FileResponse("dashboard.html")

# ---------------------------------------------------------------------------
# Capture Commands (called by UI to trigger captures)
# ---------------------------------------------------------------------------
@app.post("/api/devices/{device_id}/capture/{sensor_type}")
async def send_capture_command(device_id: str, sensor_type: str, request: Request):
    """Send capture command to a device via WebSocket"""
    if not device_manager.is_connected(device_id):
        raise HTTPException(404, f"Device {device_id} not connected")
    if sensor_type not in ["rgb", "ir", "radar", "temperature", "all"]:
        raise HTTPException(400, f"Invalid sensor type: {sensor_type}")

    # Read optional capture options from request body
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    command_id = str(uuid.uuid4())
    capture_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"

    # Determine which sensors this capture expects so the session's
    # is_complete check works correctly for single-sensor captures.
    if sensor_type == "all":
        expected_sensors = ["rgb", "ir", "radar", "temperature"]
    else:
        expected_sensors = [sensor_type]

    # Store for later use when the upload creates the session row.
    # Evict oldest entries if over cap to avoid unbounded growth.
    if len(_capture_expected) > _CAPTURE_EXPECTED_MAX:
        for key in list(_capture_expected)[:len(_capture_expected) - _CAPTURE_EXPECTED_MAX]:
            del _capture_expected[key]
    _capture_expected[capture_id] = expected_sensors

    command = {
        "type": "command",
        "action": "capture",
        "sensor": sensor_type,
        "command_id": command_id,
        "capture_id": capture_id,
        # Forward all non-null capture options to the Pi
        **{k: v for k, v in body.items() if v is not None}
    }

    success = await device_manager.send_command(device_id, command)
    if not success:
        raise HTTPException(500, "Failed to send command")

    await broadcast_dashboard_event({
        "type": "checkpoint",
        "device_id": device_id,
        "capture_id": capture_id,
        "stage": "command_sent",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "device_id": device_id,
            "sensor": sensor_type,
            "command_id": command_id
        }
    })

    return {
        "status": "sent",
        "capture_id": capture_id,
        "command_id": command_id,
        "device_id": device_id,
        "sensor": sensor_type,
        "expected_sensors": expected_sensors
    }


# ---------------------------------------------------------------------------
# Device API Proxy (forwards requests to Pi's capture_api via WebSocket)
# ---------------------------------------------------------------------------
@app.api_route("/api/devices/{device_id}/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_device(device_id: str, path: str, request: Request):
    """Proxy an API request to the Pi's capture_api via WebSocket query/response."""
    if not device_manager.is_connected(device_id):
        raise HTTPException(404, f"Device {device_id} not connected")

    body = None
    if request.method in ("POST", "PUT"):
        try:
            body = await request.json()
        except Exception:
            body = {}

    try:
        result = await device_manager.send_query(
            device_id,
            method=request.method,
            path=f"/api/{path}",
            body=body,
            timeout=65.0  # Must exceed Pi-side httpx timeout (60s) for radar init etc.
        )
        return JSONResponse(content=result.get("data", {}), status_code=result.get("status_code", 200))
    except asyncio.TimeoutError:
        raise HTTPException(504, "Device did not respond in time")
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Data Upload (called by Pi after capture)
# ---------------------------------------------------------------------------
@app.post("/api/upload/{sensor_type}")
async def upload_capture(
    sensor_type: str,
    file: UploadFile = File(...),
    device_id: str = Form(...),
    captured_at: str = Form(...),
    metadata: str = Form(None),
    raw_file: Optional[UploadFile] = File(None)
):
    """
    Upload sensor capture data.
    Creates or updates capture session and stores sensor data.
    """
    try:
        # Parse metadata
        metadata_dict = json.loads(metadata) if metadata else {}
        capture_id = metadata_dict.get("capture_id")
        
        if not capture_id:
            raise HTTPException(status_code=400, detail="capture_id missing in metadata")
        
        # Generate UUID for sensor data
        sensor_data_id = str(uuid.uuid4())
        
        # Read file data (async)
        file_data = await file.read()
        raw_data = None
        if raw_file is not None:
            raw_data = await raw_file.read()

        # Dashboard: Broadcast upload received
        await broadcast_dashboard_event({
            "type": "checkpoint",
            "device_id": device_id,
            "capture_id": capture_id,
            "stage": "upload_received",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "sensor": sensor_type,
                "size_bytes": len(file_data)
            }
        })
        
        # Run all blocking MinIO + DB work in a thread
        result = await asyncio.to_thread(
            _upload_store_and_record,
            sensor_type, device_id, captured_at, capture_id, metadata_dict,
            sensor_data_id, file_data, file.filename, file.content_type, raw_data
        )

        # Dashboard: Broadcast S3 stored
        await broadcast_dashboard_event({
            "type": "checkpoint",
            "device_id": device_id,
            "capture_id": capture_id,
            "stage": "s3_stored",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "s3_path": result["s3_path"],
                "bucket": BUCKET_NAME
            }
        })

        # Dashboard: Broadcast DB stored
        await broadcast_dashboard_event({
            "type": "checkpoint",
            "device_id": device_id,
            "capture_id": capture_id,
            "stage": "db_stored",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "session_id": result["session_id"],
                "is_complete": result["is_complete"]
            }
        })

        return result
            
    except Exception as e:
        log.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _upload_store_and_record(
    sensor_type, device_id, captured_at, capture_id, metadata_dict,
    sensor_data_id, file_data, filename, content_type, raw_data
):
    """Blocking helper: upload to MinIO and record in DB.

    Runs in a worker thread via asyncio.to_thread() so it doesn't
    block the event loop.
    """
    # Upload to MinIO
    s3_filename = f"{sensor_data_id}_{filename}"
    s3_path = f"{device_id}/{sensor_type}/{s3_filename}"
    
    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=s3_path,
        data=BytesIO(file_data),
        length=len(file_data),
        content_type=content_type
    )
    log.info(f"Uploaded {len(file_data)} bytes to MinIO: {s3_path}")
    # Extract temperature_c from frame JSON and inject into metadata
    if sensor_type == "temperature":
        try:
            frame = json.loads(file_data.decode())
            reading = frame.get("reading", {})
            # Two TMP36 sensors: s1 and s2, plus their delta
            temp_c = reading.get("s1_c")
            if temp_c is not None:
                metadata_dict["temperature_c"] = round(float(temp_c), 2)
                metadata_dict["temperature_f"] = round(float(reading.get("s1_f", temp_c * 9/5 + 32)), 2)
            if reading.get("s2_c") is not None:
                metadata_dict["temperature_c_s2"] = round(float(reading["s2_c"]), 2)
                metadata_dict["temperature_f_s2"] = round(float(reading.get("s2_f", reading["s2_c"] * 9/5 + 32)), 2)
            if reading.get("delta_c") is not None:
                metadata_dict["temperature_delta_c"] = round(float(reading["delta_c"]), 2)
                log.info(f"Extracted temperatures — s1: {metadata_dict.get('temperature_c')}°C, s2: {metadata_dict.get('temperature_c_s2')}°C")
        except Exception as e:
            log.warning(f"Could not parse temperature frame: {e}")
    # Upload raw .npy file if provided (IR only)
    s3_path_raw = None
    if raw_data is not None:
        raw_s3_filename = f"{sensor_data_id}_raw.npy"
        s3_path_raw = f"{device_id}/{sensor_type}/{raw_s3_filename}"
        minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=s3_path_raw,
            data=BytesIO(raw_data),
            length=len(raw_data),
            content_type="application/octet-stream"
        )
        log.info(f"Uploaded {len(raw_data)} bytes raw to MinIO: {s3_path_raw}")

    # Database operations
    with get_db() as conn:
        cur = conn.cursor()
        
        try:
            # Get or create capture session
            cur.execute("""
                SELECT id FROM capture_sessions WHERE capture_id = %s
            """, (capture_id,))
            
            result = cur.fetchone()
            
            if result:
                session_id = result[0]
                log.info(f"Found existing session: {session_id}")
            else:
                # Create new session with correct expected_sensors
                # (looked up from _capture_expected, falls back to the
                #  schema default when the capture wasn't initiated via UI)
                expected = _capture_expected.pop(capture_id, None)
                session_id = str(uuid.uuid4())
                if expected is not None:
                    cur.execute("""
                        INSERT INTO capture_sessions 
                        (id, device_id, capture_id, captured_at, is_complete, expected_sensors)
                        VALUES (%s, %s, %s, %s, FALSE, %s)
                    """, (session_id, device_id, capture_id, captured_at, expected))
                else:
                    cur.execute("""
                        INSERT INTO capture_sessions 
                        (id, device_id, capture_id, captured_at, is_complete)
                        VALUES (%s, %s, %s, %s, FALSE)
                    """, (session_id, device_id, capture_id, captured_at))
                log.info(f"Created new session: {session_id} (expected_sensors={expected})")
            
            # Check if this sensor already exists for this session (replacement scenario)
            cur.execute("""
                SELECT id, s3_path, s3_path_raw FROM sensor_data
                WHERE session_id = %s AND sensor_type = %s
            """, (session_id, sensor_type))
            
            existing = cur.fetchone()
            
            if existing:
                old_sensor_id, old_s3_path, old_s3_path_raw = existing

                # Delete old S3 files
                for old_path in [old_s3_path, old_s3_path_raw]:
                    if old_path:
                        try:
                            minio_client.remove_object(BUCKET_NAME, old_path)
                            log.info(f"Deleted old S3 file: {old_path}")
                        except Exception as e:
                            log.warning(f"Failed to delete old S3 file {old_path}: {e}")
                
                # Delete old sensor_data record
                cur.execute("""
                    DELETE FROM sensor_data WHERE id = %s
                """, (old_sensor_id,))
                log.info(f"Replaced {sensor_type} data for session {session_id}")
            
            # Insert new sensor data
            cur.execute("""
                INSERT INTO sensor_data
                (id, session_id, sensor_type, s3_path, s3_path_raw, metadata, uploaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (sensor_data_id, session_id, sensor_type, s3_path, s3_path_raw, json.dumps(metadata_dict)))
            
            # Check if session is complete (all expected sensors uploaded)
            cur.execute("""
                SELECT
                    (SELECT COUNT(DISTINCT sd.sensor_type)
                     FROM sensor_data sd
                     WHERE sd.session_id = cs.id
                     AND sd.sensor_type = ANY(cs.expected_sensors)
                    ) >= array_length(cs.expected_sensors, 1)
                FROM capture_sessions cs WHERE cs.id = %s
            """, (session_id,))
            is_complete = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(DISTINCT sensor_type) FROM sensor_data WHERE session_id = %s
            """, (session_id,))
            sensor_count = cur.fetchone()[0]

            if is_complete:
                cur.execute("""
                    UPDATE capture_sessions SET is_complete = TRUE
                    WHERE id = %s
                """, (session_id,))
                log.info(f"Session {session_id} marked as complete")
            
            conn.commit()
            
            return {
                "status": "success",
                "session_id": session_id,
                "capture_id": capture_id,
                "sensor_data_id": sensor_data_id,
                "sensor_type": sensor_type,
                "s3_path": s3_path,
                "s3_path_raw": s3_path_raw,
                "is_complete": is_complete,
                "sensor_count": sensor_count
            }
            
        finally:
            cur.close()


# ---------------------------------------------------------------------------
# Captures List
# ---------------------------------------------------------------------------
@app.get("/api/captures")
def list_captures(device_id: str = None, sensor_type: str = None, limit: int = 100):
    """
    Get capture sessions with optional filtering.
    Returns sessions with all associated sensor data.
    """
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Build query with LEFT JOIN to include sessions even without sensor data
            query = """
                SELECT 
                    cs.id,
                    cs.device_id,
                    cs.capture_id,
                    cs.captured_at,
                    cs.created_at,
                    cs.label,
                    cs.is_complete,
                    json_agg(
                        CASE WHEN sd.id IS NOT NULL THEN
                            json_build_object(
                                'id', sd.id,
                                'sensor_type', sd.sensor_type,
                                's3_path', sd.s3_path,
                                'metadata', sd.metadata,
                                'uploaded_at', sd.uploaded_at
                            )
                        END
                    ) FILTER (WHERE sd.id IS NOT NULL) as sensors
                FROM capture_sessions cs
                LEFT JOIN sensor_data sd ON cs.id = sd.session_id
            """
            
            conditions = []
            params = []
            
            if device_id:
                conditions.append("cs.device_id = %s")
                params.append(device_id)
            
            if sensor_type:
                conditions.append("sd.sensor_type = %s")
                params.append(sensor_type)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " GROUP BY cs.id ORDER BY cs.captured_at DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            rows = cur.fetchall()
            cur.close()
        
        captures = []
        for row in rows:
            captures.append({
                "session_id": str(row[0]),
                "device_id": row[1],
                "capture_id": row[2],
                "captured_at": row[3].isoformat() if row[3] else None,
                "created_at": row[4].isoformat() if row[4] else None,
                "label": row[5],
                "is_complete": row[6],
                "sensors": row[7] if row[7] else []
            })
        
        return {"captures": captures}
        
    except Exception as e:
        log.error(f"Failed to get captures: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
def get_sessions(
    device_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """
    Get all capture sessions with completeness information.
    Returns sessions sorted by most recent first.
    """
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Build query with optional device filter
            query = """
                SELECT 
                    cs.id,
                    cs.capture_id,
                    cs.device_id,
                    cs.captured_at,
                    cs.created_at,
                    cs.label,
                    cs.is_complete,
                    cs.expected_sensors,
                    COUNT(sd.id) as sensor_count,
                    ARRAY_AGG(sd.sensor_type) FILTER (WHERE sd.sensor_type IS NOT NULL) as captured_sensors
                FROM capture_sessions cs
                LEFT JOIN sensor_data sd ON cs.id = sd.session_id
            """
            
            params = []
            if device_id:
                query += " WHERE cs.device_id = %s"
                params.append(device_id)
            
            query += """
                GROUP BY cs.id
                ORDER BY cs.created_at DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            
            cur.execute(query, params)
            rows = cur.fetchall()
            cur.close()
        
        sessions = []
        for row in rows:
            session_id, capture_id, dev_id, captured_at, created_at, label, is_complete, expected_sensors, sensor_count, captured_sensors = row
            
            # Calculate completeness
            expected_count = len(expected_sensors) if expected_sensors else 3
            completeness_percentage = (sensor_count / expected_count * 100) if expected_count > 0 else 0
            
            sessions.append({
                "session_id": str(session_id),
                "capture_id": capture_id,
                "device_id": dev_id,
                "captured_at": captured_at.isoformat() if captured_at else None,
                "created_at": created_at.isoformat() if created_at else None,
                "label": label,
                "is_complete": is_complete,
                "expected_sensors": expected_sensors or [],
                "captured_sensors": captured_sensors or [],
                "sensor_count": sensor_count,
                "expected_count": expected_count,
                "completeness_percentage": round(completeness_percentage, 1)
            })
        
        return {
            "sessions": sessions,
            "count": len(sessions),
            "limit": limit,
            "offset": offset
        }
    
    except Exception as e:
        log.error(f"Error fetching sessions: {e}")
        raise HTTPException(500, f"Failed to fetch sessions: {e}")


@app.get("/api/sessions/{session_id}")
def get_session_detail(session_id: str):
    """
    Get detailed information about a specific session including all sensor data.
    """
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Get session info
            cur.execute("""
                SELECT id, capture_id, device_id, captured_at, created_at, label, is_complete, expected_sensors
                FROM capture_sessions
                WHERE id = %s
            """, (session_id,))
            
            session_row = cur.fetchone()
            if not session_row:
                raise HTTPException(404, "Session not found")
            
            session_id, capture_id, device_id, captured_at, created_at, label, is_complete, expected_sensors = session_row
            
            # Get all sensor data for this session
            cur.execute("""
                SELECT id, sensor_type, s3_path, metadata, uploaded_at
                FROM sensor_data
                WHERE session_id = %s
                ORDER BY sensor_type
            """, (session_id,))
            
            sensor_rows = cur.fetchall()
            cur.close()
        
        sensors = []
        for row in sensor_rows:
            s_id, sensor_type, s3_path, metadata, uploaded_at = row
            sensors.append({
                "id": str(s_id),
                "sensor_type": sensor_type,
                "s3_path": s3_path,
                "metadata": metadata,
                "uploaded_at": uploaded_at.isoformat() if uploaded_at else None
            })
        
        return {
            "session_id": str(session_id),
            "capture_id": capture_id,
            "device_id": device_id,
            "captured_at": captured_at.isoformat() if captured_at else None,
            "created_at": created_at.isoformat() if created_at else None,
            "label": label,
            "is_complete": is_complete,
            "expected_sensors": expected_sensors or [],
            "sensors": sensors
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error fetching session detail: {e}")
        raise HTTPException(500, f"Failed to fetch session: {e}")


@app.get("/api/images/{s3_path:path}")
def get_image(s3_path: str):
    """
    Serve an image from MinIO S3 storage.
    """
    try:
        # Get object from MinIO
        response = minio_client.get_object(BUCKET_NAME, s3_path)
        
        # Determine content type based on file extension
        content_type = "application/octet-stream"
        if s3_path.endswith('.jpg') or s3_path.endswith('.jpeg'):
            content_type = "image/jpeg"
        elif s3_path.endswith('.png'):
            content_type = "image/png"
        elif s3_path.endswith('.npy'):
            content_type = "application/octet-stream"
        
        # Read the data
        data = response.read()
        response.close()
        response.release_conn()
        
        from fastapi.responses import Response
        return Response(content=data, media_type=content_type)
    
    except Exception as e:
        log.error(f"Error fetching image {s3_path}: {e}")
        raise HTTPException(404, f"Image not found: {e}")

@app.get("/api/radar/heatmap/{s3_path:path}")
def generate_radar_heatmap(s3_path: str):
    """
    Generate radar heatmap visualization from S3 radar data.
    Returns PNG image of range-azimuth heatmap.
    """
    try:
        # Fetch radar JSON from S3
        response = minio_client.get_object(BUCKET_NAME, s3_path)
        data = json.loads(response.read().decode('utf-8'))
        response.close()
        response.release_conn()
        
        # Radar parameters
        TX_AZIMUTH_ANTENNAS = 2
        RX_ANTENNAS = 4
        RANGE_BINS = 256
        ANGLE_BINS = 64
        RANGE_RESOLUTION = 0.04360212053571429
        RANGE_BIAS = 0.07
        
        # Extract azimuth data
        azimuth_data = data['azimuth_static']
        
        # Convert I,Q pairs to complex numbers
        a = np.array([azimuth_data[i] + 1j * azimuth_data[i+1] 
                     for i in range(0, len(azimuth_data), 2)])
        
        # Reshape to (range_bins, virtual_antennas)
        virtual_antennas = TX_AZIMUTH_ANTENNAS * RX_ANTENNAS
        a = np.reshape(a, (RANGE_BINS, virtual_antennas))
        
        # Perform azimuth FFT
        a = np.fft.fft(a, ANGLE_BINS)
        a = np.abs(a)
        a = np.fft.fftshift(a, axes=(1,))
        
        # Create coordinate grids
        range_depth = RANGE_BINS * RANGE_RESOLUTION
        range_width = range_depth / 2
        grid_res = 400
        
        num_angles = a.shape[1]
        t = np.linspace(-np.pi/2, np.pi/2, num_angles)
        r = np.array(range(RANGE_BINS)) * RANGE_RESOLUTION
        
        x = np.array([r]).T * np.sin(t)
        y = np.array([r]).T * np.cos(t)
        y = y - RANGE_BIAS
        
        # Interpolation grid
        xi = np.linspace(-range_width, range_width, grid_res)
        yi = np.linspace(0, range_depth, grid_res)
        xi, yi = np.meshgrid(xi, yi)
        
        zi = spi.griddata((x.ravel(), y.ravel()), a.ravel(), 
                         (xi, yi), method='linear')
        zi = zi[:-1, :-1]
        
        # Create plot
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(1, 1, 1)
        fig.tight_layout(pad=2)
        
        # Plot heatmap (rotated 180 degrees)
        cm = ax.imshow(
            zi[::-1, ::-1], 
            cmap=plt.cm.jet, 
            extent=[-range_width, +range_width, 0, range_depth],
            alpha=0.95,
            aspect='auto'
        )
        
        plt.colorbar(cm, ax=ax, label='Magnitude')
        
        # Formatting
        frame_num = data['metadata'].get('frame_number', 'N/A')
        timestamp = data['metadata'].get('timestamp', 'N/A')
        
        ax.set_title(f'Range-Azimuth Heatmap - Frame #{frame_num}\n{timestamp}', 
                    fontsize=12)
        ax.set_xlabel('Lateral distance [m]')
        ax.set_ylabel('Longitudinal distance [m]')
        
        # Reference lines
        ax.plot([0, 0], [0, range_depth], color='white', 
               linewidth=0.5, linestyle=':', zorder=1)
        ax.plot([0, -range_width], [0, range_width], color='white', 
               linewidth=0.5, linestyle=':', zorder=1)
        ax.plot([0, +range_width], [0, range_width], color='white', 
               linewidth=0.5, linestyle=':', zorder=1)
        
        ax.set_ylim([0, +range_depth])
        ax.set_xlim([-range_width, +range_width])
        
        # Range arcs
        for i in range(1, int(range_depth) + 1):
            ax.add_patch(pat.Arc(
                (0, 0), 
                width=i*2, 
                height=i*2, 
                angle=90, 
                theta1=-90, 
                theta2=90,
                color='white', 
                linewidth=0.5, 
                linestyle=':', 
                zorder=1
            ))
        
        # Save to BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        log.error(f"Error generating radar heatmap: {e}")
        raise HTTPException(500, f"Failed to generate heatmap: {e}")

@app.get("/data-viewer")
async def serve_data_viewer():
    """Serve the data viewer page"""
    return FileResponse("data-viewer.html")
# ---------------------------------------------------------------------------
# Calibration Route
# ---------------------------------------------------------------------------
@app.get("/calibration")
async def serve_calibration():
    return FileResponse("calibration.html")

# ---------------------------------------------------------------------------
# Session Replacement (trigger re-capture of single sensor)
# ---------------------------------------------------------------------------
@app.post("/api/sessions/{session_id}/replace/{sensor_type}")
async def replace_sensor(session_id: str, sensor_type: str):
    """
    Trigger replacement of a single sensor in an existing capture session.
    Sends command to device with existing capture_id.
    """
    try:
        def _lookup_session():
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT capture_id, device_id FROM capture_sessions WHERE id = %s
                """, (session_id,))
                row = cur.fetchone()
                cur.close()
            return row

        result = await asyncio.to_thread(_lookup_session)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        capture_id, device_id = result
        
        # Check if device is connected
        if not device_manager.is_connected(device_id):
            raise HTTPException(status_code=400, detail=f"Device {device_id} is not connected")
        
        # Send capture command with existing capture_id
        command = {
            "type": "command",
            "command_id": str(uuid.uuid4()),
            "action": "capture",
            "sensor": sensor_type,
            "capture_id": capture_id  # Important: use existing capture_id
        }
        
        success = await device_manager.send_command(device_id, command)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to send command to device")
        
        return {
            "status": "command_sent",
            "session_id": session_id,
            "capture_id": capture_id,
            "device_id": device_id,
            "sensor_type": sensor_type,
            "command_id": command["command_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to trigger replacement: {e}")
        raise HTTPException(status_code=500, detail=str(e))