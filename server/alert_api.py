"""
alert_api.py
FastAPI router for the FrostByte pedestrian/driver alert app.

Mount into main.py with:
    from alert_api import alert_router
    app.include_router(alert_router)

Requires these pip packages (add to requirements.txt):
    passlib[bcrypt]
    python-jose[cryptography]
    httpx
"""

import os
import logging
import uuid
import httpx
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt

log = logging.getLogger("app.alert_api")

alert_router = APIRouter(prefix="/api/app", tags=["alert-app"])

# ---------------------------------------------------------------------------
# Supabase JWT validation
# The Supabase project JWT secret — found in Supabase dashboard:
# Project Settings → API → JWT Settings → JWT Secret
# Set this as SUPABASE_JWT_SECRET in docker-compose.yml environment
# ---------------------------------------------------------------------------
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
bearer_scheme = HTTPBearer()

# ---------------------------------------------------------------------------
# Ice alert config
# ---------------------------------------------------------------------------
ICE_CONFIDENCE_THRESHOLD = float(os.getenv("ICE_ALERT_THRESHOLD", "0.5"))
ALERT_EXPIRE_HOURS = int(os.getenv("ALERT_EXPIRE_HOURS", "2"))

# Expo push notification endpoint
EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class PushTokenRequest(BaseModel):
    push_token: str

class SettingsRequest(BaseModel):
    alert_radius_m: Optional[int] = None

class CreateAlertRequest(BaseModel):
    """Internal only — called by processing pipeline after inference."""
    session_id: str
    device_id: str
    latitude: float
    longitude: float
    confidence: float


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    """
    Validates a Supabase-issued JWT and returns the user_id (UUID string).
    The token is issued by Supabase auth and verified using the project JWT secret.
    """
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(500, "SUPABASE_JWT_SECRET not configured on server")
    try:
        payload = jwt.decode(
            credentials.credentials,
            SUPABASE_JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience="authenticated"
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token: missing sub claim")
        return user_id
    except JWTError as e:
        raise HTTPException(401, f"Invalid or expired token: {e}")


# ---------------------------------------------------------------------------
# DB helpers — reuse main.py's get_db connection pool
# ---------------------------------------------------------------------------

def _get_db():
    try:
        import main as app_main
        return app_main.get_db
    except Exception as e:
        raise HTTPException(500, f"Cannot access DB context: {e}")


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@alert_router.post("/push-token")
def update_push_token(
    req: PushTokenRequest,
    user_id: str = Depends(_get_current_user)
):
    """Register or update the Expo push token for this user's device."""
    get_db = _get_db()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE app_users SET push_token = %s WHERE id = %s",
            (req.push_token, user_id)
        )
        conn.commit()
        cur.close()

    return {"status": "ok"}


@alert_router.patch("/settings")
def update_settings(
    req: SettingsRequest,
    user_id: str = Depends(_get_current_user)
):
    """Update user settings (alert radius, etc.)."""
    get_db = _get_db()
    with get_db() as conn:
        cur = conn.cursor()
        if req.alert_radius_m is not None:
            cur.execute(
                "UPDATE app_users SET alert_radius_m = %s WHERE id = %s",
                (req.alert_radius_m, user_id)
            )
        conn.commit()
        cur.close()

    return {"status": "ok"}


@alert_router.get("/settings")
def get_settings(user_id: str = Depends(_get_current_user)):
    """Get current user settings."""
    get_db = _get_db()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT email, alert_radius_m FROM app_users WHERE id = %s",
            (user_id,)
        )
        row = cur.fetchone()
        cur.close()

    if not row:
        raise HTTPException(404, "User not found")

    return {
        "email": row[0],
        "alert_radius_m": row[1],
    }


# ---------------------------------------------------------------------------
# Alert endpoints
# ---------------------------------------------------------------------------

@alert_router.get("/alerts/nearby")
def get_nearby_alerts(
    lat: float,
    lon: float,
    radius_m: int = 500,
):
    # No auth required — ice alert locations are public information.
    # User identity is not needed and not logged.
    """
    Return active ice alerts near the given GPS coordinates.
    Called by the app every 30 seconds while open.

    Uses a bounding box approximation:
      0.01 degrees latitude  ≈ 1.1 km
      0.01 degrees longitude ≈ 0.8 km at mid-latitudes
    Adjust the multiplier based on radius_m.
    """
    # Convert radius_m to rough degree offset
    # 111,000 meters per degree latitude
    deg_offset = radius_m / 111000.0

    get_db = _get_db()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                id,
                latitude,
                longitude,
                confidence,
                created_at,
                expires_at,
                device_id
            FROM ice_alerts
            WHERE active = TRUE
              AND expires_at > NOW()
              AND (is_test = FALSE OR is_test IS NULL)
              AND latitude  BETWEEN %s AND %s
              AND longitude BETWEEN %s AND %s
            ORDER BY confidence DESC
            LIMIT 50
        """, (
            lat - deg_offset, lat + deg_offset,
            lon - deg_offset, lon + deg_offset
        ))
        rows = cur.fetchall()
        cur.close()

    alerts = [
        {
            "id": str(row[0]),
            "latitude": row[1],
            "longitude": row[2],
            "confidence": round(row[3], 2),
            "created_at": row[4].isoformat(),
            "expires_at": row[5].isoformat(),
            "device_id": row[6],
        }
        for row in rows
    ]

    return {"alerts": alerts, "count": len(alerts)}


@alert_router.post("/alerts")
def create_alert(req: CreateAlertRequest):
    """
    Internal endpoint — called by the processing pipeline after inference
    confirms ice with confidence above threshold.

    Do NOT expose this publicly. In production, add an internal API key check.
    """
    if req.confidence < ICE_CONFIDENCE_THRESHOLD:
        return {"status": "skipped", "reason": "confidence below threshold"}

    get_db = _get_db()
    alert_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(hours=ALERT_EXPIRE_HOURS)

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ice_alerts
                (id, session_id, device_id, latitude, longitude, confidence, expires_at, is_test)
            VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
        """, (
            alert_id,
            req.session_id,
            req.device_id,
            req.latitude,
            req.longitude,
            req.confidence,
            expires_at
        ))
        conn.commit()
        cur.close()

    log.info(
        f"Ice alert created: id={alert_id} "
        f"lat={req.latitude} lon={req.longitude} "
        f"confidence={req.confidence:.2f}"
    )

    # Publish to Supabase so users anywhere can see the alert
    # without needing the local backend to be running
    _publish_to_supabase(
        alert_id=alert_id,
        session_id=req.session_id,
        device_id=req.device_id,
        latitude=req.latitude,
        longitude=req.longitude,
        confidence=req.confidence,
        expires_at=expires_at,
        alert_type="ice",
    )

    # Send push notifications to nearby users
    _send_push_notifications_sync(req.latitude, req.longitude, req.confidence)

    return {"status": "created", "alert_id": alert_id}


def _publish_to_supabase(
    alert_id: str,
    session_id: str,
    device_id: str,
    latitude: float,
    longitude: float,
    confidence: float,
    expires_at,
    alert_type: str = "ice",
):
    """
    Write alert to Supabase so the app can read it anywhere without
    the local backend running. Uses the Supabase service role key
    set in SUPABASE_SERVICE_KEY environment variable.
    Falls back gracefully if Supabase is not configured.
    """
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")  # service role key, not anon

    if not supabase_url or not supabase_key:
        log.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not set — skipping Supabase publish")
        return

    try:
        payload = {
            "id": alert_id,
            "session_id": session_id,
            "device_id": device_id,
            "latitude": latitude,
            "longitude": longitude,
            "confidence": confidence,
            "alert_type": alert_type,
            "expires_at": expires_at.isoformat(),
            "active": True,
            "is_test": False,
        }
        resp = httpx.post(
            f"{supabase_url}/rest/v1/ice_alerts",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=payload,
            timeout=5.0,
        )
        if resp.status_code in (200, 201):
            log.info(f"Alert published to Supabase: {alert_id}")
        else:
            log.warning(f"Supabase publish returned {resp.status_code}: {resp.text}")
    except Exception as e:
        log.warning(f"Supabase publish failed (non-fatal): {e}")


def _send_push_notifications_sync(lat: float, lon: float, confidence: float):
    """
    Find nearby users with push tokens and send Expo push notifications.
    Runs synchronously — for high volume, move to a background task queue.
    """
    get_db = _get_db()
    deg_offset = 1000 / 111000.0  # notify users within 1km

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT push_token, alert_radius_m
            FROM app_users
            WHERE push_token IS NOT NULL
              AND latitude  BETWEEN %s AND %s
              AND longitude BETWEEN %s AND %s
        """, (
            lat - deg_offset, lat + deg_offset,
            lon - deg_offset, lon + deg_offset
        ))
        users = cur.fetchall()
        cur.close()

    if not users:
        return

    messages = [
        {
            "to": row[0],
            "title": "Black Ice Detected Nearby",
            "body": f"Ice detected {round(confidence * 100)}% confidence near your location. Drive and walk carefully.",
            "data": {"lat": lat, "lon": lon, "confidence": confidence},
            "sound": "default",
            "priority": "high",
        }
        for row in users
    ]

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(EXPO_PUSH_URL, json=messages)
            log.info(f"Push notifications sent to {len(messages)} users: {resp.status_code}")
    except Exception as e:
        log.warning(f"Push notification delivery failed: {e}")