"""
alert_api.py
------------
Alert and user preferences endpoints backed by local Postgres.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import math

import asyncpg
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from db import get_pool
from auth_api import get_current_user

log = logging.getLogger("app.alerts")

alert_router = APIRouter(prefix="/api/app")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PushTokenRequest(BaseModel):
    push_token: str

class UserSettings(BaseModel):
    alert_radius_m: Optional[int] = 500
    notify_ice: Optional[bool] = True
    notify_bluetooth: Optional[bool] = True
    notify_route: Optional[bool] = True

class AlertCreate(BaseModel):
    latitude: float
    longitude: float
    confidence: float
    alert_type: str = "heuristic"
    device_id: Optional[str] = None
    expires_minutes: Optional[int] = 60
    is_test: Optional[bool] = False


# ---------------------------------------------------------------------------
# Push token
# ---------------------------------------------------------------------------

@alert_router.post("/push-token")
async def store_push_token(req: PushTokenRequest, user=Depends(get_current_user)):
    pool: asyncpg.Pool = await get_pool()
    await pool.execute("""
        INSERT INTO user_preferences (user_id, push_token, updated_at)
        VALUES ($1, $2, now())
        ON CONFLICT (user_id) DO UPDATE
            SET push_token = EXCLUDED.push_token,
                updated_at = now()
    """, user["sub"], req.push_token)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

@alert_router.get("/settings")
async def get_settings(user=Depends(get_current_user)):
    pool: asyncpg.Pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT alert_radius_m, notify_ice, notify_bluetooth, notify_route "
        "FROM user_preferences WHERE user_id = $1",
        user["sub"],
    )
    if not row:
        return {"alert_radius_m": 500, "notify_ice": True, "notify_bluetooth": True, "notify_route": True}
    return dict(row)


@alert_router.patch("/settings")
async def update_settings(settings: UserSettings, user=Depends(get_current_user)):
    pool: asyncpg.Pool = await get_pool()
    await pool.execute("""
        INSERT INTO user_preferences (user_id, alert_radius_m, notify_ice, notify_bluetooth, notify_route, updated_at)
        VALUES ($1, $2, $3, $4, $5, now())
        ON CONFLICT (user_id) DO UPDATE
            SET alert_radius_m   = EXCLUDED.alert_radius_m,
                notify_ice       = EXCLUDED.notify_ice,
                notify_bluetooth = EXCLUDED.notify_bluetooth,
                notify_route     = EXCLUDED.notify_route,
                updated_at       = now()
    """, user["sub"], settings.alert_radius_m, settings.notify_ice,
        settings.notify_bluetooth, settings.notify_route)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@alert_router.get("/alerts/nearby")
async def get_nearby_alerts(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_m: int = Query(2000),
):
    deg_offset = radius_m / 111000.0
    now = datetime.now(timezone.utc)
    pool: asyncpg.Pool = await get_pool()

    rows = await pool.fetch("""
        SELECT id, latitude, longitude, confidence, alert_type, device_id, created_at, expires_at
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY created_at DESC) AS rn
            FROM ice_alerts
            WHERE active = true
              AND is_test = false
              AND expires_at > $1
              AND latitude  BETWEEN $2 AND $3
              AND longitude BETWEEN $4 AND $5
        ) ranked
        WHERE rn <= 2
        ORDER BY confidence DESC
        LIMIT 50
    """, now,
        lat - deg_offset, lat + deg_offset,
        lon - deg_offset, lon + deg_offset,
    )

    alerts = [
        {**dict(r), "id": str(r["id"]), "created_at": r["created_at"].isoformat(), "expires_at": r["expires_at"].isoformat()}
        for r in rows
    ]
    return {"alerts": alerts, "count": len(alerts)}


# ---------------------------------------------------------------------------
# Devices nearby (internet-based proximity scan)
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@alert_router.get("/devices/nearby")
async def get_devices_nearby(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_m: int = Query(80),
):
    deg_offset = radius_m / 111000.0
    now = datetime.now(timezone.utc)
    pool: asyncpg.Pool = await get_pool()

    rows = await pool.fetch("""
        SELECT DISTINCT ON (device_id)
            device_id, latitude, longitude, confidence, created_at
        FROM ice_alerts
        WHERE active = true
          AND is_test = false
          AND expires_at > $1
          AND device_id IS NOT NULL
          AND latitude  BETWEEN $2 AND $3
          AND longitude BETWEEN $4 AND $5
        ORDER BY device_id, created_at DESC
    """, now,
        lat - deg_offset, lat + deg_offset,
        lon - deg_offset, lon + deg_offset,
    )

    devices = []
    for r in rows:
        dist = haversine_m(lat, lon, r["latitude"], r["longitude"])
        devices.append({
            "device_id": r["device_id"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "confidence": r["confidence"],
            "distance_m": round(dist),
            "last_seen": r["created_at"].isoformat(),
        })

    devices.sort(key=lambda d: d["distance_m"])
    return {"devices": devices, "count": len(devices)}


# ---------------------------------------------------------------------------
# Create / delete alerts
# ---------------------------------------------------------------------------

@alert_router.post("/alerts")
async def create_alert(alert: AlertCreate):
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=alert.expires_minutes)
    pool: asyncpg.Pool = await get_pool()

    row = await pool.fetchrow("""
        INSERT INTO ice_alerts (latitude, longitude, confidence, alert_type, device_id, expires_at, active, is_test)
        VALUES ($1, $2, $3, $4, $5, $6, true, $7)
        RETURNING id
    """, alert.latitude, alert.longitude, alert.confidence,
        alert.alert_type, alert.device_id, expires_at, alert.is_test)

    alert_id = str(row["id"])
    log.info(f"Alert created: id={alert_id} conf={alert.confidence:.2f}")
    return {"status": "ok", "alert_id": alert_id}


@alert_router.delete("/alerts/expired")
async def clear_expired_alerts():
    pool: asyncpg.Pool = await get_pool()
    result = await pool.execute("DELETE FROM ice_alerts WHERE expires_at < now()")
    deleted = result.split()[-1]
    log.info(f"Cleared {deleted} expired alerts")
    return {"status": "ok", "deleted": int(deleted)}


@alert_router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    pool: asyncpg.Pool = await get_pool()
    await pool.execute("DELETE FROM ice_alerts WHERE id = $1", alert_id)
    log.info(f"Deleted alert: {alert_id}")
    return {"status": "ok"}
