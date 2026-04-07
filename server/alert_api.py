"""
alert_api.py
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import httpx
import os
import logging
from datetime import datetime, timedelta, timezone

log = logging.getLogger("app.alerts")

alert_router = APIRouter(prefix="/api/app")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PushTokenRequest(BaseModel):
    user_id: str
    push_token: str

class UserSettings(BaseModel):
    user_id: str
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
async def store_push_token(req: PushTokenRequest):
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.patch(
            f"{SUPABASE_URL}/rest/v1/user_preferences",
            headers=HEADERS,
            params={"user_id": f"eq.{req.user_id}"},
            json={"push_token": req.push_token, "updated_at": datetime.now(timezone.utc).isoformat()},
        )
        if r.status_code == 404 or r.json() == []:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/user_preferences",
                headers=HEADERS,
                json={"user_id": req.user_id, "push_token": req.push_token},
            )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

@alert_router.get("/settings")
async def get_settings(user_id: str = Query(...)):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/user_preferences",
            headers=HEADERS,
            params={"user_id": f"eq.{user_id}", "select": "alert_radius_m,notify_ice,notify_bluetooth,notify_route"},
        )
    data = r.json()
    if not data:
        return {"alert_radius_m": 500, "notify_ice": True, "notify_bluetooth": True, "notify_route": True}
    return data[0]


@alert_router.patch("/settings")
async def update_settings(settings: UserSettings):
    payload = {k: v for k, v in settings.dict().items() if k != "user_id" and v is not None}
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient() as client:
        r = await client.patch(
            f"{SUPABASE_URL}/rest/v1/user_preferences",
            headers=HEADERS,
            params={"user_id": f"eq.{settings.user_id}"},
            json=payload,
        )
        if r.json() == []:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/user_preferences",
                headers=HEADERS,
                json={"user_id": settings.user_id, **payload},
            )
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
    now = datetime.now(timezone.utc).isoformat()

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/ice_alerts",
            headers=HEADERS,
            params={
                "select": "id,latitude,longitude,confidence,alert_type,device_id,created_at,expires_at",
                "active": "eq.true",
                "is_test": "eq.false",
                "expires_at": f"gt.{now}",
                "order": "confidence.desc",
                "limit": "50",
            },
        )

    alerts = r.json() if r.status_code == 200 else []
    return {"alerts": alerts, "count": len(alerts)}


@alert_router.post("/alerts")
async def create_alert(alert: AlertCreate):
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=alert.expires_minutes)).isoformat()
    payload = {
        "latitude": alert.latitude,
        "longitude": alert.longitude,
        "confidence": alert.confidence,
        "alert_type": alert.alert_type,
        "device_id": alert.device_id,
        "expires_at": expires_at,
        "active": True,
        "is_test": alert.is_test,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/ice_alerts",
            headers=HEADERS,
            json=payload,
        )

    if r.status_code not in (200, 201):
        log.error(f"Failed to create alert: {r.text}")
        raise HTTPException(500, "Failed to create alert in Supabase")

    data = r.json()
    alert_id = data[0]["id"] if data else None
    log.info(f"Alert created: id={alert_id} conf={alert.confidence:.2f}")
    return {"status": "ok", "alert_id": alert_id}