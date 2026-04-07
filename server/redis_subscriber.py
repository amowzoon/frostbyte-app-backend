"""
redis_subscriber.py
-------------------
Background task that subscribes to keyspace notifications on the shared Redis.
When frostbyte-backend pushes a detection entry (detection:<device_id>),
this task reads the newest entry, parses it, and writes an ice alert to Supabase.

Detection entry format (set by frostbyte-backend / publish_test_detection.py):
    <json_bytes>\n---MASK---\n<png_bytes>

JSON fields used:
    device_id   str
    latitude    float
    longitude   float
    timestamp   ISO-8601 str
    (confidence is not in the detection payload yet — defaults to 0.7 until
     Connor's ML pipeline pushes it; update DEFAULT_CONFIDENCE when ready)
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
import redis.asyncio as aioredis

log = logging.getLogger("app.subscriber")

SEPARATOR = b"\n---MASK---\n"
DEFAULT_CONFIDENCE = 0.7       # placeholder until ML pushes confidence
ALERT_EXPIRES_MINUTES = 60
DEFAULT_ALERT_TYPE = "heuristic"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SHARED_REDIS_URL = os.environ["SHARED_REDIS_URL"]

SUPABASE_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}


def parse_detection(raw: bytes) -> dict | None:
    """Parse a raw Redis detection entry into a dict. Returns None on failure."""
    try:
        parts = raw.split(SEPARATOR, 1)
        if len(parts) < 1:
            return None
        return json.loads(parts[0].decode("utf-8"))
    except Exception as e:
        log.warning(f"Failed to parse detection entry: {e}")
        return None


async def write_alert_to_supabase(client: httpx.AsyncClient, meta: dict) -> str | None:
    """Insert an ice alert row into Supabase. Returns the new alert_id or None."""
    expires_at = (
        datetime.now(timezone.utc) + timedelta(minutes=ALERT_EXPIRES_MINUTES)
    ).isoformat()

    payload = {
        "latitude":   meta["latitude"],
        "longitude":  meta["longitude"],
        "confidence": meta.get("confidence", DEFAULT_CONFIDENCE),
        "alert_type": meta.get("alert_type", DEFAULT_ALERT_TYPE),
        "device_id":  meta.get("device_id"),
        "expires_at": expires_at,
        "active":     True,
        "is_test":    False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    r = await client.post(
        f"{SUPABASE_URL}/rest/v1/ice_alerts",
        headers=SUPABASE_HEADERS,
        json=payload,
    )

    if r.status_code not in (200, 201):
        log.error(f"Supabase insert failed ({r.status_code}): {r.text}")
        return None

    data = r.json()
    alert_id = data[0]["id"] if data else None
    log.info(
        f"Alert created: id={alert_id} "
        f"device={meta.get('device_id')} "
        f"lat={meta['latitude']:.5f} lon={meta['longitude']:.5f} "
        f"conf={payload['confidence']:.2f}"
    )
    return alert_id


async def handle_detection(redis_client: aioredis.Redis, key: str) -> None:
    """Read the newest detection from a key and write an alert to Supabase."""
    try:
        entries = await redis_client.lrange(key, 0, 0)  # newest entry only
        if not entries:
            log.warning(f"Keyspace event for {key} but list is empty")
            return

        meta = parse_detection(entries[0])
        if not meta:
            return

        if not all(k in meta for k in ("latitude", "longitude")):
            log.warning(f"Detection missing lat/lon: {meta}")
            return

        async with httpx.AsyncClient(timeout=10.0) as http:
            await write_alert_to_supabase(http, meta)

    except Exception as e:
        log.error(f"handle_detection error for {key}: {e}")


async def run_subscriber() -> None:
    """
    Subscribe to keyspace notifications on detection:* keys.
    Runs forever — restarts automatically on connection failure.
    """
    log.info(f"Redis subscriber connecting to {SHARED_REDIS_URL}")

    while True:
        try:
            client = aioredis.from_url(
                SHARED_REDIS_URL,
                decode_responses=False,
                socket_connect_timeout=5,
            )
            await client.ping()
            log.info("Redis subscriber connected")

            # Keyspace notifications: __keyevent@0__:lpush fires when any key gets an lpush
            pubsub = client.pubsub()
            await pubsub.psubscribe("__keyevent@0__:lpush")
            log.info("Subscribed to __keyevent@0__:lpush")

            async for message in pubsub.listen():
                if message["type"] != "pmessage":
                    continue
                key = message["data"].decode("utf-8")
                if not key.startswith("detection:"):
                    continue
                log.info(f"Detection event on key: {key}")
                await handle_detection(client, key)

        except asyncio.CancelledError:
            log.info("Redis subscriber shutting down")
            return
        except Exception as e:
            log.error(f"Redis subscriber error: {e} — retrying in 5s")
            await asyncio.sleep(5)