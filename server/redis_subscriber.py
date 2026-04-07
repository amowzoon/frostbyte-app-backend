"""
redis_subscriber.py
-------------------
Background task: subscribes to keyspace notifications on shared Redis.
On detection:* lpush events, reads the newest entry and writes an
ice_alert row directly to local Postgres.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import asyncpg
import redis.asyncio as aioredis

from db import get_pool

log = logging.getLogger("app.subscriber")

SEPARATOR = b"\n---MASK---\n"
DEFAULT_CONFIDENCE = 0.7
ALERT_EXPIRES_MINUTES = 60
SHARED_REDIS_URL = os.environ["SHARED_REDIS_URL"]


def parse_detection(raw: bytes) -> dict | None:
    try:
        parts = raw.split(SEPARATOR, 1)
        return json.loads(parts[0].decode("utf-8"))
    except Exception as e:
        log.warning(f"Failed to parse detection: {e}")
        return None


async def write_alert(meta: dict):
    pool: asyncpg.Pool = await get_pool()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ALERT_EXPIRES_MINUTES)
    row = await pool.fetchrow("""
        INSERT INTO ice_alerts (latitude, longitude, confidence, alert_type, device_id, expires_at, active, is_test)
        VALUES ($1, $2, $3, $4, $5, $6, true, false)
        RETURNING id
    """,
        meta["latitude"],
        meta["longitude"],
        meta.get("confidence", DEFAULT_CONFIDENCE),
        meta.get("alert_type", "heuristic"),
        meta.get("device_id"),
        expires_at,
    )
    log.info(
        f"Alert created: id={row['id']} device={meta.get('device_id')} "
        f"lat={meta['latitude']:.5f} lon={meta['longitude']:.5f} "
        f"conf={meta.get('confidence', DEFAULT_CONFIDENCE):.2f}"
    )


async def handle_detection(redis_client: aioredis.Redis, key: str):
    try:
        entries = await redis_client.lrange(key, 0, 0)
        if not entries:
            return
        meta = parse_detection(entries[0])
        if not meta:
            return
        if not all(k in meta for k in ("latitude", "longitude")):
            log.warning(f"Detection missing lat/lon: {meta}")
            return
        await write_alert(meta)
    except Exception as e:
        log.error(f"handle_detection error for {key}: {e}")


async def run_subscriber():
    log.info(f"Redis subscriber connecting to {SHARED_REDIS_URL}")
    while True:
        try:
            client = aioredis.from_url(SHARED_REDIS_URL, decode_responses=False, socket_connect_timeout=5)
            await client.ping()
            log.info("Redis subscriber connected")

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