#!/usr/bin/env python3
"""
Publish a test detection to Redis and verify the full pipeline:
1. Publishes detection to Redis
2. Waits for subscriber to process it
3. Checks Redis for the key
4. Checks Postgres for the resulting alert
5. Checks the backend HTTP endpoint for the alert

Usage:
    python scripts/publish_test_detection.py
"""
from __future__ import annotations

import json
import os
import struct
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import redis
import urllib.request

DETECTION_KEY_PREFIX = "detection:"
DETECTION_TTL_SECONDS = 3600
MAX_DETECTIONS_PER_DEVICE = 2
SEPARATOR = b"\n---MASK---\n"

DEVICE_ID   = "test-pi"
LATITUDE    = 42.348555
LONGITUDE   = -71.116347
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")


def _make_test_mask(width: int = 64, height: int = 64) -> bytes:
    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw_data = b"".join(b"\x00" + bytes([128] * width) for _ in range(height))
    idat = zlib.compress(raw_data)
    return signature + make_chunk(b"IHDR", ihdr) + make_chunk(b"IDAT", idat) + make_chunk(b"IEND", b"")


def publish_detection(client: redis.Redis) -> str:
    key = f"{DETECTION_KEY_PREFIX}{DEVICE_ID}"
    entry = {
        "device_id": DEVICE_ID,
        "geotag": "Test Location",
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "capture_id": None,
        "session_id": None,
        "metadata": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mask": None,
    }
    payload = json.dumps(entry).encode("utf-8") + SEPARATOR + _make_test_mask()
    pipe = client.pipeline()
    pipe.lpush(key, payload)
    pipe.ltrim(key, 0, MAX_DETECTIONS_PER_DEVICE - 1)
    pipe.expire(key, DETECTION_TTL_SECONDS)
    pipe.execute()
    return key


def main() -> None:
    password = os.environ.get("REDIS_PASSWORD")
    if not password:
        raise SystemExit("Error: REDIS_PASSWORD not set.")

    print("=" * 50)
    print("FrostByte Pipeline Test")
    print("=" * 50)

    # Step 1: Connect to Redis
    print("\n[1/4] Connecting to Redis...")
    client = redis.Redis(host="localhost", port=6380, password=password, decode_responses=False)
    client.ping()
    print("      Redis connected")

    # Step 2: Publish detection
    print(f"\n[2/4] Publishing detection at ({LATITUDE}, {LONGITUDE})...")
    key = publish_detection(client)
    list_len = client.llen(key)
    print(f"      Published to key={key}  list_len={list_len}")

    # Step 3: Wait and check Redis
    print("\n[3/4] Checking Redis for key...")
    time.sleep(1)
    keys = client.keys("detection:*")
    print(f"      Keys in Redis: {[k.decode() for k in keys]}")

    # Step 4: Wait for subscriber and check backend
    print("\n[4/4] Waiting for subscriber to write alert to Postgres...")
    time.sleep(3)
    try:
        url = f"{BACKEND_URL}/api/app/alerts/nearby?lat={LATITUDE}&lon={LONGITUDE}&radius_m=500"
        with urllib.request.urlopen(url, timeout=5) as res:
            data = json.loads(res.read())
            alerts = data.get("alerts", [])
            if alerts:
                print(f"      Alert found in Postgres and returned by backend!")
                for a in alerts:
                    print(f"      id={a['id']} conf={a['confidence']} lat={a['latitude']} lon={a['longitude']}")
            else:
                print("      No alerts found near test location yet — check docker logs frostbyte-app-server")
    except Exception as e:
        print(f"      Backend check failed: {e}")

    print("\n" + "=" * 50)
    print("Done. Check the app map for the alert.")
    print("=" * 50)


if __name__ == "__main__":
    main()
