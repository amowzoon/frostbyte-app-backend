"""
test_pseudo_alert.py
Injects a fake ice alert that mimics exactly what the real inference pipeline produces.

Run from frostbyte-backend directory inside the running Docker container:
    docker compose exec server python /app/test_pseudo_alert.py

Or from outside the container:
    docker compose exec server python /app/test_pseudo_alert.py --lat 42.3505 --lon -71.1054 --confidence 0.82

What the real pipeline does (from processing_api.py run_inference):
    1. Runs heuristic model on captured session, produces mean_conf
    2. If mean_conf > 0.5, fetches device lat/lon from devices table via session_id
    3. POSTs to /api/app/alerts with session_id, device_id, lat, lon, confidence
    4. alert_api.py inserts into ice_alerts table with a 2-hour expiry
    5. Sends Expo push notifications to nearby users

This script replicates steps 2-5 exactly, using a real session_id and device_id
from the database so the foreign key constraints are satisfied.

Test alerts are marked is_test=TRUE in the database so they are never shown
to real app users — the /api/app/alerts/nearby endpoint filters them out.
They are visible in the raw database and /docs interface for verification only.
"""

import sys
import uuid
import argparse
import json
import psycopg2
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config — matches docker-compose.yml
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": "postgres",
    "port": 5432,
    "dbname": "frostbyte",
    "user": "frosty",
    "password": "fr0stbyte",
}

ALERT_EXPIRE_HOURS = 2
ICE_CONFIDENCE_THRESHOLD = 0.5


def get_db():
    return psycopg2.connect(**DB_CONFIG)


def get_real_session(conn):
    """
    Get the most recent completed capture session and its device's GPS coords.
    This mirrors what processing_api.py does when it queries for device lat/lon.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT cs.id, cs.capture_id, cs.device_id, d.latitude, d.longitude
        FROM capture_sessions cs
        JOIN devices d ON cs.device_id = d.id
        WHERE cs.is_complete = TRUE
        ORDER BY cs.created_at DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    cur.close()
    return row


def get_any_device(conn):
    """
    Fall back to any registered device if no completed sessions exist.
    """
    cur = conn.cursor()
    cur.execute("SELECT id, latitude, longitude FROM devices LIMIT 1")
    row = cur.fetchone()
    cur.close()
    return row


def insert_alert(conn, session_id, device_id, lat, lon, confidence, is_test=True):
    """
    Insert into ice_alerts exactly as alert_api.py does in create_alert().
    Uses the same column order, same expiry calculation, same UUID generation.

    When no real device exists in the devices table (Pi not yet registered),
    device_id is omitted to avoid a foreign key violation. This is test-only
    behavior — the real pipeline always has a valid device_id.
    """
    alert_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(hours=ALERT_EXPIRE_HOURS)

    # Check if the device_id actually exists in the devices table
    cur = conn.cursor()
    if device_id:
        cur.execute("SELECT id FROM devices WHERE id = %s", (device_id,))
        device_exists = cur.fetchone() is not None
    else:
        device_exists = False

    if device_exists:
        cur.execute("""
            INSERT INTO ice_alerts
                (id, session_id, device_id, latitude, longitude, confidence, expires_at, is_test)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, created_at, expires_at
        """, (
            alert_id,
            str(session_id) if session_id else None,
            device_id,
            lat,
            lon,
            confidence,
            expires_at,
            is_test,
        ))
    else:
        # Pi not registered yet — omit device_id and session_id to avoid FK violation
        print(f"  Note: device '{device_id}' not in devices table — inserting without device_id (test only)")
        cur.execute("""
            INSERT INTO ice_alerts
                (id, latitude, longitude, confidence, expires_at, is_test)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, created_at, expires_at
        """, (
            alert_id,
            lat,
            lon,
            confidence,
            expires_at,
            is_test,
        ))

    row = cur.fetchone()
    conn.commit()
    cur.close()
    return row


def verify_alert_appears(conn, lat, lon, radius_m=1000):
    """
    Run the same query that /api/app/alerts/nearby uses to confirm
    the alert will actually appear in the app.
    """
    deg_offset = radius_m / 111000.0
    cur = conn.cursor()
    cur.execute("""
        SELECT id, latitude, longitude, confidence, created_at, expires_at, device_id, is_test
        FROM ice_alerts
        WHERE active = TRUE
          AND expires_at > NOW()
          AND latitude  BETWEEN %s AND %s
          AND longitude BETWEEN %s AND %s
        ORDER BY confidence DESC
        LIMIT 10
    """, (
        lat - deg_offset, lat + deg_offset,
        lon - deg_offset, lon + deg_offset,
    ))
    rows = cur.fetchall()
    cur.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Inject a pseudo ice alert for testing")
    parser.add_argument("--lat",        type=float, default=None,  help="Alert latitude (default: use device GPS or BU coords)")
    parser.add_argument("--lon",        type=float, default=None,  help="Alert longitude (default: use device GPS or BU coords)")
    parser.add_argument("--confidence", type=float, default=0.78,  help="Ice confidence 0.0-1.0 (default: 0.78)")
    parser.add_argument("--device-id",  type=str,   default=None,  help="Override device_id (default: use most recent session)")
    parser.add_argument("--verify-radius", type=int, default=1000, help="Radius in meters to verify alert appears in app query (default: 1000)")
    parser.add_argument("--no-test-flag", action="store_true", help="Insert as a real alert (is_test=FALSE) so it appears in the app. Default is is_test=TRUE which is filtered out from app users.")
    args = parser.parse_args()

    if args.confidence <= ICE_CONFIDENCE_THRESHOLD:
        print(f"WARNING: confidence {args.confidence} is at or below threshold {ICE_CONFIDENCE_THRESHOLD}")
        print("The real pipeline would skip this alert. Use --confidence > 0.5")
        sys.exit(1)

    print("Connecting to database...")
    conn = get_db()

    # Step 1: get a real session and device — same as processing_api.py
    session_id = None
    device_id = args.device_id
    lat = args.lat
    lon = args.lon

    if not device_id or lat is None or lon is None:
        print("Looking up most recent completed session...")
        row = get_real_session(conn)

        if row:
            session_id, capture_id, db_device_id, db_lat, db_lon = row
            print(f"  Found session:   {capture_id}")
            print(f"  Device:          {db_device_id}")
            print(f"  Device GPS:      {db_lat}, {db_lon}")

            if not device_id: device_id = db_device_id
            if lat is None:   lat = db_lat
            if lon is None:   lon = db_lon
        else:
            print("No completed sessions found, falling back to any device...")
            dev_row = get_any_device(conn)
            if dev_row:
                db_device_id, db_lat, db_lon = dev_row
                if not device_id: device_id = db_device_id
                if lat is None:   lat = db_lat
                if lon is None:   lon = db_lon
                print(f"  Device: {device_id}  GPS: {lat}, {lon}")

    # If device has no GPS coords stored, use BU campus as default
    if lat is None or lon is None:
        lat = 42.3505
        lon = -71.1054
        print(f"  No device GPS found — using BU coordinates: {lat}, {lon}")

    if not device_id:
        device_id = "pi-001"
        print(f"  No device found — using default: {device_id}")

    print()
    print("Injecting alert with the following parameters:")
    print(f"  session_id:  {session_id}")
    print(f"  device_id:   {device_id}")
    print(f"  latitude:    {lat}")
    print(f"  longitude:   {lon}")
    print(f"  confidence:  {args.confidence} ({round(args.confidence * 100)}%)")
    print(f"  expires_at:  {datetime.utcnow() + timedelta(hours=ALERT_EXPIRE_HOURS)} UTC")
    print()

    # Step 2: insert into ice_alerts exactly as alert_api.py does
    is_test = not args.no_test_flag
    if not is_test:
        print("WARNING: inserting as real alert (is_test=FALSE) — will appear in the app for all users.")
    alert_id, created_at, expires_at = insert_alert(
        conn, session_id, device_id, lat, lon, args.confidence, is_test
    )

    print(f"Alert inserted successfully.")
    print(f"  alert_id:    {alert_id}")
    print(f"  created_at:  {created_at} UTC")
    print(f"  expires_at:  {expires_at} UTC")
    print()

    # Step 3: verify it appears in the same query the app uses
    print(f"Verifying alert appears in /api/app/alerts/nearby (radius={args.verify_radius}m)...")
    nearby = verify_alert_appears(conn, lat, lon, args.verify_radius)

    if nearby:
        print(f"  {len(nearby)} alert(s) found in query zone:")
        for r in nearby:
            test_flag = "  [TEST]" if r[7] else ""
            print(f"    id={r[0]}  conf={round(r[3]*100)}%  device={r[6]}{test_flag}")
        print()
        print("The alert will appear in the app within 30 seconds (next poll cycle).")
    else:
        print("  WARNING: Alert not found in nearby query.")
        print("  Check that the device has latitude/longitude set in the devices table.")
        print("  You can override with: --lat <lat> --lon <lon>")

    conn.close()


if __name__ == "__main__":
    main()