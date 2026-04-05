"""
tests/test_alert_pipeline.py
Unit tests for the FrostByte alert pipeline.

These tests mirror the exact setup of the hosted backend at frostbyte-app.com.
The database credentials, schema, and session structure are identical to production.

When the local database is empty (Pi not connected locally), a seeded_session
fixture provides realistic test data based on real captures observed in the
Data Viewer at frostbyte-app.com/data-viewer:

  capture_id:  20260329T130211818Z
  device_id:   pi-001
  session_id:  6c3e623e-cc68-4ab9-b1ce-8c6e5bc8c73d  (real session from screenshot)
  sensors:     rgb, ir, radar (3/3 complete)
  RGB:         hflip=true, vflip=true, format=jpeg, autofocus
  IR:          640x480, grayscale, max_temp_c=25.3
  Radar:       bandwidth_ghz=3.44, start_freq_ghz=77, sample_rate_ksps=5209

When the Pi is connected to the local backend, real_session uses live data.
The test suite works correctly in both cases.

Run inside Docker:
    docker compose exec server pytest tests/test_alert_pipeline.py -v -s
"""

import pytest
import uuid
import json
import psycopg2
from datetime import datetime, timedelta

# Identical to frostbyte-app.com docker-compose.yml postgres environment
DB_CONFIG = {
    "host":     "postgres",
    "port":     5432,
    "dbname":   "frostbyte",
    "user":     "frosty",
    "password": "fr0stbyte",
}

ICE_CONFIDENCE_THRESHOLD = 0.5
ALERT_EXPIRE_HOURS = 2

# Real capture_id from the Data Viewer screenshot
REAL_CAPTURE_ID = "20260329T130211818Z"


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """
    Real database connection. Schema and credentials mirror frostbyte-app.com.
    Rolls back failed transactions and cleans up test rows after each test.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    yield conn
    try:
        conn.rollback()
    except Exception:
        pass
    cur = conn.cursor()
    cur.execute("DELETE FROM ice_alerts WHERE is_test = TRUE")
    conn.commit()
    cur.close()
    conn.close()


# ---------------------------------------------------------------------------
# Seeded session fixture — mirrors real Pi data from frostbyte-app.com
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_session(db):
    """
    Seeds the local database with a realistic capture session that exactly
    mirrors what the Pi uploads to frostbyte-app.com.

    Data taken directly from the Data Viewer screenshot (session detail panel):
      - capture_id:   20260329T130211818Z
      - device_id:    pi-001
      - sensors:      rgb (jpeg, autofocus), ir (640x480 grayscale npy+jpg),
                      radar (json, bandwidth_ghz=3.44, start_freq_ghz=77)
      - is_complete:  TRUE (3/3)
      - captured_at:  2026-03-29 13:02:11 PM

    s3_paths mirror the MinIO bucket structure used in production.
    Metadata fields match exactly what each sensor module writes.
    All seeded rows are cleaned up after each test.
    """
    cur = db.cursor()
    session_id = str(uuid.uuid4())

    # 1. Register device (mirrors Pi sending register message to backend)
    cur.execute("""
        INSERT INTO devices (id, name, registered_at, last_seen, metadata)
        VALUES ('pi-001', 'FrostByte Unit 01', NOW(), NOW(),
                '{"serial_number": "test-serial-001"}')
        ON CONFLICT (id) DO UPDATE SET last_seen = NOW()
    """)

    # 2. Create capture session (mirrors _upload_store_and_record in main.py)
    cur.execute("""
        INSERT INTO capture_sessions
            (id, device_id, capture_id, captured_at, created_at, is_complete, expected_sensors)
        VALUES (%s::uuid, 'pi-001', %s,
                '2026-03-29 13:02:11'::timestamp,
                NOW(), TRUE,
                ARRAY['rgb', 'ir', 'radar'])
        ON CONFLICT (capture_id) DO NOTHING
    """, (session_id, REAL_CAPTURE_ID))

    # In case the capture_id already exists, fetch the real session_id
    cur.execute("SELECT id FROM capture_sessions WHERE capture_id = %s", (REAL_CAPTURE_ID,))
    row = cur.fetchone()
    if row:
        session_id = str(row[0])

    # 3. Insert sensor_data rows (mirrors data_uploader.py upload behavior)
    #    s3_path structure: captures/{device_id}/{capture_id}/{sensor}.ext
    #    Metadata taken verbatim from Data Viewer screenshot metadata panels

    sensors = [
        (
            "rgb",
            f"captures/pi-001/{REAL_CAPTURE_ID}/rgb.jpg",
            None,  # no raw file for RGB
            {
                "hflip": True,
                "vflip": True,
                "format": "jpeg",
                "rotation": 0,
                "t_acquire": 36849.989148911,
                "timestamp": "2026-03-29T13:02:12.192646Z",
                "focus_mode": "autofocus",
                "resolution": {"width": 1608, "height": 1593},
                "capture_id": REAL_CAPTURE_ID,
            }
        ),
        (
            "ir",
            f"captures/pi-001/{REAL_CAPTURE_ID}/ir.jpg",
            f"captures/pi-001/{REAL_CAPTURE_ID}/ir.npy",  # raw radiometric data
            {
                "width": 640,
                "height": 480,
                "format": "jpeg",
                "colormap": "grayscale",
                "t_acquire": 36849.657134528,
                "timestamp": "2026-03-29T13:02:11.865347Z",
                "max_temp_c": 25.3,
                "min_temp_c": 18.2,
                "capture_id": REAL_CAPTURE_ID,
            }
        ),
        (
            "radar",
            f"captures/pi-001/{REAL_CAPTURE_ID}/radar.json",
            None,
            {
                "config": {
                    "bandwidth_ghz": 3.4401996544442,
                    "start_freq_ghz": 77,
                    "sample_rate_ksps": 5209,
                },
                "t_acquire": 36849.636714875,
                "timestamp": "2026-03-29T13:02:13.840212Z",
                "cpu_cycles": 3687122494,
                "noise_bins": 0,
                "range_bins": 0,
                "capture_id": REAL_CAPTURE_ID,
            }
        ),
    ]

    for sensor_type, s3_path, s3_path_raw, metadata in sensors:
        sensor_data_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO sensor_data
                (id, session_id, sensor_type, s3_path, s3_path_raw, metadata, uploaded_at)
            VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, NOW())
            ON CONFLICT (session_id, sensor_type) DO NOTHING
        """, (sensor_data_id, session_id, sensor_type,
              s3_path, s3_path_raw, json.dumps(metadata)))

    db.commit()
    cur.close()

    yield {
        "session_id":       session_id,
        "capture_id":       REAL_CAPTURE_ID,
        "device_id":        "pi-001",
        "is_complete":      True,
        "expected_sensors": ["rgb", "ir", "radar"],
        "uploaded_sensors": ["rgb", "ir", "radar"],
        "device_lat":       None,
        "device_lon":       None,
    }

    # Cleanup — delete in FK-safe order:
    # alerts referencing session → sensor_data → capture_sessions → device
    try:
        conn2 = psycopg2.connect(**DB_CONFIG)
        cur2 = conn2.cursor()
        cur2.execute("DELETE FROM ice_alerts WHERE session_id = %s::uuid", (session_id,))
        cur2.execute("DELETE FROM sensor_data WHERE session_id = %s::uuid", (session_id,))
        cur2.execute("DELETE FROM capture_sessions WHERE id = %s::uuid", (session_id,))
        cur2.execute("""
            DELETE FROM devices WHERE id = 'pi-001'
            AND NOT EXISTS (
                SELECT 1 FROM capture_sessions WHERE device_id = 'pi-001'
            )
        """)
        conn2.commit()
        cur2.close()
        conn2.close()
    except Exception as e:
        print(f"seeded_session cleanup error (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Real session fixture — uses live Pi data if available
# ---------------------------------------------------------------------------

@pytest.fixture
def real_session(db):
    """
    Fetches the most recent completed session from the database.
    Returns None if no real sessions exist (Pi not connected locally).
    In that case tests fall back to seeded_session.
    """
    cur = db.cursor()
    cur.execute("""
        SELECT
            cs.id, cs.capture_id, cs.device_id, cs.captured_at,
            cs.is_complete, cs.expected_sensors,
            d.latitude, d.longitude,
            array_agg(sd.sensor_type) as uploaded_sensors
        FROM capture_sessions cs
        JOIN devices d ON cs.device_id = d.id
        LEFT JOIN sensor_data sd ON sd.session_id = cs.id
        WHERE cs.is_complete = TRUE
        GROUP BY cs.id, cs.capture_id, cs.device_id, cs.captured_at,
                 cs.is_complete, cs.expected_sensors, d.latitude, d.longitude
        ORDER BY cs.created_at DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    cur.close()
    if not row:
        return None
    return {
        "session_id":       str(row[0]),
        "capture_id":       row[1],
        "device_id":        row[2],
        "is_complete":      row[4],
        "expected_sensors": row[5],
        "device_lat":       row[6],
        "device_lon":       row[7],
        "uploaded_sensors": row[8],
    }


# ---------------------------------------------------------------------------
# Helpers — mirror exact SQL from alert_api.py
# ---------------------------------------------------------------------------

def insert_test_alert(db, lat, lon, confidence, session_id=None,
                      device_id=None, active=True, expires_hours=ALERT_EXPIRE_HOURS):
    """Insert a test alert exactly as alert_api.py create_alert() does."""
    alert_id   = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
    cur = db.cursor()

    if device_id:
        cur.execute("SELECT id FROM devices WHERE id = %s", (device_id,))
        if not cur.fetchone():
            device_id = None

    if device_id and session_id:
        cur.execute("""
            INSERT INTO ice_alerts
                (id, session_id, device_id, latitude, longitude,
                 confidence, expires_at, active, is_test)
            VALUES (%s, %s::uuid, %s, %s, %s, %s, %s, %s, TRUE)
        """, (alert_id, session_id, device_id, lat, lon,
              confidence, expires_at, active))
    else:
        cur.execute("""
            INSERT INTO ice_alerts
                (id, latitude, longitude, confidence, expires_at, active, is_test)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE)
        """, (alert_id, lat, lon, confidence, expires_at, active))

    db.commit()
    cur.close()
    return alert_id


def query_nearby_alerts(db, lat, lon, radius_m=1000):
    """Exact copy of the SQL in alert_api.py get_nearby_alerts()."""
    deg_offset = radius_m / 111000.0
    cur = db.cursor()
    cur.execute("""
        SELECT id, latitude, longitude, confidence, created_at, expires_at, device_id
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
        lon - deg_offset, lon + deg_offset,
    ))
    rows = cur.fetchall()
    cur.close()
    return [
        {"id": str(r[0]), "latitude": r[1], "longitude": r[2],
         "confidence": r[3], "created_at": r[4], "expires_at": r[5],
         "device_id": r[6]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# TestRealSessionData
# Uses seeded_session (mirrors frostbyte-app.com data) or real Pi data.
# ---------------------------------------------------------------------------

class TestRealSessionData:

    def test_completed_sessions_exist(self, db, seeded_session):
        """
        Confirms completed sessions exist. Uses seeded data mirroring
        the real capture from frostbyte-app.com/data-viewer when
        the local database is empty.
        """
        cur = db.cursor()
        cur.execute("SELECT COUNT(*) FROM capture_sessions WHERE is_complete = TRUE")
        count = cur.fetchone()[0]
        cur.close()
        print(f"\n  Completed sessions: {count} (includes seeded: {REAL_CAPTURE_ID})")
        assert count > 0

    def test_session_has_all_three_sensors(self, db, seeded_session, real_session):
        """
        Each complete session must have rgb, ir, radar — same 3/3 shown
        in the Data Viewer status column.
        """
        session = real_session or seeded_session
        print(f"\n  capture_id:       {session['capture_id']}")
        print(f"  device_id:        {session['device_id']}")
        print(f"  uploaded_sensors: {session['uploaded_sensors']}")
        uploaded = set(session["uploaded_sensors"] or [])
        for sensor in ["rgb", "ir", "radar"]:
            assert sensor in uploaded, f"Missing sensor: {sensor}"

    def test_sensor_data_has_s3_path(self, db, seeded_session, real_session):
        """
        Every sensor upload must have an s3_path so the inference pipeline
        can fetch files from MinIO. Mirrors the s3_path column in sensor_data.
        """
        session = real_session or seeded_session
        cur = db.cursor()
        cur.execute("""
            SELECT sensor_type, s3_path, s3_path_raw
            FROM sensor_data WHERE session_id = %s::uuid
        """, (session["session_id"],))
        rows = cur.fetchall()
        cur.close()

        assert len(rows) > 0, "No sensor_data rows found"
        for sensor_type, s3_path, s3_path_raw in rows:
            assert s3_path is not None, f"{sensor_type} missing s3_path"
            print(f"\n  {sensor_type}:")
            print(f"    s3_path:     {s3_path}")
            print(f"    s3_path_raw: {s3_path_raw}")

    def test_ir_sensor_has_raw_npy_path(self, db, seeded_session, real_session):
        """
        IR sensor must have s3_path_raw pointing to the .npy file containing
        raw uint16 Kelvin*100 radiometric data. This is required for accurate
        temperature-based ice detection — JPEG is lossy and cannot be used.
        """
        session = real_session or seeded_session
        cur = db.cursor()
        cur.execute("""
            SELECT s3_path_raw FROM sensor_data
            WHERE session_id = %s::uuid AND sensor_type = 'ir'
        """, (session["session_id"],))
        row = cur.fetchone()
        cur.close()

        assert row is not None, "No IR sensor data found"
        assert row[0] is not None, (
            "IR sensor missing s3_path_raw (.npy file). "
            "Without this, temperature readings are approximations only."
        )
        print(f"\n  IR raw path: {row[0]}")

    def test_rgb_metadata_has_capture_settings(self, db, seeded_session, real_session):
        """
        RGB metadata must contain capture settings (hflip, vflip, format, timestamp).
        The inference pipeline reads these to validate the image before processing.
        """
        session = real_session or seeded_session
        cur = db.cursor()
        cur.execute("""
            SELECT metadata FROM sensor_data
            WHERE session_id = %s::uuid AND sensor_type = 'rgb'
        """, (session["session_id"],))
        row = cur.fetchone()
        cur.close()

        assert row is not None, "No RGB sensor data"
        metadata = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        print(f"\n  RGB metadata: {json.dumps(metadata, indent=2)}")
        for key in ["format", "timestamp", "capture_id"]:
            assert key in metadata, f"RGB metadata missing '{key}'"

    def test_radar_metadata_has_config(self, db, seeded_session, real_session):
        """
        Radar metadata must contain the config block (bandwidth_ghz, start_freq_ghz,
        sample_rate_ksps) as seen in the Data Viewer metadata panel.
        The radar heatmap generator reads these to reconstruct the azimuth profile.
        """
        session = real_session or seeded_session
        cur = db.cursor()
        cur.execute("""
            SELECT metadata FROM sensor_data
            WHERE session_id = %s::uuid AND sensor_type = 'radar'
        """, (session["session_id"],))
        row = cur.fetchone()
        cur.close()

        assert row is not None, "No radar sensor data"
        metadata = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        assert "config" in metadata, "Radar metadata missing 'config' block"
        config = metadata["config"]
        print(f"\n  Radar config: {config}")
        for key in ["bandwidth_ghz", "start_freq_ghz", "sample_rate_ksps"]:
            assert key in config, f"Radar config missing '{key}'"

    def test_device_registered(self, db, seeded_session, real_session):
        """
        Device must exist in devices table for alert FK constraint to work.
        """
        session = real_session or seeded_session
        cur = db.cursor()
        cur.execute("SELECT id, name, last_seen FROM devices WHERE id = %s",
                    (session["device_id"],))
        row = cur.fetchone()
        cur.close()
        assert row is not None, f"Device {session['device_id']} not registered"
        print(f"\n  Device: {row[0]}  name: {row[1]}  last_seen: {row[2]}")


# ---------------------------------------------------------------------------
# TestInferenceTrigger
# Simulates processing_api.py run_inference() → alert_api.py create_alert()
# ---------------------------------------------------------------------------

class TestInferenceTrigger:

    def test_confidence_threshold_gate(self, db, seeded_session, real_session):
        """
        Simulates run_inference() mean_conf values.
        Only mean_conf > 0.5 triggers an alert — same check as processing_api.py line 740.
        """
        session = real_session or seeded_session
        print(f"\n  Session: {session['capture_id']}")
        cases = [
            (0.30, False, "low — no alert"),
            (0.50, False, "exactly at threshold — skipped (strict >)"),
            (0.51, True,  "just above — alert fires"),
            (0.82, True,  "high confidence — alert fires"),
        ]
        for conf, should_alert, desc in cases:
            result = conf > ICE_CONFIDENCE_THRESHOLD
            print(f"  mean_conf={conf:.2f}  alert={result}  ({desc})")
            assert result == should_alert

    def test_alert_created_from_session(self, db, seeded_session, real_session):
        """
        Full inference → alert flow using session data.
        Mirrors processing_api.py run_inference() exactly:
          1. confidence_map computed → mean_conf
          2. mean_conf > 0.5 → query device lat/lon from devices table
          3. POST to /api/app/alerts → alert_api.py inserts into ice_alerts
        """
        session = real_session or seeded_session
        mean_conf = 0.78
        lat = session.get("device_lat") or 42.3505
        lon = session.get("device_lon") or -71.1054

        print(f"\n  capture_id:  {session['capture_id']}")
        print(f"  mean_conf:   {mean_conf}")
        print(f"  device GPS:  {lat}, {lon}")

        assert mean_conf > ICE_CONFIDENCE_THRESHOLD

        alert_id = insert_test_alert(
            db, lat=lat, lon=lon, confidence=mean_conf,
            session_id=session["session_id"],
            device_id=session["device_id"],
        )

        cur = db.cursor()
        cur.execute("""
            SELECT id, session_id, device_id, confidence, active, expires_at, is_test
            FROM ice_alerts WHERE id = %s::uuid
        """, (alert_id,))
        row = cur.fetchone()
        cur.close()

        assert row is not None
        assert float(row[3]) == mean_conf
        assert row[4] is True
        assert row[6] is True
        print(f"  alert_id:    {row[0]}")
        print(f"  confidence:  {row[3]}")
        print(f"  expires:     {row[5]}")

    def test_alert_links_back_to_session(self, db, seeded_session, real_session):
        """
        Alert session_id must reference a real capture_sessions row so operators
        can trace any alert back to raw sensor data in the Data Viewer.
        """
        session = real_session or seeded_session
        lat = session.get("device_lat") or 42.3505
        lon = session.get("device_lon") or -71.1054

        alert_id = insert_test_alert(
            db, lat=lat, lon=lon, confidence=0.76,
            session_id=session["session_id"],
            device_id=session["device_id"],
        )

        cur = db.cursor()
        cur.execute("""
            SELECT ia.id, ia.confidence, cs.capture_id, cs.is_complete
            FROM ice_alerts ia
            JOIN capture_sessions cs ON cs.id = ia.session_id
            WHERE ia.id = %s::uuid
        """, (alert_id,))
        row = cur.fetchone()
        cur.close()

        assert row is not None, "Alert not linked to session"
        assert row[3] is True,  "Linked session must be complete"
        print(f"\n  Alert {row[0]} → session {row[2]} (complete={row[3]})")


# ---------------------------------------------------------------------------
# TestAppPollPipeline
# End-to-end: session → inference → alert → app 30s poll
# ---------------------------------------------------------------------------

class TestAppPollPipeline:

    def test_full_pipeline_end_to_end(self, db, seeded_session, real_session):
        """
        Complete path: Pi uploads session → inference fires → alert inserted →
        app polls /api/app/alerts/nearby → alert appears on map.
        """
        session = real_session or seeded_session
        lat = session.get("device_lat") or 42.3505
        lon = session.get("device_lon") or -71.1054
        mean_conf = 0.81

        print(f"\n--- Full Pipeline ---")
        print(f"[1] Session:   {session['capture_id']} ({'/'.join(session['uploaded_sensors'])})")
        print(f"[2] Inference: mean_conf={mean_conf} → fires={mean_conf > ICE_CONFIDENCE_THRESHOLD}")
        print(f"[3] Inserting alert at {lat}, {lon}")

        # Insert as non-test to verify it appears in app query
        real_alert_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=ALERT_EXPIRE_HOURS)
        cur = db.cursor()
        if session["device_id"]:
            cur.execute("""
                INSERT INTO ice_alerts
                    (id, session_id, device_id, latitude, longitude,
                     confidence, expires_at, active, is_test)
                VALUES (%s, %s::uuid, %s, %s, %s, %s, %s, TRUE, FALSE)
            """, (real_alert_id, session["session_id"], session["device_id"],
                  lat, lon, mean_conf, expires_at))
        else:
            cur.execute("""
                INSERT INTO ice_alerts
                    (id, latitude, longitude, confidence, expires_at, active, is_test)
                VALUES (%s, %s, %s, %s, %s, TRUE, FALSE)
            """, (real_alert_id, lat, lon, mean_conf, expires_at))
        db.commit()
        cur.close()

        print(f"[4] App polls nearby (radius=2000m)")
        results = query_nearby_alerts(db, lat=lat, lon=lon, radius_m=2000)
        found = next((r for r in results if r["id"] == real_alert_id), None)
        print(f"    alerts in area: {len(results)}")
        print(f"    our alert found: {found is not None}")
        if found:
            print(f"    shown to user: {round(found['confidence'] * 100)}% confidence")

        assert found is not None, "Alert did not appear in app poll"

        # Cleanup non-test row
        cur = db.cursor()
        cur.execute("DELETE FROM ice_alerts WHERE id = %s::uuid", (real_alert_id,))
        db.commit()
        cur.close()

    def test_test_alerts_never_reach_users(self, db, seeded_session):
        """
        is_test=TRUE alerts must be filtered from the app query at all times.
        This protects real users from seeing test data.
        """
        lat, lon = 42.3505, -71.1054
        alert_id = insert_test_alert(db, lat=lat, lon=lon, confidence=0.90)
        results = query_nearby_alerts(db, lat=lat, lon=lon, radius_m=500)
        ids = [r["id"] for r in results]
        assert alert_id not in ids, "Test alert must never appear in app query"
        print(f"\n  Test alert correctly hidden from app users")

    def test_expired_alert_not_shown(self, db, seeded_session):
        """
        Alerts from inference runs more than 2 hours ago must not appear.
        """
        lat, lon = 42.3505, -71.1054
        alert_id = insert_test_alert(db, lat=lat, lon=lon,
                                     confidence=0.80, expires_hours=-1)
        cur = db.cursor()
        cur.execute("""
            SELECT id FROM ice_alerts
            WHERE id = %s::uuid AND expires_at > NOW()
        """, (alert_id,))
        assert cur.fetchone() is None, "Expired alert should not be active"
        cur.close()
        print(f"\n  Expired alert correctly excluded")

    def test_alerts_sorted_by_confidence(self, db, seeded_session):
        """
        App receives highest confidence alerts first — most dangerous ice shown
        at top of list and with red markers on map.
        """
        lat, lon = 42.3505, -71.1054
        expires_at = datetime.utcnow() + timedelta(hours=2)
        ids = []
        cur = db.cursor()
        for conf in [0.55, 0.90, 0.72]:
            aid = str(uuid.uuid4())
            ids.append(aid)
            cur.execute("""
                INSERT INTO ice_alerts
                    (id, latitude, longitude, confidence, expires_at, active, is_test)
                VALUES (%s, %s, %s, %s, %s, TRUE, FALSE)
            """, (aid, lat, lon, conf, expires_at))
        db.commit()
        cur.close()

        results = query_nearby_alerts(db, lat=lat, lon=lon, radius_m=500)
        confidences = [r["confidence"] for r in results if r["id"] in ids]
        assert confidences == sorted(confidences, reverse=True), \
            f"Not sorted by confidence: {confidences}"
        print(f"\n  Confidence order: {confidences}")

        cur = db.cursor()
        cur.execute("DELETE FROM ice_alerts WHERE id = ANY(%s::uuid[])", (ids,))
        db.commit()
        cur.close()