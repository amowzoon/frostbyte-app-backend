# Local Dev Testing Instructions

These steps let you test your subscriber code without frostbyte-backend running.
The test script simulates what the Pi + frostbyte-backend would publish.

## Prerequisites

```bash
cp .env.template .env        # set REDIS_PASSWORD to anything (e.g. "devpassword")
pip install redis python-dotenv
```

## 1. Start the dev stack

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

This brings up:
- `mobile-backend` — your app container
- `mobile-redis` — your private Redis
- `redis` (dev) — local stand-in/dummy for frostbyte-backend's shared Redis, exposed on localhost:6380

## 2. Publish a test detection

```bash
python scripts/publish_test_detection.py
```

Expected output:
```
[1/1] key=detection:test-pi  list_len=1  mask=168B  ts=2026-...
```

Optional args:
```bash
python scripts/publish_test_detection.py --device-id my-pi --geotag "My Roof" --lat 42.35 --lon -71.06
```

## 3. Verify keyspace notification fires

Open two terminals.

**Terminal 1 — subscribe:**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec redis \
    redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning psubscribe "__keyevent@0__:lpush"
```

**Terminal 2 — publish:**
```bash
python scripts/publish_test_detection.py
```

Terminal 1 should show:
```
pmessage    __keyevent@0__:lpush    detection:test-pi
```

## 4. Test LTRIM (only 2 entries kept per device)

```bash
python scripts/publish_test_detection.py --count 3
```

Expected output:
```
[1/3] key=detection:test-pi  list_len=1  ...
[2/3] key=detection:test-pi  list_len=2  ...
[3/3] key=detection:test-pi  list_len=2  ...

LTRIM check: list has 2 entries (expect 2)
```

## 5. Read entries back manually

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec redis \
    redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning lrange detection:test-pi 0 -1
```

Entry 0 = newest, entry 1 = previous. Your subscriber reads these to compare masks.

## Teardown

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```
