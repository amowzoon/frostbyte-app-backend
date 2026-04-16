# Local Dev Testing Instructions

These steps let you test your backend code without frostbyte-backend running.
The test script simulates what the Pi + frostbyte-backend would publish to shared Redis.

## Prerequisites

**WSL is strongly recommended on Windows.** Open a WSL terminal in the mobile backend directory and work from there.

```bash
cp .env.template .env        # set REDIS_PASSWORD to anything (e.g. "devpassword")
sudo apt install -y python3.12-venv
python3 -m venv .venv
.venv/bin/pip install redis python-dotenv
```

## 1. Start the dev stack

```bash
sudo docker compose --profile mobile -f docker-compose.yml -f docker-compose.dev.yml up --build -d
```

This brings up:
- `mobile-backend` — your app container
- `mobile-redis` — your private Redis
- `redis` (dev) — local stand-in/dummy for frostbyte-backend's shared Redis, exposed on localhost:6380

## 2. Load env vars

```bash
source .env
```

## 3. Publish a test detection (mock backend sending a detection to redis)

```bash
.venv/bin/python3 scripts/publish_test_detection.py
```

Expected output:
```
[1/1] key=detection:test-pi  list_len=1  mask=168B  ts=2026-...
```

Optional args:
```bash
.venv/bin/python3 scripts/publish_test_detection.py --device-id my-pi --geotag "My Roof" --lat 42.35 --lon -71.06
```

## 4. Verify only 2 entries are being kept in shared Redis per device

```bash
.venv/bin/python3 scripts/publish_test_detection.py --count 3
```

Expected output:
```
[1/3] key=detection:test-pi  list_len=1  ...
[2/3] key=detection:test-pi  list_len=2  ...
[3/3] key=detection:test-pi  list_len=2  ...

LTRIM check: list has 2 entries (expect 2)
```

## 5. Inspect stored, shared Redis, entries

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec redis \
    redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning lrange detection:test-pi 0 -1
```

Returns at most 2 entries. Entry 0 = newest, entry 1 = previous.
Output will look garbled after the `---MASK---` separator — that's the raw PNG bytes, expected.

## Teardown

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```
