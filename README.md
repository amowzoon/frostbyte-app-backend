# frostbyte-app-backend

Backend for the frostbyte mobile app. Handles push notifications for ice detection alerts.

## Architecture

- Reads ice mask detections from shared Redis (published by frostbyte-backend)
- Maintains active alerts table in its own private Redis
- Compares consecutive detections to decide when to push alerts
- Serves the mobile app at (future) `/api/app/*`

See the detection contract: `frostbyte-backend/docs/detection_contract.md`

## Running modes

### Combined with frostbyte-backend (production)

From the `frostbyte-backend` directory:
```bash
docker compose up
```
Uses `include:` to pull in this compose file. Shared Redis comes from frostbyte-backend. Private Redis runs in this stack.

### Standalone (solo development)

```bash
cp .env.template .env  # edit REDIS_PASSWORD
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```
Brings up a local shared Redis + private Redis alongside mobile-backend.

## Redis instances

| Instance | Purpose | Provided by |
|---|---|---|
| `redis` (shared) | Detection keys from frostbyte-backend | frostbyte-backend (prod) or dev override (standalone) |
| `mobile-redis` (private) | Active alerts table, internal state | This compose file |

## Subscribing to detections

Subscribe to keyspace notifications on the shared Redis:

```python
client = redis.Redis.from_url(os.environ["SHARED_REDIS_URL"], decode_responses=False)
pubsub = client.pubsub()
pubsub.psubscribe("__keyevent@0__:lpush")

for message in pubsub.listen():
    if message["type"] != "pmessage":
        continue
    key = message["data"]
    if not key.startswith(b"detection:"):
        continue
    entries = client.lrange(key, 0, 1)
    # entries[0] = newest, entries[1] = previous
    # Compare masks, update alerts table in private Redis, push if needed
```

Full schema in `frostbyte-backend/docs/detection_contract.md`.

## Status / next steps

The `mobile-backend` service is a **placeholder**. Ari implements:

- [ ] FastAPI app with `/api/app/*` routes
- [ ] Own Postgres service in `docker-compose.yml`
- [ ] Redis subscriber loop on keyspace notifications
- [ ] Detection comparison logic (two consecutive masks → alert decision)
- [ ] Push notification integration
- [ ] Own Dockerfile (replace `image: python:3.12-slim` with `build: .`)
