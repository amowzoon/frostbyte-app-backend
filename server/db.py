"""
db.py
-----
Postgres connection pool and table initialization.
Tables are created on startup if they don't exist.
"""

import os
import asyncpg

_pool: asyncpg.Pool | None = None


async def init_pool():
    global _pool
    _pool = await asyncpg.create_pool(os.environ["POSTGRES_DSN"], min_size=2, max_size=10)
    await _create_tables()


async def get_pool() -> asyncpg.Pool:
    return _pool


async def _create_tables():
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email         TEXT UNIQUE NOT NULL,
                password      TEXT NOT NULL,
                created_at    TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id          UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                alert_radius_m   INT DEFAULT 500,
                notify_ice       BOOL DEFAULT true,
                notify_bluetooth BOOL DEFAULT true,
                notify_route     BOOL DEFAULT true,
                conf_min         FLOAT DEFAULT 0.0,
                push_token       TEXT,
                updated_at       TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS ice_alerts (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                latitude    DOUBLE PRECISION NOT NULL,
                longitude   DOUBLE PRECISION NOT NULL,
                confidence  DOUBLE PRECISION NOT NULL,
                alert_type  TEXT DEFAULT 'heuristic',
                device_id   TEXT,
                active      BOOL DEFAULT true,
                is_test     BOOL DEFAULT false,
                created_at  TIMESTAMPTZ DEFAULT now(),
                expires_at  TIMESTAMPTZ NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_ice_alerts_active
                ON ice_alerts (active, is_test, expires_at);
        """)

        # Migrate existing tables if needed
        await conn.execute("""
            ALTER TABLE user_preferences
                ADD COLUMN IF NOT EXISTS conf_min FLOAT DEFAULT 0.0;
        """)