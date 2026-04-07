"""
FrostByte App Backend
---------------------
FastAPI server backed by local Postgres (no Supabase).
Handles auth, alerts, user settings, and Redis detection subscriber.
"""

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import init_pool
from alert_api import alert_router
from auth_api import auth_router
from redis_subscriber import run_subscriber

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app.server")

app = FastAPI(title="FrostByte App Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(alert_router)


@app.on_event("startup")
async def startup():
    log.info("Initializing Postgres connection pool")
    await init_pool()
    log.info("Starting Redis subscriber background task")
    asyncio.create_task(run_subscriber())


@app.get("/health")
def health():
    return {"status": "ok"}