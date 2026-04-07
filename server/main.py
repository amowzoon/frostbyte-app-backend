"""
FrostByte App Backend
---------------------
Minimal FastAPI server that serves the mobile app only.
Handles alert publishing to Supabase, user settings, and push tokens.

The Pi data pipeline (MinIO, WebSocket, inference, data viewer) lives
on the team's Linux machine — not here.
"""

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alert_api import alert_router
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

app.include_router(alert_router)


@app.on_event("startup")
async def start_subscriber():
    log.info("Starting Redis subscriber background task")
    asyncio.create_task(run_subscriber())


@app.get("/health")
def health():
    return {"status": "ok"}