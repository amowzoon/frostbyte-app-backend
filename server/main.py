"""
FrostByte App Backend
---------------------
Minimal FastAPI server that serves the mobile app only.
Handles alert publishing to Supabase, user settings, and push tokens.

The Pi data pipeline (MinIO, WebSocket, inference, data viewer) lives
on the team's Linux machine — not here.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from alert_api import alert_router

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


@app.get("/health")
def health():
    return {"status": "ok"}
