"""
ws_manager.py
-------------
WebSocket connection manager.
Tracks connected app clients and broadcasts alert events to all of them.
"""

import json
import logging
from fastapi import WebSocket

log = logging.getLogger("app.ws")


class ConnectionManager:
    def __init__(self):
        self.active: set = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        log.info(f"WS client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        log.info(f"WS client disconnected. Total: {len(self.active)}")

    async def broadcast(self, data: dict):
        if not self.active:
            return
        message = json.dumps(data)
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)


manager = ConnectionManager()