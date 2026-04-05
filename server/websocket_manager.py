"""
WebSocket connection manager for device communication
"""
from fastapi import WebSocket
from typing import Dict, Optional
import asyncio
import logging
import json
import uuid

log = logging.getLogger("app.websocket")


class DeviceManager:
    """Manages WebSocket connections to devices"""

    def __init__(self):
        # device_id -> WebSocket connection
        self.connected_devices: Dict[str, WebSocket] = {}
        # query_id -> Future, for query/response pattern
        self.pending_queries: Dict[str, asyncio.Future] = {}
    
    async def connect(self, device_id: str, websocket: WebSocket) -> None:
        """Accept new device connection"""
        await websocket.accept()
        
        # Close existing connection if device reconnects
        if device_id in self.connected_devices:
            try:
                await self.connected_devices[device_id].close()
            except:
                pass
        
        self.connected_devices[device_id] = websocket
        log.info(f"Device connected: {device_id}")
        
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "device_id": device_id
        })
    
    def disconnect(self, device_id: str) -> None:
        """Handle device disconnect"""
        if device_id in self.connected_devices:
            del self.connected_devices[device_id]
            log.info(f"Device disconnected: {device_id}")
    
    async def send_command(self, device_id: str, command: dict) -> bool:
        """Send command to specific device"""
        if device_id not in self.connected_devices:
            log.warning(f"Device not connected: {device_id}")
            return False
        
        websocket = self.connected_devices[device_id]
        try:
            await websocket.send_json(command)
            log.info(f"Sent command to {device_id}: {command.get('action')}")
            return True
        except Exception as e:
            log.error(f"Failed to send to {device_id}: {e}")
            self.disconnect(device_id)
            return False
    
    async def send_query(self, device_id: str, method: str, path: str, body=None, timeout: float = 15.0) -> dict:
        """Send a query to a device and wait for the response."""
        if device_id not in self.connected_devices:
            raise Exception(f"Device {device_id} not connected")

        query_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_queries[query_id] = future

        try:
            await self.connected_devices[device_id].send_json({
                "type": "query",
                "query_id": query_id,
                "method": method,
                "path": path,
                "body": body
            })
        except Exception as e:
            self.pending_queries.pop(query_id, None)
            raise Exception(f"Failed to send query to device: {e}")

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.pending_queries.pop(query_id, None)
            raise asyncio.TimeoutError(f"Device did not respond within {timeout}s")

    def resolve_query(self, query_id: str, data: dict) -> None:
        """Resolve a pending query future with the device's response."""
        future = self.pending_queries.pop(query_id, None)
        if future and not future.done():
            future.set_result(data)
        else:
            log.warning(f"Received query_response for unknown or expired query_id: {query_id}")

    def is_connected(self, device_id: str) -> bool:
        return device_id in self.connected_devices
    
    def get_connected_devices(self) -> list:
        return list(self.connected_devices.keys())


# Global instance
device_manager = DeviceManager()
