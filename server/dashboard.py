"""
Dashboard WebSocket management and event broadcasting.
Isolated from core application logic for easy maintenance.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set, Dict, Any
from datetime import datetime
import logging
import asyncio
import json
import traceback as tb
import inspect

log = logging.getLogger(__name__)

# Global sets to track connected clients
# Maps WebSocket -> set of subscribed device IDs (empty set = receive all)
dashboard_connections: Dict[WebSocket, set] = {}
log_connections: Set[WebSocket] = set()

# Store checkpoint timing data per capture (capped to prevent unbounded growth)
checkpoint_times: Dict[str, Dict[str, float]] = {}
_CHECKPOINT_MAX = 200

# Grace period for sensor shutdown on dashboard disconnect
_DISCONNECT_GRACE_SECONDS = 8.0
_shutdown_timer: asyncio.TimerHandle = None
_shutdown_callback = None  # set by set_shutdown_callback()


def set_shutdown_callback(callback):
    """Register a coroutine to call when all dashboards disconnect.

    The callback receives no arguments and should send shutdown commands
    to connected devices.  Called from main.py during startup::

        set_shutdown_callback(shutdown_all_sensors)
    """
    global _shutdown_callback
    _shutdown_callback = callback
    log.info("Dashboard shutdown callback registered")


def _cancel_shutdown_timer():
    global _shutdown_timer
    if _shutdown_timer is not None:
        _shutdown_timer.cancel()
        _shutdown_timer = None
        log.debug("Shutdown grace timer cancelled (dashboard reconnected)")


def _start_shutdown_timer():
    """Start a grace-period timer.  If no dashboard reconnects within
    _DISCONNECT_GRACE_SECONDS, fire the shutdown callback."""
    global _shutdown_timer

    _cancel_shutdown_timer()

    if _shutdown_callback is None:
        return

    loop = asyncio.get_event_loop()

    def _fire():
        global _shutdown_timer
        _shutdown_timer = None
        log.info("No dashboard reconnected within %.0fs — shutting down sensors",
                 _DISCONNECT_GRACE_SECONDS)
        asyncio.ensure_future(_shutdown_callback())

    _shutdown_timer = loop.call_later(_DISCONNECT_GRACE_SECONDS, _fire)
    log.info("Shutdown grace timer started (%.0fs)", _DISCONNECT_GRACE_SECONDS)


async def dashboard_websocket_handler(websocket: WebSocket):
    """
    WebSocket endpoint handler for dashboard connections.

    Clients may send JSON messages to control which device events they receive:
        {"action": "subscribe", "device_id": "pi-001"}
        {"action": "unsubscribe", "device_id": "pi-001"}

    A client with no subscriptions receives events for ALL devices (backwards
    compatible with existing behavior).
    """
    await websocket.accept()
    dashboard_connections[websocket] = set()
    _cancel_shutdown_timer()  # dashboard connected — cancel any pending shutdown
    log.info(f"Dashboard client connected. Total connections: {len(dashboard_connections)}")

    # Send initial connection confirmation
    await websocket.send_json({
        "type": "connected",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Dashboard connected successfully"
    })

    try:
        # Keep connection alive and handle subscription messages
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            action = data.get("action")
            device_id = data.get("device_id")

            if action == "subscribe" and device_id:
                dashboard_connections[websocket].add(device_id)
                log.debug(f"Dashboard subscribed to {device_id}")
            elif action == "unsubscribe" and device_id:
                dashboard_connections[websocket].discard(device_id)
                log.debug(f"Dashboard unsubscribed from {device_id}")

    except WebSocketDisconnect:
        log.info("Dashboard client disconnected")
    finally:
        dashboard_connections.pop(websocket, None)
        remaining = len(dashboard_connections)
        log.info(f"Dashboard client removed. Remaining connections: {remaining}")
        if remaining == 0:
            _start_shutdown_timer()


async def logs_websocket_handler(websocket: WebSocket):
    """
    WebSocket endpoint handler for log streaming connections.
    """
    await websocket.accept()
    log_connections.add(websocket)
    log.info(f"Log stream client connected. Total connections: {len(log_connections)}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        log.info("Log stream client disconnected")
    finally:
        log_connections.discard(websocket)


def get_caller_info():
    """Get file, function, and line number of the caller."""
    try:
        frame = inspect.currentframe().f_back.f_back  # Go up 2 frames
        return {
            "file": frame.f_code.co_filename.split("/")[-1].split("\\")[-1],
            "function": frame.f_code.co_name,
            "line": frame.f_lineno
        }
    except Exception:
        return {"file": "unknown", "function": "unknown", "line": 0}


async def broadcast_dashboard_event(event: Dict[str, Any]):
    """
    Broadcast an event to connected dashboard clients.

    If the event contains a ``device_id`` field, it is only sent to clients
    that have subscribed to that device (or clients with no subscriptions,
    which receive everything for backwards compatibility).
    """
    if not dashboard_connections:
        return  # No dashboards connected, skip broadcasting

    # Add caller info
    caller_info = get_caller_info()
    event["file"] = caller_info["file"]
    event["function"] = caller_info["function"]
    event["line"] = caller_info["line"]

    # Track timing for duration calculation
    capture_id = event.get("capture_id")
    stage = event.get("stage")

    if capture_id and stage:
        # Evict oldest entries if over cap
        if len(checkpoint_times) > _CHECKPOINT_MAX:
            excess = len(checkpoint_times) - _CHECKPOINT_MAX
            for key in list(checkpoint_times)[:excess]:
                del checkpoint_times[key]

        if capture_id not in checkpoint_times:
            checkpoint_times[capture_id] = {}

        current_time = datetime.utcnow().timestamp()
        checkpoint_times[capture_id][stage] = current_time

        # Calculate duration from previous stage
        stages_order = ["command_sent", "sensor_capture_complete", "upload_received", "s3_stored", "db_stored"]
        try:
            current_idx = stages_order.index(stage)
            if current_idx > 0:
                prev_stage = stages_order[current_idx - 1]
                if prev_stage in checkpoint_times[capture_id]:
                    duration_ms = (current_time - checkpoint_times[capture_id][prev_stage]) * 1000
                    event["duration_ms"] = round(duration_ms, 2)
        except ValueError:
            pass

    # Determine which clients should receive this event
    event_device_id = event.get("device_id")
    dead_connections = set()

    for ws, subscribed_devices in dashboard_connections.items():
        # Send to this client if:
        #   - client has no subscriptions (wildcard / legacy mode), OR
        #   - event has no device_id (global event), OR
        #   - client is subscribed to this device
        if subscribed_devices and event_device_id and event_device_id not in subscribed_devices:
            continue
        try:
            await ws.send_json(event)
        except Exception as e:
            log.warning(f"Failed to send event to dashboard client: {e}")
            dead_connections.add(ws)

    # Clean up dead connections
    for ws in dead_connections:
        dashboard_connections.pop(ws, None)


async def broadcast_log_entry(log_entry: Dict[str, Any]):
    """Broadcast log entries to all connected log stream clients."""
    if not log_connections:
        return

    dead_connections = set()
    for ws in log_connections:
        try:
            await ws.send_json(log_entry)
        except Exception as e:
            dead_connections.add(ws)

    if dead_connections:
        log_connections.difference_update(dead_connections)


class WebSocketLogHandler(logging.Handler):
    """Custom logging handler that broadcasts to WebSocket clients.

    Thread-safe: uses ``loop.call_soon_threadsafe`` so that log calls from
    worker threads (``asyncio.to_thread``, ``ThreadPoolExecutor``) schedule
    the broadcast on the main event loop instead of crashing with
    "no running event loop".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loop = None

    def set_loop(self, loop):
        """Capture the running event loop (call from an async context)."""
        self._loop = loop

    def emit(self, record):
        if not log_connections:
            return  # No log clients, skip entirely
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "module": record.name,
                "message": self.format(record),
                "file": record.pathname.split("/")[-1].split("\\")[-1],
                "line": record.lineno,
                "function": record.funcName
            }

            # Add exception info if present
            if record.exc_info:
                log_entry["traceback"] = "".join(tb.format_exception(*record.exc_info))

            loop = self._loop
            if loop is None:
                return

            if loop.is_running():
                # Safe from any thread: schedules coroutine on the event loop
                loop.call_soon_threadsafe(loop.create_task, broadcast_log_entry(log_entry))
        except Exception:
            self.handleError(record)


# Module-level reference so setup_websocket_logging can configure it
# and startup can set the loop.
_ws_log_handler: WebSocketLogHandler = None


def setup_websocket_logging():
    """Add WebSocket handler to root logger.

    Call ``set_websocket_log_loop(loop)`` after startup to enable
    thread-safe broadcasting.
    """
    global _ws_log_handler
    _ws_log_handler = WebSocketLogHandler()
    _ws_log_handler.setLevel(logging.DEBUG)

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(_ws_log_handler)

    log.info("WebSocket log streaming enabled")


def set_websocket_log_loop(loop):
    """Provide the running event loop to the WebSocket log handler.

    Must be called from an async context during startup, e.g.::

        @app.on_event("startup")
        async def startup():
            set_websocket_log_loop(asyncio.get_running_loop())
    """
    if _ws_log_handler is not None:
        _ws_log_handler.set_loop(loop)
        log.info("WebSocket log handler bound to event loop")
