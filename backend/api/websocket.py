"""
websocket.py — WebSocket handler for real-time MDP graph and event updates.

Provides a :class:`WebSocketManager` that tracks connected clients and
broadcasts typed JSON messages (turn results, graph updates, NPC actions,
random events, game-over, and full state syncs).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import logger

ws_router = APIRouter()


# ─── WebSocket Manager ───────────────────────────────────────────────────────


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting.

    All connected clients receive the same broadcast messages.  Broken
    or disconnected sockets are removed silently.
    """

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The incoming WebSocket instance.
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            "WebSocket connected — %d active connection(s)",
            len(self.active_connections),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket from the active pool.

        Args:
            websocket: The socket to remove.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            "WebSocket disconnected — %d active connection(s)",
            len(self.active_connections),
        )

    async def broadcast(self, message: dict) -> None:
        """Send *message* as JSON to every connected client.

        Broken connections are removed automatically so subsequent
        broadcasts don't retry dead sockets.

        Args:
            message: Dict payload; must be JSON-serialisable.
        """
        dead: list[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_personal(self, websocket: WebSocket, message: dict) -> None:
        """Send *message* to a single client.

        Args:
            websocket: Target socket.
            message: Dict payload.
        """
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)

    # ── Convenience typed senders ─────────────────────────────────────────

    async def broadcast_turn_result(self, data: dict) -> None:
        """Broadcast a turn result to all clients."""
        await self.broadcast({"type": "turn_result", "data": data})

    async def broadcast_graph_update(self, data: dict) -> None:
        """Broadcast an MDP graph update."""
        await self.broadcast({"type": "graph_update", "data": data})

    async def broadcast_npc_action(self, data: dict) -> None:
        """Broadcast an NPC action visible to the player."""
        await self.broadcast({"type": "npc_action", "data": data})

    async def broadcast_event(self, data: dict) -> None:
        """Broadcast a random event notification."""
        await self.broadcast({"type": "event", "data": data})

    async def broadcast_game_over(self, data: dict) -> None:
        """Broadcast game-over information."""
        await self.broadcast({"type": "game_over", "data": data})

    async def send_state_sync(self, websocket: WebSocket, data: dict) -> None:
        """Send a full state sync to a single reconnecting client."""
        await self.send_personal(websocket, {"type": "state_sync", "data": data})


# ─── Module-level singleton ──────────────────────────────────────────────────

ws_manager = WebSocketManager()


# ─── WebSocket endpoint ──────────────────────────────────────────────────────


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time game updates.

    On connect the client receives a ``state_sync`` message with the
    current game state (if a session is active).  The server then keeps
    the socket open, relaying broadcasts until the client disconnects.
    """
    await ws_manager.connect(websocket)

    # Send initial state if a game session exists
    try:
        from backend.api.routes import get_engine  # deferred to avoid circular

        engine = get_engine(raise_on_missing=False)
        if engine is not None:
            await ws_manager.send_state_sync(
                websocket,
                engine.get_full_state(),
            )
    except Exception:
        pass  # No active session — that's fine

    try:
        while True:
            # Keep connection alive; handle optional client messages
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws_manager.send_personal(
                    websocket,
                    {"type": "error", "data": {"message": "Invalid JSON"}},
                )
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await ws_manager.send_personal(
                    websocket, {"type": "pong", "data": {}}
                )

            elif msg_type == "request_state":
                try:
                    engine = get_engine(raise_on_missing=False)
                    if engine is not None:
                        await ws_manager.send_state_sync(
                            websocket,
                            engine.get_full_state(),
                        )
                    else:
                        await ws_manager.send_personal(
                            websocket,
                            {
                                "type": "error",
                                "data": {"message": "No active game session"},
                            },
                        )
                except Exception:
                    pass

            else:
                await ws_manager.send_personal(
                    websocket,
                    {
                        "type": "error",
                        "data": {"message": f"Unknown message type: {msg_type}"},
                    },
                )

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
