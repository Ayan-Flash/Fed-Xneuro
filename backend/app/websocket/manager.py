"""
WebSocket connection manager for real-time simulation streaming.
"""

from typing import Dict, List, Any
from fastapi import WebSocket


class ConnectionManager:
    """Manages active WebSocket connections per simulation run."""

    def __init__(self) -> None:
        # Maps simulation_id -> list of active WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, simulation_id: str) -> None:
        await websocket.accept()
        if simulation_id not in self.active_connections:
            self.active_connections[simulation_id] = []
        self.active_connections[simulation_id].append(websocket)

    def disconnect(self, websocket: WebSocket, simulation_id: str) -> None:
        if simulation_id in self.active_connections:
            if websocket in self.active_connections[simulation_id]:
                self.active_connections[simulation_id].remove(websocket)
            if not self.active_connections[simulation_id]:
                del self.active_connections[simulation_id]

    async def broadcast(self, simulation_id: str, message: Dict[str, Any]) -> None:
        if simulation_id in self.active_connections:
            for connection in self.active_connections[simulation_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


ws_manager = ConnectionManager()
