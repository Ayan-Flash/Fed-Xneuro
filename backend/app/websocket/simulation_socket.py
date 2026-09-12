"""
WebSocket route for real-time simulation updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.app.websocket.manager import ws_manager

ws_router = APIRouter()


@ws_router.websocket("/ws/simulations/{simulation_id}")
async def simulation_websocket_endpoint(websocket: WebSocket, simulation_id: str):
    await ws_manager.connect(websocket, simulation_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo ping / status request
            await websocket.send_json({"event": "ack", "message": f"Listening to {simulation_id}"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, simulation_id)
