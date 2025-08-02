import asyncio
import json
import uuid
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect


class WebIO:
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.clients: Dict[str, WebSocket] = {}
        self.rooms: Dict[str, Set[str]] = defaultdict(set)

    def on(self, event: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            self.handlers[event].append(func)
            return func
        return decorator

    def _generate_sid(self) -> str:
        return str(uuid.uuid4())

    async def _send_message(self, websocket: WebSocket, payload: Dict[str, Any]):
        await websocket.send_text(json.dumps(payload))

    async def emit(self, event: str, data: Any, room: str = None, to: str = None):
        payload = {"event": event, "data": data}

        if to and to in self.clients:
            await self._send_message(self.clients[to], payload)
        elif room and room in self.rooms:
            tasks = [
                self._send_message(self.clients[client_sid], payload)
                for client_sid in self.rooms[room] if client_sid in self.clients
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:  # Broadcast to all
            tasks = [self._send_message(ws, payload)
                     for ws in self.clients.values()]
            await asyncio.gather(*tasks, return_exceptions=True)

    def enter_room(self, sid: str, room: str):
        self.rooms[room].add(sid)
        print(f"SID {sid} entered room '{room}'")

    def leave_room(self, sid: str, room: str):
        if room in self.rooms:
            self.rooms[room].discard(sid)
            if not self.rooms[room]:  # Clean up empty rooms
                del self.rooms[room]
        print(f"SID {sid} left room '{room}'")

    async def disconnect(self, sid: str, reason: str = "Server initiated disconnect"):
        if sid in self.clients:
            websocket: WebSocket = self.clients[sid]
            # await self._cleanup_client(sid)
            await websocket.close(code=1000, reason=reason)
            await self._cleanup_client(sid)
            # print(websocket.application_state, websocket.client_state)
            print(f"Server initiated disconnect for SID {sid}")

    async def _cleanup_client(self, sid: str):
        if sid in self.clients:
            del self.clients[sid]
        for room in list(self.rooms.keys()):
            self.leave_room(sid, room)
        if 'disconnect' in self.handlers:
            for handler in self.handlers['disconnect']:
                await handler(sid)

        print(
            f"Client disconnected: {sid}. Total clients: {len(self.clients)}")

    async def handle_websocket(self, websocket: WebSocket):
        sid = self._generate_sid()
        self.clients[sid] = websocket
        if 'connect' in self.handlers:
            meta = {}
            print(websocket.scope['headers'])
            for name, value in websocket.scope['headers']:
                if name == b"meta":
                    meta = json.loads(value)
            for handler in self.handlers['connect']:
                await handler(sid, websocket.scope, meta)
        if websocket.application_state.name == "DISCONNECTED":
            return
        await websocket.accept()
        print(f"Client connected: {sid}. Total clients: {len(self.clients)}")
        try:
            while True:
                try:
                    if websocket.application_state.name == "DISCONNECTED":
                        return
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                    await websocket.send_text("")
                    payload = json.loads(message)
                    event = payload.get("event")
                    data = payload.get("data")
                    ack_id = payload.get("ackId")

                    if event in self.handlers:
                        ack_result = None
                        for handler in self.handlers[event]:
                            result = await handler(sid, data)
                            if ack_result is None and result is not None:
                                ack_result = result

                        if ack_id and ack_result is not None:
                            await self._send_message(websocket, {
                                "event": "ack",
                                "ackId": ack_id,
                                "data": ack_result
                            })
                except json.JSONDecodeError:
                    print(
                        f"! SID {sid}: Could not decode JSON from message: {message}")
                except asyncio.TimeoutError:
                    pass
                except WebSocketDisconnect as e:
                    await self._cleanup_client(sid)
                except RuntimeError as e:
                    break
        except WebSocketDisconnect as e:
            pass
            await self._cleanup_client(sid)
