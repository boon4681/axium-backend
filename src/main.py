import socketio
from axium.core import Axium
from api.routes import execute, node, project

from fastapi import FastAPI

Axium.setup()

api = FastAPI(root_path="/api/v1")
api.include_router(node.router)
api.include_router(project.router)
api.include_router(execute.router)

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
sio_app = socketio.ASGIApp(sio) 

api.mount("/", sio_app)