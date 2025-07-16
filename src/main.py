import socketio
from axium.core import Axium
from api.routes import node, project

from fastapi import FastAPI

Axium.setup()

api = FastAPI(root_path="/v1/api")
api.include_router(node.router)
api.include_router(project.router)

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
sio_app = socketio.ASGIApp(sio) 

api.mount("/", sio_app)