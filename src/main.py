import socketio
from axium.core import Axium
from api.routes import node, execute, project

from fastapi import FastAPI

Axium.setup()
Axium.get_node(id="300000")

api = FastAPI(root_path="/v1/api")
api.include_router(node.router)
api.include_router(execute.router)
api.include_router(project.router)

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
sio_app = socketio.ASGIApp(sio) 

api.mount("/", sio_app)