import socketio
from axium.core import Axium
from api.routes import execute, node, project

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

Axium.setup()

api = FastAPI(root_path="/api/v1")
api.include_router(node.router)
api.include_router(project.router)
api.include_router(execute.router)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
sio_app = socketio.ASGIApp(sio) 

api.mount("/", sio_app)