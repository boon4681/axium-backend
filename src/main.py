import socketio
from axium.core import Axium
from api.routes import execute, node, project

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference

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

@api.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=api.openapi_url,
        title=api.title,
    )

api.mount("/", sio_app)
