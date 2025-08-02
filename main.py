from axium.graph import Graph
import core.sys_setup
import axium

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference
from core.logger import setup_logger
from core.server import AxiumServer

setup_logger(log_level="INFO", use_stdout=True)

axium.register.load_nodes()
app = FastAPI()
server = AxiumServer(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )