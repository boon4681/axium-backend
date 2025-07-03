from axium.core import Axium
from api.routes import node, execute, project

from fastapi import FastAPI

Axium.setup()

api = FastAPI(root_path="/v1/api")
api.include_router(node.router)
api.include_router(execute.router)
api.include_router(project.router)