from typing import Dict, Union
from fastapi import APIRouter
from pydantic import BaseModel
from axium.core import Axium

router = APIRouter(prefix="/execute", tags=["Execute"])

class AxiumAPIExecute(BaseModel):
    name: str
    parameters: Dict[str, int|str]

@router.post("/")
def execute_function(exec_body: AxiumAPIExecute):
    node = Axium.get_node(exec_body.name)

    return {
        "result": node.execute(exec_body.parameters)
    }