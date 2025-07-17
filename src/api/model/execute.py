from typing import Dict, List, Optional
from pydantic import BaseModel

class Node(BaseModel):
    id: int
    template_id: str
    required:   List['Edge']
    parameters: Optional[Dict[str, str | int]] = None
    property:   Optional[Dict[str, str | int]] = None

class Edge(BaseModel):
    id: int
    src_output: str
    dst_param: str

class ExecuteBody(BaseModel):
    nodes: List[Node]