from pydantic import BaseModel, ConfigDict, Field, Json
from annotated_types import Len
from typing_extensions import TypedDict, NotRequired, Literal, Optional, Union, Any, List, Annotated, Dict


class Position(BaseModel):
    x: float
    y: float


class AxiumNodeData(BaseModel):
    id: str
    _ref: str
    parameters: dict[str, Any]
    position: Position


class AxiumVertexData(BaseModel):
    id: str
    node_id: str


class AxiumGraphData(BaseModel):
    edges: List[
        Annotated[List[AxiumVertexData], Len(min_length=2, max_length=2)]
    ]
    nodes: List[AxiumNodeData]


class AxiumProjectMetaData(BaseModel):
    current_tab: Union[None, str] = Field(default=None)
    open_tabs: List[str] = Field(default=[])
    expanded_folders: List[str] = Field(default=[])


class AxiumErrorJSONResponse(BaseModel):
    status: str
    type: str
    description: str
    details: str
