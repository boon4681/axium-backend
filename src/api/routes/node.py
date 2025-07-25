from fastapi import APIRouter, HTTPException

from axium import Axium

router = APIRouter(prefix="/node", tags=["Node"])

@router.get("")
def get_all_nodes():
    return Axium.get_all_node()

@router.get("/{id}")  
def get_node(id: str):
    node = Axium.get_node(id)

    if not node:
        raise HTTPException(status_code=404, detail=f"Node id {id} is not found")

    return node.object
    