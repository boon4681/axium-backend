from fastapi import APIRouter, HTTPException
import axium

router = APIRouter(prefix="/node", tags=["Node"])

@router.get("")
def get_all_nodes():
    return axium.register.get_all_node()


@router.get("/{id}")
def get_node_by_id(id: str):
    node = axium.register.get_node_by_id(id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node \"{id}\" not found")
    return node.get_node_details()
