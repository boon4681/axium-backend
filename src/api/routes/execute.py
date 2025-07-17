from fastapi import APIRouter, HTTPException

from api.model.execute import ExecuteBody
from axium.graph import AxiumGraph

router = APIRouter(prefix="/execute", tags=["Execute"])

@router.post("")
def run_graph(body: ExecuteBody):
    try:
        nodes = body.model_dump().get("nodes")

        AxiumGraph.load_from_object(nodes)
        result = AxiumGraph.run()

        return [
            {
                "id": node.id,
                **node.last_result
            }
            for node in result
        ]
        
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))