from fastapi import APIRouter, HTTPException
import core.api.v1.node as node

router = APIRouter(prefix="/v1")
router.include_router(node.router)
