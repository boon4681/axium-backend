from fastapi import APIRouter, HTTPException
import core.api.v1 as v1

router = APIRouter(prefix="/api")
router.include_router(v1.router)
