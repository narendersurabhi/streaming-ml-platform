from fastapi import APIRouter, Depends

from api.dependencies import get_service

router = APIRouter()


@router.get("/metrics")
def metrics(service=Depends(get_service)) -> dict:
    return service.metrics.snapshot()
