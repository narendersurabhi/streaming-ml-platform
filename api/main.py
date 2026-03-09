from fastapi import FastAPI

from api.routes.health import router as health_router
from api.routes.metrics import router as metrics_router
from api.routes.recommend import router as recommend_router

app = FastAPI(title="Streaming ML Platform API", version="0.1.0")
app.include_router(health_router)
app.include_router(recommend_router)
app.include_router(metrics_router)
