from fastapi import FastAPI

from backend.app.api.routers.health_router import router as health_router
from backend.app.api.routers.model_router import router as model_router
from backend.app.api.routers.predict_router import router as predict_router
from backend.app.api.routers.root_router import router as root_router

app = FastAPI()

app.include_router(root_router)
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(model_router)
