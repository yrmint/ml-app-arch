from fastapi import FastAPI

from app.routers import health_router, root_router

app = FastAPI()

app.include_router(health_router.router)
app.include_router(root_router.router)
