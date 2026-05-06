import logging
import time

from fastapi import FastAPI, Request

from backend.app.api.routers.health_router import router as health_router
from backend.app.api.routers.model_router import router as model_router
from backend.app.api.routers.predict_router import router as predict_router
from backend.app.api.routers.root_router import router as root_router
from backend.app.core.logging_config import configure_logging


configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI()


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    """
    Logs backend HTTP requests and responses.
    """
    start_time = time.perf_counter()

    logger.info(
        "HTTP request started | method=%s | path=%s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "HTTP request failed | method=%s | path=%s",
            request.method,
            request.url.path,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "HTTP request completed | method=%s | path=%s | "
        "status_code=%s | duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


app.include_router(root_router)
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(model_router)
