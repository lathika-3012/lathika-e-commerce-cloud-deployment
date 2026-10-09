"""FastAPI backend: Profit & Risk-Aware E-Commerce AI Gateway."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import configure_logging, ensure_directories
from pipeline import GatewayPipeline
from schemas import ChatRequest, ChatResponse

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create directories and load the pipeline once at startup (no training here)."""
    ensure_directories()
    app.state.pipeline = GatewayPipeline()
    yield


app = FastAPI(title="Profit & Risk-Aware E-Commerce AI Gateway", version="1.0.0", lifespan=lifespan)


def _get_pipeline(request: Request) -> GatewayPipeline:
    """Return the shared pipeline (created lazily if lifespan did not run)."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        pipeline = GatewayPipeline()
        request.app.state.pipeline = pipeline
    return pipeline


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Clean 422 body without internals."""
    details = [{"field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"),
                "message": str(err.get("msg", "Invalid value"))} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_error",
                                                  "message": "Invalid request payload.",
                                                  "details": details})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Clean 500 body; the stack trace is only written to the server log."""
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"error": "internal_server_error",
                                                  "message": "An unexpected error occurred. Please try again later."})


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "healthy"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Route one customer query through the seven-layer pipeline."""
    return _get_pipeline(request).handle(payload)


@app.get("/api/v1/telemetry/summary")
def telemetry_summary(request: Request) -> Dict[str, object]:
    """Aggregate routing statistics since the server started."""
    return _get_pipeline(request).telemetry.summary()


@app.get("/api/v1/telemetry/recent")
def telemetry_recent(request: Request, n: int = 20) -> Dict[str, object]:
    """Most recent routed requests (no query text)."""
    return {"records": _get_pipeline(request).telemetry.recent(min(max(n, 1), 200))}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
