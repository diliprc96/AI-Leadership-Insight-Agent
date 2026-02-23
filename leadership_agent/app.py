"""
app.py — FastAPI application for the Leadership Agent.

Endpoints:
    POST /query        — Run agent and return structured response
    GET  /health       — Health check
    GET  /static/...   — Serve generated plot images

Run with:
    uvicorn leadership_agent.app:app --reload --port 8000
"""

import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path regardless of launch directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Bootstrap logging before service imports
from leadership_agent.logging_config import setup_logging
setup_logging("INFO")

from leadership_agent.services.agent_service import AgentService
from leadership_agent.config import STATIC_DIR

logger = logging.getLogger(__name__)

# ─── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Leadership Insight Agent",
    description="RAG + Tool-Augmented agent for Microsoft 10-K analysis (FY2023–FY2025)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (generated plots)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Lazy service singleton
_service: AgentService | None = None


def _get_service() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService()
    return _service


# ─── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    tools_used: list[str]
    sources: list[dict]
    image_path: str | None = None
    metrics: dict = {}
    error: str | None = None


# ─── Timing Middleware ─────────────────────────────────────────────────────────

@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    t0 = time.perf_counter()
    logger.info("Request: %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Response: %s %s → %d in %.3fs",
        request.method, request.url.path, response.status_code, elapsed,
    )
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"
    return response


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "AI Leadership Insight Agent"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Run the Leadership Agent for a user query.

    Request body:
        {"query": "What are the key risks in 2024?"}

    Response:
        {
            "answer": "...",
            "tools_used": ["retriever"],
            "sources": [...],
            "image_path": null,
            "metrics": {...},
            "error": null
        }
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    logger.info("POST /query — query=%r", request.query[:100])

    service = _get_service()
    response = service.run(request.query)

    return QueryResponse(
        answer=response.get("answer", ""),
        tools_used=response.get("tools_used", []),
        sources=response.get("sources", []),
        image_path=response.get("image_path"),
        metrics=response.get("metrics", {}),
        error=response.get("error"),
    )


# ─── Startup / Shutdown Events ────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    logger.info("FastAPI startup — Leadership Agent ready.")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("FastAPI shutdown.")
