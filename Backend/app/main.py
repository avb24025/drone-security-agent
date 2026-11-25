"""
app/main.py
───────────
FastAPI application entry point.

Startup sequence
────────────────
1. Configure logging.
2. Initialise SQLite (create tables if absent).
3. Mount all API routers.
4. Add CORS middleware for the React frontend.
5. Add a health-check endpoint.

Run with:
    cd Backend
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import configure_logging, get_settings
from app.database import init_db

# ── Logging must be configured before any module-level loggers fire ──────────
configure_logging()
logger = logging.getLogger(__name__)

# ── Application factory ───────────────────────────────────────────────────────
app = FastAPI(
    title="Drone Security Analyst Agent",
    description=(
        "AI-powered property security monitoring system. "
        "Processes simulated drone frames with Gemini, extracts events, "
        "generates automatic security alerts, and answers questions through "
        "an AI Security Analyst chatbot."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,   # e.g. http://localhost:5173
        "http://localhost:3000",    # CRA default, kept as fallback
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database initialisation ───────────────────────────────────────────────────
@app.on_event("startup")
def on_startup() -> None:
    logger.info("Starting Drone Security Analyst Agent API…")
    init_db()
    logger.info("Database ready.")


# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.alerts import router as alerts_router          # noqa: E402
from app.api.chat import router as chat_router              # noqa: E402
from app.api.dashboard import router as dashboard_router    # noqa: E402
from app.api.events import router as events_router          # noqa: E402
from app.api.frames import router as frames_router          # noqa: E402

app.include_router(frames_router)
app.include_router(events_router)
app.include_router(alerts_router)
app.include_router(chat_router)
app.include_router(dashboard_router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health_check() -> dict:
    """Liveness probe used by the React frontend to verify the API is up."""
    return {"status": "ok", "service": "drone-security-agent"}


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "message": "Drone Security Analyst Agent API",
        "docs": "/docs",
        "health": "/health",
    }
