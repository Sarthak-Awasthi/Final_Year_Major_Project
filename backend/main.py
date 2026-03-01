"""
main.py — FastAPI application entry point for the MVP Research Game.

Assembles routers, middleware, static file serving, and startup / shutdown
hooks.  Run with ``python -m backend.main`` or via uvicorn directly.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import GAME_VERSION, logger

# ─── Paths ────────────────────────────────────────────────────────────────────

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks for the FastAPI application."""
    # ── Startup ───────────────────────────────────────────────────────
    logger.info("MVP Research Game server starting — version %s", GAME_VERSION)

    # Optionally pre-load the spaCy model so the first text parse is fast
    try:
        from backend.player.input_parser import init_nlp

        init_nlp()
        logger.info("spaCy NLP model loaded at startup")
    except Exception as exc:
        logger.warning("spaCy model not loaded at startup: %s", exc)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("MVP Research Game server shutting down")

    # End any active session gracefully
    try:
        from backend.api.routes import session_mgr

        if session_mgr.is_active():
            engine = session_mgr.current_engine
            if engine is not None:
                try:
                    engine.save_game(slot="auto_shutdown")
                    logger.info("Auto-saved on shutdown (slot=auto_shutdown)")
                except Exception as exc:
                    logger.warning("Auto-save on shutdown failed: %s", exc)
            session_mgr.end_session()
    except Exception as exc:
        logger.warning("Shutdown cleanup error: %s", exc)

    # Disconnect all WebSocket clients
    try:
        from backend.api.websocket import ws_manager

        for ws in list(ws_manager.active_connections):
            try:
                await ws.close()
            except Exception:
                pass
        ws_manager.active_connections.clear()
    except Exception:
        pass


# ─── App creation ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="MVP Research Game API",
    description=(
        "REST + WebSocket API for a single-player research game combining "
        "hierarchical MDP quest systems, NPC reinforcement learning agents, "
        "and optional local LLM integration."
    ),
    version=GAME_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow all origins for local development) ────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ───────────────────────────────────────────────────────────

from backend.api.routes import router as api_router  # noqa: E402
from backend.api.websocket import ws_router  # noqa: E402

app.include_router(api_router)
app.include_router(ws_router)


# ── Static files & frontend serving ──────────────────────────────────────────

if _FRONTEND_DIR.exists():
    # Serve CSS and JS as static files
    _css_dir = _FRONTEND_DIR / "css"
    _js_dir = _FRONTEND_DIR / "js"

    if _css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(_css_dir)), name="css")

    if _js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(_js_dir)), name="js")

    # Catch-all: serve index.html for the root and any unmatched path
    _index_path = _FRONTEND_DIR / "index.html"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_index() -> FileResponse:
        """Serve the frontend index.html."""
        if _index_path.exists():
            return FileResponse(str(_index_path), media_type="text/html")
        return HTMLResponse(
            "<h1>Frontend not found</h1><p>Place index.html in frontend/</p>",
            status_code=404,
        )

    _favicon_path = _FRONTEND_DIR / "favicon.svg"

    @app.get("/favicon.ico", include_in_schema=False)
    @app.get("/favicon.svg", include_in_schema=False)
    async def serve_favicon() -> FileResponse:
        """Serve the favicon."""
        if _favicon_path.exists():
            return FileResponse(str(_favicon_path), media_type="image/svg+xml")
        return FileResponse(str(_index_path), status_code=404)

    # Serve any other static frontend assets (images, fonts, etc.)
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
else:
    @app.get("/", include_in_schema=False)
    async def no_frontend() -> dict:
        """Placeholder when frontend directory is missing."""
        return {
            "message": "MVP Research Game API is running.",
            "docs": "/docs",
            "redoc": "/redoc",
            "version": GAME_VERSION,
        }


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"], summary="Health check")
async def health_check() -> dict:
    """Simple health-check endpoint."""
    from backend.api.routes import session_mgr

    return {
        "status": "ok",
        "version": GAME_VERSION,
        "session_active": session_mgr.is_active(),
    }


# ─── Uvicorn runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
