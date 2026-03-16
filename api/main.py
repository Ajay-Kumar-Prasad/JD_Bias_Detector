"""
main.py — FastAPI application entry point

Run:
    uvicorn api.main:app --reload           # dev
    uvicorn api.main:app --host 0.0.0.0     # production
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analyze, health
from api.dependencies import get_classifier, get_rewriter, get_scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models at startup so the first request isn't slow."""
    print("[Startup] Loading models...")
    get_classifier()
    get_rewriter()
    get_scorer()
    print("[Startup] All models ready.")
    yield
    print("[Shutdown] Cleaning up.")


app = FastAPI(
    title="JD Bias Detector API",
    description=(
        "Detects gender-coded, ageist, exclusionary, and ability-coded language "
        "in job descriptions. Returns flagged spans, explanations, neutral rewrites, "
        "and an inclusivity score."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,  tags=["health"])
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
