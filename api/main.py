"""
main.py — FastAPI application entry point

Run:
    uvicorn api.main:app --reload           # dev
    uvicorn api.main:app --host 0.0.0.0     # production
"""
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from api.routes import analyze, health
    from api.dependencies import get_rewriter, get_scorer
    from api.security import verify_api_key
except ModuleNotFoundError:
    # Support running from inside `api/` via `uvicorn main:app`.
    from routes import analyze, health
    from dependencies import get_rewriter, get_scorer
    from security import verify_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load remaining shared services at startup."""
    print("[Startup] Loading services...")
    get_rewriter()
    get_scorer()
    print("[Startup] Services ready.")
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

app.include_router(health.router, tags=["health"])
app.include_router(
    analyze.router,
    prefix="/analyze",
    tags=["analyze"],
    dependencies=[Depends(verify_api_key)],
)


@app.get("/", summary="API root")
def root():
    return {
        "name": "JD Bias Detector API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }
