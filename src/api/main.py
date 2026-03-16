"""
FastAPI application entry point.

Configures the application with CORS middleware, includes API routes,
and sets up logging and exception handling.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.config.settings import AppSettings
from src.utils.logger import get_logger, setup_logging

settings = AppSettings()
logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    setup_logging(level=settings.log_level)
    logger.info(
        "Starting %s v%s", settings.app_name, settings.app_version
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="ML Model Drift Monitor",
    description=(
        "Sistema de monitoramento de modelos ML em producao com deteccao "
        "de data drift, concept drift e degradacao de performance."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "status_code": 500,
        },
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
