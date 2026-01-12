"""Health check endpoints."""

from fastapi import APIRouter

from localcrew.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Check API health status."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/health/ready")
async def readiness_check() -> dict:
    """Check if service is ready to accept requests."""
    # TODO: Add database connectivity check
    return {
        "status": "ready",
        "database": "connected",
        "mlx": "available",
    }
