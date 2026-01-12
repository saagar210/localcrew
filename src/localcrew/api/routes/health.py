"""Health check endpoints."""

from fastapi import APIRouter

from localcrew.core.config import settings
from localcrew.integrations.kas import get_kas

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
    """Check if service is ready to accept requests.

    Returns status of all service dependencies including
    optional integrations like KAS.
    """
    services = {
        "database": "connected",  # TODO: Add actual database check
        "mlx": "available",
    }

    # Add KAS status if enabled
    if settings.kas_enabled:
        kas = get_kas()
        if kas:
            try:
                is_healthy = await kas.health_check()
                services["kas"] = "connected" if is_healthy else "unavailable"
            except Exception:
                services["kas"] = "unavailable"
        else:
            services["kas"] = "unavailable"
    else:
        services["kas"] = "disabled"

    # Determine overall status
    # "degraded" if optional services (kas) are unavailable but core is ok
    # "error" if core services fail
    core_healthy = services["database"] == "connected" and services["mlx"] == "available"

    if not core_healthy:
        status = "error"
    elif services["kas"] == "unavailable":
        status = "degraded"
    else:
        status = "ready"

    return {
        "status": status,
        "services": services,
    }
