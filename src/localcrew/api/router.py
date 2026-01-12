"""Main API router combining all route modules."""

from fastapi import APIRouter

from localcrew.api.routes import crews, executions, workflows, reviews, health

api_router = APIRouter()

# Health check
api_router.include_router(health.router, tags=["health"])

# Core routes
api_router.include_router(crews.router, prefix="/crews", tags=["crews"])
api_router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
api_router.include_router(executions.router, prefix="/executions", tags=["executions"])
api_router.include_router(reviews.router, prefix="/reviews", tags=["reviews"])
