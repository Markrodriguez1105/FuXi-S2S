"""Health check endpoints."""

from fastapi import APIRouter

from services.mongo_client import get_mongo_client

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "service": "weather-service",
        "database": db_status,
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        return {"ready": True}
    except Exception:
        return {"ready": False}
