"""Health endpoints: shallow liveness and Mongo readiness."""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import Settings, get_settings
from app.db.mongo import get_db
from app.schemas.common import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

#: Readiness must fail fast: a Mongo that cannot answer within this budget is
#: not ready. Used as the command ``maxTimeMS`` and as the asyncio bound.
READY_TIMEOUT_MS = 1500


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness: never touches Mongo, so a transient blip cannot kill the instance."""
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.app_version,
        time=datetime.now(timezone.utc),
    )


@router.get("/health/ready")
async def ready(db: AsyncIOMotorDatabase = Depends(get_db)) -> JSONResponse:
    """Readiness: ping Mongo within a tight budget; 503 when it does not answer."""
    try:
        await asyncio.wait_for(
            db.command("ping", maxTimeMS=READY_TIMEOUT_MS),
            timeout=READY_TIMEOUT_MS / 1000,
        )
    except Exception:  # noqa: BLE001 - any ping failure means "not ready"
        logger.warning("Readiness check failed", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "database": db.name},
        )
    return JSONResponse(status_code=200, content={"status": "ok", "database": db.name})
