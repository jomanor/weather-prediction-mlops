"""FastAPI application factory.

Run with: ``uvicorn app.main:app --host 0.0.0.0 --port 8000``
"""

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.logging import RequestContextMiddleware, configure_logging
from app.db.mongo import create_client, ensure_indexes, ping
from app.routers import benchmark, health, models, predictions, weather
from app.services.aemet import AemetService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = create_client(settings)
        try:
            await ping(client)
            db = client[settings.mongo_db]
            await ensure_indexes(db)
        except Exception:
            logger.exception("MongoDB is not reachable; refusing to start")
            client.close()
            raise

        app.state.mongo_client = client
        app.state.db = db
        app.state.aemet = AemetService(settings)
        logger.info(
            "Started %s v%s (db=%s, aemet_configured=%s)",
            settings.service_name,
            settings.app_version,
            settings.mongo_db,
            app.state.aemet.configured,
        )
        try:
            yield
        finally:
            await app.state.aemet.aclose()
            client.close()
            logger.info("MongoDB client closed")

    app = FastAPI(
        title="MLOps Weather API",
        version=settings.app_version,
        description="Observations, Spark model predictions, and an AEMET benchmark.",
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = APIRouter(prefix="/api")
    api.include_router(health.router)
    api.include_router(weather.cities_router)
    api.include_router(weather.router)
    api.include_router(predictions.router)
    api.include_router(benchmark.router)
    api.include_router(models.router)
    app.include_router(api)

    return app


app = create_app()
