"""FastAPI application factory.

Run with: ``uvicorn app.main:app --host 0.0.0.0 --port 8000``
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.cache import TTLCache
from app.core.config import Settings, get_settings
from app.core.logging import RequestContextMiddleware, configure_logging
from app.db.mongo import create_client, ensure_indexes, ping
from app.repositories.city_repo import seed_default_cities
from app.routers import benchmark, cities, health, models, predictions, weather
from app.routers.map import router as map_router
from app.services.aemet import AemetService
from app.services.geo import GeocodingService

logger = logging.getLogger(__name__)

#: Bound on awaiting the background index/seed task during shutdown. A slow
#: ``create_index`` on a free-tier Atlas cluster must not hold the process open
#: forever; on timeout the task is cancelled and shutdown continues.
PREPARE_SHUTDOWN_TIMEOUT_SECONDS = 10.0


async def _prepare_database(db) -> None:
    """Ensure indexes and seed the registry, once per process, off the request path.

    Runs as a background task so a cold request never waits for it. Failures are
    logged with a full traceback (loudly) but do not kill an already-serving
    instance: index creation is retried on the next cold start.
    """
    try:
        await ensure_indexes(db)
    except Exception:  # noqa: BLE001 - logged loudly, service keeps serving
        logger.exception("ensure_indexes failed; continuing without guaranteed indexes")
    await seed_default_cities(db)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = create_client(settings)
        try:
            await ping(client)
        except Exception:
            logger.exception("MongoDB is not reachable; refusing to start")
            client.close()
            raise

        db = client[settings.mongo_db]
        app.state.mongo_client = client
        app.state.db = db
        app.state.aemet = AemetService(settings)
        app.state.geo = GeocodingService(settings)
        # One background task per process; awaited on shutdown so it is never
        # orphaned. ``ping`` above already guaranteed Mongo is reachable, so the
        # first request is served without waiting for indexes/seeding.
        app.state.prepare_database_task = (
            asyncio.create_task(_prepare_database(db)) if settings.ensure_indexes_on_start else None
        )
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
            task = app.state.prepare_database_task
            if task is not None:
                # Bounded wait: ``asyncio.wait`` returns ``done`` empty on timeout
                # and leaves the task running, instead of blocking shutdown.
                done, _ = await asyncio.wait({task}, timeout=PREPARE_SHUTDOWN_TIMEOUT_SECONDS)
                if task in done:
                    error = task.exception()
                    if error is not None:
                        logger.exception("Background database preparation failed", exc_info=error)
                else:
                    logger.warning(
                        "Background database preparation did not finish within %.0fs; cancelling",
                        PREPARE_SHUTDOWN_TIMEOUT_SECONDS,
                    )
                    task.cancel()
            await app.state.aemet.aclose()
            await app.state.geo.aclose()
            client.close()
            logger.info("MongoDB client closed")

    app = FastAPI(
        title="MLOps Weather API",
        version=settings.app_version,
        description="Observations, Spark model predictions, and an AEMET benchmark.",
        lifespan=lifespan,
    )

    # Single process => one in-memory cache; created per app so tests and any
    # future multi-app embedding never share state across instances.
    app.state.ttl_cache = TTLCache()

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # The frontend is cross-origin (Netlify -> Render); without this JS
        # cannot read the ETag or show the X-Data-Age-Seconds freshness badge.
        expose_headers=["ETag", "Cache-Control", "X-Data-Age-Seconds"],
    )

    api = APIRouter(prefix="/api")
    api.include_router(health.router)
    api.include_router(cities.router)
    api.include_router(weather.router)
    api.include_router(map_router)
    api.include_router(predictions.router)
    api.include_router(benchmark.router)
    api.include_router(models.router)
    app.include_router(api)

    return app


app = create_app()
