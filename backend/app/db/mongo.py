"""MongoDB client lifecycle, index management and the ``get_db`` dependency.

The client lives on ``app.state`` and is created/closed by the FastAPI
lifespan; nothing here runs at import time.
"""

import logging
from collections.abc import AsyncIterator

from fastapi import Request
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import Settings

logger = logging.getLogger(__name__)

# collection -> list of index specs
INDEXES: dict[str, list[list[tuple[str, int]]]] = {
    "weather_data": [[("city", 1), ("timestamp", -1)]],
    "raw_weather": [[("city", 1), ("timestamp", -1)]],
    "weather_predictions": [
        [("city", 1), ("source_timestamp", -1)],
        [("city", 1), ("prediction_timestamp", -1)],
    ],
    "model_registry": [[("model_name", 1), ("timestamp", -1)]],
}


def create_client(settings: Settings) -> AsyncIOMotorClient:
    """Create the Motor client. ``tz_aware`` guarantees UTC-aware datetimes."""
    return AsyncIOMotorClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=settings.mongo_timeout_ms,
        tz_aware=True,
    )


async def ping(client: AsyncIOMotorClient) -> None:
    await client.admin.command("ping")


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    for collection, specs in INDEXES.items():
        for spec in specs:
            await db[collection].create_index(spec)
    logger.info("MongoDB indexes ensured for %s", ", ".join(sorted(INDEXES)))


def get_client(request: Request) -> AsyncIOMotorClient:
    return request.app.state.mongo_client


async def get_db(request: Request) -> AsyncIterator[AsyncIOMotorDatabase]:
    """Yield the request-scoped database handle from ``app.state``."""
    yield request.app.state.db
