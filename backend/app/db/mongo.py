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
    "weather_features": [[("city", 1), ("timestamp", -1)]],
    "weather_predictions": [
        [("city", 1), ("source_timestamp", -1)],
        [("city", 1), ("prediction_timestamp", -1)],
        [("city", 1), ("horizon_hours", 1), ("prediction_timestamp", -1)],
    ],
    "model_registry": [[("model_name", 1), ("timestamp", -1)]],
}

# collection -> list of index specs that must also be unique
UNIQUE_INDEXES: dict[str, list[list[tuple[str, int]]]] = {
    "cities": [[("name_key", 1)]],
}

# collection -> list of (field, expireAfterSeconds) TTL specs. The field must be
# a BSON date; TTL deletes are irreversible, so these windows ARE the retention
# policy. ``weather_features`` is the training input and deliberately has none.
# ``timestamp`` is a BSON date for ``raw_weather`` (ingest/consumer) and
# ``weather_data`` (consumer); ``prediction_timestamp`` is a Spark timestamp.
TTL_INDEXES: dict[str, list[tuple[str, int]]] = {
    "raw_weather": [("timestamp", 180 * 86400)],
    "weather_predictions": [("prediction_timestamp", 90 * 86400)],
    "weather_data": [("timestamp", 30 * 86400)],
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
    # ``create_index`` creates the collection on first use, so a collection that
    # does not exist yet is not an error here.
    for collection, specs in INDEXES.items():
        for spec in specs:
            await db[collection].create_index(spec)
    for collection, specs in UNIQUE_INDEXES.items():
        for spec in specs:
            await db[collection].create_index(spec, unique=True)
    for collection, specs in TTL_INDEXES.items():
        for field, expire_after_seconds in specs:
            await db[collection].create_index([(field, 1)], expireAfterSeconds=expire_after_seconds)
    logger.info(
        "MongoDB indexes ensured for %s",
        ", ".join(sorted(set(INDEXES) | set(UNIQUE_INDEXES) | set(TTL_INDEXES))),
    )


def get_client(request: Request) -> AsyncIOMotorClient:
    return request.app.state.mongo_client


async def get_db(request: Request) -> AsyncIterator[AsyncIOMotorDatabase]:
    """Yield the request-scoped database handle from ``app.state``."""
    yield request.app.state.db
