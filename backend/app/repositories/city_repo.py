"""MongoDB access for the ``cities`` station registry.

The registry is the source of truth for ingestion: producers read it, the API
exposes and manages it. ``name_key`` is the case-insensitive identity used by
the unique index and by lookups, so names stay unique regardless of casing
while the operator's original casing is preserved in ``name``.
"""

import logging
from typing import Any

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.cities import DEFAULT_CITIES
from app.core.coerce import to_float
from app.db.mongo import get_db
from app.schemas.city import City

logger = logging.getLogger(__name__)


def name_key(name: str) -> str:
    """Case-insensitive identity for a station name."""
    return name.strip().casefold()


def city_from_doc(doc: dict[str, Any]) -> City | None:
    """``cities`` document -> City, or None when it cannot be trusted."""
    name = doc.get("name")
    latitude = to_float(doc.get("latitude"))
    longitude = to_float(doc.get("longitude"))
    if not name or latitude is None or longitude is None:
        return None
    return City(name=name, latitude=latitude, longitude=longitude)


class CityRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    @property
    def _collection(self):
        return self._db["cities"]

    async def list(self) -> list[City]:
        cursor = self._collection.find({}).sort("name", 1)
        docs = [doc async for doc in cursor]
        return [city for city in (city_from_doc(doc) for doc in docs) if city is not None]

    async def get(self, name: str) -> City | None:
        doc = await self._collection.find_one({"name_key": name_key(name)})
        return city_from_doc(doc) if doc is not None else None

    async def upsert(self, city: City) -> City:
        doc = {
            "name": city.name,
            "name_key": name_key(city.name),
            "latitude": city.latitude,
            "longitude": city.longitude,
        }
        await self._collection.replace_one({"name_key": doc["name_key"]}, doc, upsert=True)
        return city

    async def delete(self, name: str) -> bool:
        result = await self._collection.delete_one({"name_key": name_key(name)})
        return result.deleted_count > 0


async def seed_default_cities(db: AsyncIOMotorDatabase) -> bool:
    """Insert ``DEFAULT_CITIES`` when the registry is empty.

    Idempotent and never raises: a Mongo hiccup at startup is logged and the
    service starts anyway. Returns True only when it actually inserted rows.
    """
    collection = db["cities"]
    try:
        if await collection.count_documents({}) > 0:
            return False
        await collection.insert_many(
            [
                {"name": name, "name_key": name_key(name), "latitude": lat, "longitude": lon}
                for name, lat, lon in DEFAULT_CITIES
            ],
            ordered=False,
        )
    except Exception:  # noqa: BLE001 - startup must survive Mongo being unreachable
        logger.warning("Could not seed the cities registry; starting without it", exc_info=True)
        return False
    logger.info("Seeded %d default cities", len(DEFAULT_CITIES))
    return True


def get_city_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> CityRepository:
    return CityRepository(db)
