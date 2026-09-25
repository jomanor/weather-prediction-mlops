"""MongoDB access for ``weather_predictions`` (written by the Spark inference job)."""

from datetime import datetime
from typing import Any

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.coerce import as_utc, to_float, to_int
from app.db.mongo import get_db
from app.schemas.predictions import Prediction


def prediction_from_doc(doc: dict[str, Any]) -> Prediction | None:
    """``weather_predictions`` document -> Prediction."""
    source_timestamp = as_utc(doc.get("source_timestamp"))
    if source_timestamp is None:
        return None

    return Prediction(
        city=doc.get("city", "unknown"),
        source_timestamp=source_timestamp,
        prediction_timestamp=as_utc(doc.get("prediction_timestamp")) or source_timestamp,
        horizon_hours=to_int(doc.get("horizon_hours")) or 1,
        predicted_temperature=to_float(doc.get("predicted_temperature")),
        predicted_rain=to_float(doc.get("predicted_rain")),
        observed_temperature=to_float(doc.get("observed_temperature")),
        temp_model_name=doc.get("temp_model_name"),
        temp_model_version=doc.get("temp_model_version"),
        rain_model_name=doc.get("rain_model_name"),
        rain_model_version=doc.get("rain_model_version"),
    )


def _map_all(docs: list[dict[str, Any]]) -> list[Prediction]:
    return [pred for pred in (prediction_from_doc(doc) for doc in docs) if pred is not None]


class PredictionRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    @property
    def _collection(self):
        return self._db["weather_predictions"]

    async def latest_per_city(self) -> list[Prediction]:
        # Sort by (city, prediction_timestamp) to match the
        # ``{city: 1, prediction_timestamp: -1}`` index, and allow disk use so
        # the blocking sort cannot fail with MongoDB code 292 as the collection
        # grows. See the same pattern in ``WeatherRepository.latest_per_city``.
        pipeline = [
            {"$sort": {"city": 1, "prediction_timestamp": -1}},
            {"$group": {"_id": "$city", "latest": {"$first": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$latest"}},
            {"$sort": {"city": 1}},
        ]
        docs = [doc async for doc in self._collection.aggregate(pipeline, allowDiskUse=True)]
        return _map_all(docs)

    async def for_city(self, city: str, limit: int = 48) -> list[Prediction]:
        cursor = self._collection.find({"city": city}).sort("prediction_timestamp", -1).limit(limit)
        return _map_all([doc async for doc in cursor])

    async def find_range(self, city: str, start: datetime, end: datetime) -> list[Prediction]:
        cursor = self._collection.find(
            {"city": city, "source_timestamp": {"$gte": start, "$lte": end}}
        ).sort("source_timestamp", 1)
        return _map_all([doc async for doc in cursor])


def get_prediction_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> PredictionRepository:
    return PredictionRepository(db)
