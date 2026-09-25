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
        temp_lower=to_float(doc.get("temp_lower")),
        temp_upper=to_float(doc.get("temp_upper")),
        interval_level=to_float(doc.get("interval_level")),
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

    async def latest_per_city(self, horizon: int | None = None) -> list[Prediction]:
        # One row per ``(city, horizon_hours)``: the inner sort matches the
        # existing ``{city: 1, horizon_hours: 1, prediction_timestamp: -1}``
        # index so the group is fed without a blocking SORT (city-prefixed per
        # the M0 rule); the outer sort orders the returned rows. ``allowDiskUse``
        # guards the aggregate if the planner cannot use the index. With
        # ``horizon`` set, the ``$match`` pins one horizon, so the result is one
        # row per city for that horizon.
        pipeline: list[dict[str, Any]] = []
        if horizon is not None:
            pipeline.append({"$match": {"horizon_hours": horizon}})
        pipeline += [
            {"$sort": {"city": 1, "horizon_hours": 1, "prediction_timestamp": -1}},
            {
                "$group": {
                    "_id": {"city": "$city", "horizon_hours": "$horizon_hours"},
                    "latest": {"$first": "$$ROOT"},
                }
            },
            {"$replaceRoot": {"newRoot": "$latest"}},
            {"$sort": {"city": 1, "horizon_hours": 1}},
        ]
        docs = [doc async for doc in self._collection.aggregate(pipeline, allowDiskUse=True)]
        return _map_all(docs)

    async def for_city(
        self, city: str, limit: int = 48, horizon: int | None = None
    ) -> list[Prediction]:
        query: dict[str, Any] = {"city": city}
        if horizon is not None:
            query["horizon_hours"] = horizon
        cursor = self._collection.find(query).sort("prediction_timestamp", -1).limit(limit)
        return _map_all([doc async for doc in cursor])

    async def find_range(self, city: str, start: datetime, end: datetime) -> list[Prediction]:
        cursor = self._collection.find(
            {"city": city, "source_timestamp": {"$gte": start, "$lte": end}}
        ).sort("source_timestamp", 1)
        return _map_all([doc async for doc in cursor])


def get_prediction_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> PredictionRepository:
    return PredictionRepository(db)
