"""GeoJSON schemas for ``GET /api/map/stations``.

Properties are the latest observation per city, joined by the weather repo.
No bbox or spatial index yet: the whole 15-city `FeatureCollection` is served.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class StationProperties(BaseModel):
    city: str
    temperature: float | None = None
    apparent_temperature: float | None = None
    relative_humidity: float | None = None
    wind_speed: float | None = None
    wind_direction: float | None = None
    precipitation: float | None = None
    weather_code: int | None = None
    observed_at: datetime


class StationPoint(BaseModel):
    type: Literal["Point"] = "Point"
    coordinates: list[float]  # [longitude, latitude]

    model_config = ConfigDict(
        json_schema_extra={"example": {"type": "Point", "coordinates": [-3.7038, 40.4168]}}
    )


class StationFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: StationPoint
    properties: StationProperties


class StationCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[StationFeature]
