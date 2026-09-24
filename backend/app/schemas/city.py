"""Station registry and geocoding schemas (``docs/api-contract.md`` → v2.1)."""

from pydantic import BaseModel, ConfigDict


class City(BaseModel):
    """A station in the registry; the POST body uses the same shape."""

    name: str
    latitude: float
    longitude: float

    model_config = ConfigDict(
        json_schema_extra={"example": {"name": "Madrid", "latitude": 40.4168, "longitude": -3.7038}}
    )


class GeoResult(BaseModel):
    """One place returned by the Open-Meteo geocoding proxy."""

    name: str
    latitude: float
    longitude: float
    country: str | None = None
    admin1: str | None = None


class GeoSearchResponse(BaseModel):
    results: list[GeoResult]
