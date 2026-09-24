"""Weather observation schemas (``docs/api-contract.md`` → CurrentWeather / WeatherPoint)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CurrentWeather(BaseModel):
    """A single Open-Meteo observation. Nullable fields are always present."""

    city: str
    latitude: float | None = None
    longitude: float | None = None
    temperature: float | None = None  # °C
    apparent_temperature: float | None = None  # °C
    humidity: float | None = None  # %
    pressure: float | None = None  # hPa
    wind_speed: float | None = None  # km/h
    wind_direction: float | None = None  # degrees, 0 = N
    precipitation: float | None = None  # mm
    cloud_cover: float | None = None  # %
    weather_code: int | None = None  # WMO code
    observed_at: datetime  # ISO-8601 UTC

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Madrid",
                "latitude": 40.4168,
                "longitude": -3.7038,
                "temperature": 21.4,
                "apparent_temperature": 21.0,
                "humidity": 61.0,
                "pressure": 1014.2,
                "wind_speed": 11.3,
                "wind_direction": 220.0,
                "precipitation": 0.0,
                "cloud_cover": 40.0,
                "weather_code": 2,
                "observed_at": "2026-09-23T14:00:00Z",
            }
        }
    )


#: The contract names the history series element ``WeatherPoint``; it is the
#: exact same shape as a current observation.
WeatherPoint = CurrentWeather


class CurrentWeatherList(BaseModel):
    count: int
    stations: list[CurrentWeather]


class HistoryResponse(BaseModel):
    city: str
    hours: int
    count: int
    points: list[WeatherPoint]


class Stats(BaseModel):
    avg: float | None = None
    min: float | None = None
    max: float | None = None


class WeatherStats(BaseModel):
    city: str
    hours: int
    count: int
    temperature: Stats
    humidity: Stats
    pressure: Stats
    wind_speed: Stats
    precipitation_total: float | None = None
    start: datetime | None = None
    end: datetime | None = None
