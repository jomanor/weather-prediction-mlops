from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


# ---------------------------------------------------------------------------
# Internal / legacy raw-format helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

class Coordinates(BaseModel):
    lon: float
    lat: float

class Weather(BaseModel):
    id: int
    main: str
    description: str
    icon: str

class MainWeather(BaseModel):
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    pressure: int
    humidity: int
    sea_level: Optional[int] = None
    grnd_level: Optional[int] = None

class Wind(BaseModel):
    speed: float
    deg: int
    gust: Optional[float] = None

class Clouds(BaseModel):
    all: int

class Sys(BaseModel):
    country: str
    sunrise: int
    sunset: int


# ---------------------------------------------------------------------------
# Response models for /weather/* endpoints
# ---------------------------------------------------------------------------

class CurrentWeatherResponse(BaseModel):
    city: str
    temperature: float = Field(description="Temperature in Celsius")
    feels_like: float = Field(description="Feels like temperature")
    humidity: float = Field(description="Relative humidity percentage")
    pressure: float = Field(description="Atmospheric pressure in hPa")
    wind_speed: float = Field(description="Wind speed in m/s")
    description: str = Field(description="Weather description")
    timestamp: datetime = Field(description="Measurement timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "city": "Madrid",
                "temperature": 22.75,
                "feels_like": 22.88,
                "humidity": 69,
                "pressure": 1015,
                "wind_speed": 6.76,
                "description": "overcast clouds",
                "timestamp": "2025-01-23T16:21:09Z",
            }
        }


class WeatherStatsResponse(BaseModel):
    city: str
    period_hours: int
    avg_temperature: float
    min_temperature: float
    max_temperature: float
    avg_humidity: float
    avg_pressure: float
    total_records: int
    start_time: datetime
    end_time: datetime


class HistoricalWeatherResponse(BaseModel):
    city: str
    data: List[CurrentWeatherResponse]
    count: int


class CityComparisonResponse(BaseModel):
    cities: List[str]
    period_hours: int
    comparison: dict  # Yet to see


# ---------------------------------------------------------------------------
# Response models for /predictions/* endpoints
# ---------------------------------------------------------------------------

class WeatherPredictionResponse(BaseModel):
    """A single model-generated forecast record stored in weather_predictions."""

    city: str = Field(description="City name")
    source_timestamp: datetime = Field(
        description="Timestamp of the latest observed features used as input"
    )
    prediction_timestamp: datetime = Field(
        description="When the inference job ran"
    )
    predicted_temperature: Optional[float] = Field(
        None, description="Predicted temperature in Celsius (horizon_hours ahead)"
    )
    predicted_rain: Optional[float] = Field(
        None, description="Rain probability / binary flag (1 = will rain)"
    )
    observed_temperature: Optional[float] = Field(
        None, description="Observed temperature at source_timestamp (for error calculation)"
    )
    horizon_hours: int = Field(description="Forecast horizon in hours")
    temp_model_name: Optional[str] = Field(None, description="Name of the temperature model used")
    temp_model_version: Optional[str] = Field(None, description="Version tag of the temperature model")
    rain_model_name: Optional[str] = Field(None, description="Name of the rain model used")
    rain_model_version: Optional[str] = Field(None, description="Version tag of the rain model")

    class Config:
        json_schema_extra = {
            "example": {
                "city": "Madrid",
                "source_timestamp": "2025-06-26T12:00:00Z",
                "prediction_timestamp": "2025-06-26T13:00:00Z",
                "predicted_temperature": 28.4,
                "predicted_rain": 0.0,
                "observed_temperature": 27.1,
                "horizon_hours": 1,
                "temp_model_name": "temp_prediction_1h_GradientBoostedTrees",
                "temp_model_version": "20250626_120000",
                "rain_model_name": "rain_prediction_1h_GradientBoostedTrees",
                "rain_model_version": "20250626_120000",
            }
        }


class LatestPredictionsResponse(BaseModel):
    """Aggregated latest prediction for every city (or the requested city)."""

    count: int
    predictions: List[WeatherPredictionResponse]