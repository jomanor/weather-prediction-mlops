from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

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

# Response Models for API endpoints
class CurrentWeatherResponse(BaseModel):
    city: str
    temperature: float = Field(description="Temperature in Celsius")
    feels_like: float = Field(description="Feels like temperature")
    humidity: int = Field(description="Humidity percentage")
    pressure: int = Field(description="Atmospheric pressure in hPa")
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
                "timestamp": "2025-01-23T16:21:09Z"
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