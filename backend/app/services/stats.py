"""Descriptive statistics over a window of observations."""

from app.schemas.weather import CurrentWeather, Stats, WeatherStats


def _stats(values: list[float]) -> Stats:
    if not values:
        return Stats(avg=None, min=None, max=None)
    return Stats(
        avg=round(sum(values) / len(values), 2),
        min=round(min(values), 2),
        max=round(max(values), 2),
    )


def compute_stats(city: str, hours: int, points: list[CurrentWeather]) -> WeatherStats:
    """Stats over *points*. Missing measurements are excluded, never treated as 0."""
    temperatures = [p.temperature for p in points if p.temperature is not None]
    humidities = [p.humidity for p in points if p.humidity is not None]
    pressures = [p.pressure for p in points if p.pressure is not None]
    wind_speeds = [p.wind_speed for p in points if p.wind_speed is not None]
    precipitation = [p.precipitation for p in points if p.precipitation is not None]

    timestamps = [p.observed_at for p in points]
    return WeatherStats(
        city=city,
        hours=hours,
        count=len(points),
        temperature=_stats(temperatures),
        humidity=_stats(humidities),
        pressure=_stats(pressures),
        wind_speed=_stats(wind_speeds),
        precipitation_total=round(sum(precipitation), 2) if precipitation else None,
        start=min(timestamps) if timestamps else None,
        end=max(timestamps) if timestamps else None,
    )
