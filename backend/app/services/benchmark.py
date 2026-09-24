"""Benchmark service: our model vs AEMET vs observed, joined on timestamp."""

import asyncio
import logging
import math
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from fastapi import Depends

from app.repositories.prediction_repo import PredictionRepository, get_prediction_repo
from app.repositories.weather_repo import WeatherRepository, get_weather_repo
from app.schemas.benchmark import (
    AemetStatus,
    BenchmarkResponse,
    BenchmarkSummary,
    CityBenchmark,
    Metrics,
    MetricsPair,
    SeriesPoint,
)
from app.services.aemet import AemetService, get_aemet_service

logger = logging.getLogger(__name__)

#: Concurrent AEMET requests when summarising every city (the API rate limits).
AEMET_CONCURRENCY = 4


class CityDataNotFound(Exception):
    """Raised when a city has neither observations nor predictions in the window."""


def compute_metrics(pairs: Iterable[tuple[float | None, float | None]]) -> Metrics:
    """MAE / RMSE / bias over the pairs where both sides are non-null.

    ``n`` is the number of such pairs; all metrics are null when there are none.
    """
    errors = [pred - obs for obs, pred in pairs if obs is not None and pred is not None]
    if not errors:
        return Metrics(mae=None, rmse=None, bias=None, n=0)

    count = len(errors)
    mae = sum(abs(error) for error in errors) / count
    rmse = math.sqrt(sum(error * error for error in errors) / count)
    bias = sum(errors) / count
    return Metrics(mae=round(mae, 2), rmse=round(rmse, 2), bias=round(bias, 2), n=count)


def build_series(
    observed: dict[datetime, float],
    model: dict[datetime, float],
    aemet: dict[datetime, float],
) -> list[SeriesPoint]:
    """Timestamp-joined series, ascending. Residuals are null unless observed is known."""
    series: list[SeriesPoint] = []
    for timestamp in sorted(set(observed) | set(model) | set(aemet)):
        observed_value = observed.get(timestamp)
        model_value = model.get(timestamp)
        aemet_value = aemet.get(timestamp)
        series.append(
            SeriesPoint(
                timestamp=timestamp,
                observed=observed_value,
                model=model_value,
                aemet=aemet_value,
                residual_model=(
                    None
                    if observed_value is None or model_value is None
                    else round(model_value - observed_value, 2)
                ),
                residual_aemet=(
                    None
                    if observed_value is None or aemet_value is None
                    else round(aemet_value - observed_value, 2)
                ),
            )
        )
    return series


class BenchmarkService:
    def __init__(
        self,
        weather_repo: WeatherRepository,
        prediction_repo: PredictionRepository,
        aemet: AemetService,
    ) -> None:
        self._weather = weather_repo
        self._predictions = prediction_repo
        self._aemet = aemet

    async def city_benchmark(
        self, city: str, hours: int = 24, now: datetime | None = None
    ) -> BenchmarkResponse:
        now = now or datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        observations = await self._weather.find_range(
            city, start, now, limit=1000, newest_first=False
        )
        predictions = await self._predictions.find_range(city, start, now)

        observed = {
            point.observed_at: point.temperature
            for point in observations
            if point.temperature is not None
        }
        model = {
            prediction.source_timestamp: prediction.predicted_temperature
            for prediction in predictions
            if prediction.predicted_temperature is not None
        }
        if not observed and not model:
            raise CityDataNotFound(city)

        aemet_temps: dict[datetime, float] = {}
        forecast = await self._aemet.forecast(city)
        if forecast.available:
            aemet_temps = {
                timestamp: value
                for timestamp, value in forecast.temps.items()
                if start <= timestamp <= now
            }
        aemet_status = AemetStatus(
            available=forecast.available,
            error=forecast.error,
            issued_at=forecast.issued_at,
        )

        series = build_series(observed, model, aemet_temps)
        return BenchmarkResponse(
            city=city,
            hours=hours,
            generated_at=now,
            aemet=aemet_status,
            series=series,
            metrics=MetricsPair(
                model=compute_metrics((point.observed, point.model) for point in series),
                aemet=compute_metrics((point.observed, point.aemet) for point in series),
            ),
        )

    async def summary(self, hours: int = 24, now: datetime | None = None) -> BenchmarkSummary:
        now = now or datetime.now(timezone.utc)
        cities = await self._weather.list_cities()

        semaphore = asyncio.Semaphore(AEMET_CONCURRENCY)

        async def one(city: str) -> BenchmarkResponse | None:
            async with semaphore:
                try:
                    return await self.city_benchmark(city, hours=hours, now=now)
                except CityDataNotFound:
                    return None

        results = await asyncio.gather(*(one(city) for city in cities))

        return BenchmarkSummary(
            generated_at=now,
            aemet_configured=self._aemet.configured,
            cities=[
                CityBenchmark(
                    city=result.city,
                    n=len(result.series),
                    model=result.metrics.model,
                    aemet=result.metrics.aemet,
                )
                for result in results
                if result is not None
            ],
        )


def get_benchmark_service(
    weather_repo: WeatherRepository = Depends(get_weather_repo),
    prediction_repo: PredictionRepository = Depends(get_prediction_repo),
    aemet: AemetService = Depends(get_aemet_service),
) -> BenchmarkService:
    return BenchmarkService(weather_repo, prediction_repo, aemet)
