"""Benchmark math, series joining and the service contract shape."""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.benchmark import (
    BenchmarkService,
    CityDataNotFound,
    build_series,
    compute_metrics,
)
from tests.fakes import FakeAemetService, FakePredictionRepository, FakeWeatherRepository

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Metric math
# ---------------------------------------------------------------------------


def test_metrics_on_empty_input_are_null():
    metrics = compute_metrics([])
    assert metrics.mae is None
    assert metrics.rmse is None
    assert metrics.bias is None
    assert metrics.n == 0


def test_metrics_ignore_pairs_with_missing_values():
    pairs = [
        (20.0, None),
        (None, 21.0),
        (None, None),
    ]
    metrics = compute_metrics(pairs)
    assert metrics.n == 0
    assert metrics.mae is None


def test_metrics_use_only_complete_pairs():
    pairs = [
        (10.0, 12.0),  # +2
        (20.0, 18.0),  # -2
        (None, 99.0),  # dropped
        (30.0, None),  # dropped
    ]
    metrics = compute_metrics(pairs)

    assert metrics.n == 2
    assert metrics.mae == 2.0
    assert metrics.rmse == 2.0
    assert metrics.bias == 0.0


def test_metrics_known_values():
    pairs = [(21.0, 21.5), (20.0, 20.4)]
    metrics = compute_metrics(pairs)

    assert metrics.n == 2
    assert metrics.mae == 0.45
    assert metrics.rmse == 0.45  # sqrt((0.25 + 0.16) / 2) = 0.4528 -> 0.45
    assert metrics.bias == 0.45


def test_bias_keeps_its_sign():
    metrics = compute_metrics([(10.0, 9.0)])
    assert metrics.bias == -1.0
    assert metrics.mae == 1.0


# ---------------------------------------------------------------------------
# Series joining
# ---------------------------------------------------------------------------


def test_series_joins_on_timestamp_and_nulls_missing_side():
    t0 = datetime(2026, 9, 23, 10, 0, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)

    series = build_series(
        observed={t0: 20.0, t1: 21.0},
        model={t1: 21.5, t2: 22.5},
        aemet={t0: 20.2},
    )

    assert [point.timestamp for point in series] == [t0, t1, t2]

    assert series[0].observed == 20.0
    assert series[0].model is None
    assert series[0].aemet == 20.2
    assert series[0].residual_model is None
    assert series[0].residual_aemet == 0.2

    assert series[1].residual_model == 0.5
    assert series[1].residual_aemet is None

    assert series[2].observed is None
    assert series[2].model == 22.5
    assert series[2].residual_model is None


def test_series_is_empty_when_everything_is_missing():
    assert build_series({}, {}, {}) == []


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@pytest.fixture
def service(weather_repo, prediction_repo, aemet):
    return BenchmarkService(weather_repo, prediction_repo, aemet)


async def test_city_benchmark_shape(service, now, aemet_temps):
    response = await service.city_benchmark("Madrid", hours=24, now=now)

    assert response.city == "Madrid"
    assert response.hours == 24
    assert response.generated_at == now
    assert response.aemet.available is True
    assert response.aemet.error is None
    assert len(response.series) == 3

    newest = response.series[0]
    assert newest.observed == 20.0
    assert newest.model == 20.4
    assert newest.aemet == 20.2
    assert newest.residual_model == 0.4
    assert newest.residual_aemet == 0.2

    assert response.metrics.model.n == 2
    assert response.metrics.model.mae == 0.45
    assert response.metrics.aemet.n == 2
    assert response.metrics.aemet.mae == 0.5


async def test_city_benchmark_marks_aemet_unavailable(weather_repo, prediction_repo, now):
    aemet = FakeAemetService(error="AEMET request failed: boom")
    service = BenchmarkService(weather_repo, prediction_repo, aemet)

    response = await service.city_benchmark("Madrid", hours=24, now=now)

    assert response.aemet.available is False
    assert response.aemet.error == "AEMET request failed: boom"
    assert all(point.aemet is None for point in response.series)
    assert response.metrics.aemet.n == 0
    assert response.metrics.aemet.mae is None
    # observed vs model is unaffected
    assert response.metrics.model.n == 2


async def test_city_benchmark_without_data_raises(now):
    service = BenchmarkService(
        FakeWeatherRepository([]), FakePredictionRepository([]), FakeAemetService()
    )
    with pytest.raises(CityDataNotFound):
        await service.city_benchmark("Madrid", hours=24, now=now)


async def test_city_benchmark_ignores_data_outside_the_window(service, now):
    # 48h window includes the 2h of data; a 1h window would miss it
    response = await service.city_benchmark("Madrid", hours=1, now=now)
    assert len(response.series) == 2
    assert response.metrics.model.n == 1


async def test_summary_lists_each_city(service, now):
    summary = await service.summary(hours=24, now=now)

    assert summary.generated_at == now
    assert summary.aemet_configured is True
    assert [entry.city for entry in summary.cities] == ["Madrid"]
    assert summary.cities[0].n == 3
    assert summary.cities[0].model.mae == 0.45
    assert summary.cities[0].aemet.mae == 0.5


async def test_summary_skips_cities_without_data(now):
    service = BenchmarkService(
        FakeWeatherRepository([]), FakePredictionRepository([]), FakeAemetService()
    )
    summary = await service.summary(hours=24, now=now)
    assert summary.cities == []


async def test_summary_reports_unconfigured_aemet(weather_repo, prediction_repo, now):
    aemet = FakeAemetService(configured=False)
    service = BenchmarkService(weather_repo, prediction_repo, aemet)
    summary = await service.summary(hours=24, now=now)
    assert summary.aemet_configured is False
