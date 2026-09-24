"""Statistics over a window, including missing measurements."""

from datetime import datetime, timedelta, timezone

from app.schemas.weather import CurrentWeather
from app.services.stats import compute_stats

UTC = timezone.utc


def _point(offset_hours: int, **overrides) -> CurrentWeather:
    values = {
        "city": "Madrid",
        "temperature": 20.0,
        "humidity": 50.0,
        "pressure": 1000.0,
        "wind_speed": 5.0,
        "precipitation": 0.0,
    }
    values.update(overrides)
    return CurrentWeather(
        **values, observed_at=datetime(2026, 9, 23, 12, tzinfo=UTC) + timedelta(hours=offset_hours)
    )


def test_stats_over_all_fields():
    stats = compute_stats(
        "Madrid",
        24,
        [_point(0, temperature=10.0), _point(1, temperature=20.0), _point(2, temperature=30.0)],
    )

    assert stats.count == 3
    assert stats.temperature.avg == 20.0
    assert stats.temperature.min == 10.0
    assert stats.temperature.max == 30.0
    assert stats.start < stats.end


def test_stats_skip_missing_values_instead_of_zeroing_them():
    stats = compute_stats(
        "Madrid",
        24,
        [
            _point(0, temperature=10.0, precipitation=0.5),
            _point(1, temperature=None, precipitation=None),
        ],
    )

    assert stats.count == 2
    assert stats.temperature.avg == 10.0
    assert stats.precipitation_total == 0.5


def test_stats_with_every_value_missing_are_null():
    stats = compute_stats(
        "Madrid",
        24,
        [_point(0, temperature=None, humidity=None, precipitation=None)],
    )

    assert stats.temperature.avg is None
    assert stats.humidity.avg is None
    assert stats.precipitation_total is None
    assert stats.count == 1


def test_stats_without_points():
    stats = compute_stats("Madrid", 24, [])

    assert stats.count == 0
    assert stats.temperature.avg is None
    assert stats.start is None
