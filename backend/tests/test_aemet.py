"""AEMET service: parser, two-step flow, retries, decoding, cache, failure modes."""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.services.aemet import AEMET_INE_CODES, AemetService, parse_forecast

FIXTURE = Path(__file__).parent / "fixtures" / "aemet_almeria_horaria.json"
UTC = timezone.utc


def _settings(**overrides) -> Settings:
    values = {
        "aemet_api_key": "test-key",
        "aemet_retries": 3,
        "aemet_backoff_seconds": 0.0,
        "aemet_cache_ttl_seconds": 3600,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def fixture_bytes() -> bytes:
    return FIXTURE.read_bytes()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_fixture_is_really_iso_8859_15(fixture_bytes):
    assert b"\xc3\xad" not in fixture_bytes  # not UTF-8
    assert b"\xed" in fixture_bytes  # "í" in ISO-8859-15
    assert "Almería" in fixture_bytes.decode("iso-8859-15")


def test_parse_forecast_maps_hours_and_skips_blanks(fixture_bytes):
    payload = json.loads(fixture_bytes.decode("iso-8859-15"))

    issued_at, temps = parse_forecast(payload)

    # elaborado 2026-09-23T06:00 local (CEST, UTC+2)
    assert issued_at == datetime(2026, 9, 23, 4, 0, tzinfo=UTC)
    # hour 0 local -> 22:00Z the previous day
    assert temps[datetime(2026, 9, 22, 22, 0, tzinfo=UTC)] == 15.0
    # hour 14 local -> 12:00Z
    assert temps[datetime(2026, 9, 23, 12, 0, tzinfo=UTC)] == 28.0
    # hours 12 and 13 are empty strings: dropped, never 0
    assert datetime(2026, 9, 23, 10, 0, tzinfo=UTC) not in temps
    assert datetime(2026, 9, 23, 11, 0, tzinfo=UTC) not in temps


def test_parse_forecast_indexes_hours_without_metadata(fixture_bytes):
    payload = json.loads(fixture_bytes.decode("iso-8859-15"))

    _, temps = parse_forecast(payload)

    # second day has no "hora": values are 10..33 by index, hour = index
    assert temps[datetime(2026, 9, 24, 3, 0, tzinfo=UTC)] == 15.0  # local 05:00, value 15
    assert temps[datetime(2026, 9, 24, 21, 0, tzinfo=UTC)] == 33.0  # local 23:00, value 33
    assert len(temps) == 24 + 24 - 2  # two blank values dropped


def test_parse_forecast_accepts_periodo_value_variant():
    payload = [
        {
            "elaborado": "2026-09-23T06:00:00",
            "prediccion": {
                "dia": [
                    {
                        "fecha": "2026-09-23T00:00:00",
                        "temperatura": [
                            {"periodo": "07", "value": "18"},
                            {"periodo": "08", "value": "19.5"},
                            {"periodo": "09", "value": ""},
                        ],
                    }
                ]
            },
        }
    ]

    _, temps = parse_forecast(payload)

    assert temps[datetime(2026, 9, 23, 5, 0, tzinfo=UTC)] == 18.0
    assert temps[datetime(2026, 9, 23, 6, 0, tzinfo=UTC)] == 19.5
    assert len(temps) == 2


def test_parse_forecast_on_garbage_returns_empty():
    assert parse_forecast(None) == (None, {})
    assert parse_forecast([]) == (None, {})
    assert parse_forecast(["nope"]) == (None, {})
    assert (
        parse_forecast([{"prediccion": {"dia": [{"fecha": "bad", "temperatura": {"dato": [1]}}]}}])[
            1
        ]
        == {}
    )


# ---------------------------------------------------------------------------
# Two-step flow
# ---------------------------------------------------------------------------


def _handler(calls: list[str], fixture_bytes: bytes, datos_failures: int = 0):
    remaining = {"failures": datos_failures}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "municipio/horaria" in str(request.url):
            assert request.headers["api_key"] == "test-key"
            return httpx.Response(
                200,
                json={
                    "descripcion": "Éxito",
                    "estado": 200,
                    "datos": "https://opendata.aemet.es/datos/28079",
                    "metadatos": "https://opendata.aemet.es/metadatos/28079",
                },
            )
        if remaining["failures"]:
            remaining["failures"] -= 1
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, content=fixture_bytes, headers={"content-type": "text/plain"})

    return handler


async def test_forecast_retries_then_decodes_iso_8859_15(fixture_bytes):
    calls: list[str] = []
    service = AemetService(
        _settings(), transport=httpx.MockTransport(_handler(calls, fixture_bytes, datos_failures=2))
    )
    try:
        forecast = await service.forecast("Almería")
    finally:
        await service.aclose()

    assert forecast.available is True
    assert forecast.error is None
    assert forecast.temps[datetime(2026, 9, 22, 22, 0, tzinfo=UTC)] == 15.0
    assert len(calls) == 4  # 1 metadata + 2 failed datos + 1 success


async def test_forecast_uses_the_municipality_code(fixture_bytes):
    calls: list[str] = []
    service = AemetService(
        _settings(), transport=httpx.MockTransport(_handler(calls, fixture_bytes))
    )
    try:
        await service.forecast("Madrid")
    finally:
        await service.aclose()

    assert AEMET_INE_CODES["Madrid"] == "28079"
    assert calls[0].endswith("/prediccion/especifica/municipio/horaria/28079")


async def test_forecast_is_cached(fixture_bytes):
    calls: list[str] = []
    service = AemetService(
        _settings(), transport=httpx.MockTransport(_handler(calls, fixture_bytes))
    )
    try:
        first = await service.forecast("Madrid")
        second = await service.forecast("Madrid")
    finally:
        await service.aclose()

    assert first.available and second.available
    assert len(calls) == 2  # metadata + datos, once


async def test_forecast_without_api_key_does_not_call_the_api():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("AEMET must not be called without a key")

    service = AemetService(_settings(aemet_api_key=None), transport=httpx.MockTransport(handler))
    try:
        forecast = await service.forecast("Madrid")
    finally:
        await service.aclose()

    assert forecast.available is False
    assert forecast.error == "AEMET_API_KEY is not configured"
    assert forecast.temps == {}
    assert service.configured is False


async def test_forecast_unknown_city_reports_it():
    service = AemetService(_settings())
    try:
        forecast = await service.forecast("Atlantis")
    finally:
        await service.aclose()

    assert forecast.available is False
    assert "Atlantis" in forecast.error


async def test_forecast_reports_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"estado": 429, "descripcion": "Too many requests"})

    service = AemetService(_settings(), transport=httpx.MockTransport(handler))
    try:
        forecast = await service.forecast("Madrid")
    finally:
        await service.aclose()

    assert forecast.available is False
    assert "429" in forecast.error
    assert forecast.temps == {}


async def test_forecast_gives_up_after_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        if "municipio/horaria" in str(request.url):
            return httpx.Response(200, json={"estado": 200, "datos": "https://x/datos"})
        return httpx.Response(200, content=b"")

    service = AemetService(_settings(aemet_retries=2), transport=httpx.MockTransport(handler))
    try:
        forecast = await service.forecast("Madrid")
    finally:
        await service.aclose()

    assert forecast.available is False
    assert "2 attempts" in forecast.error


async def test_forecast_reports_empty_payload_as_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if "municipio/horaria" in str(request.url):
            return httpx.Response(200, json={"estado": 200, "datos": "https://x/datos"})
        return httpx.Response(200, content=b"[]", headers={"content-type": "text/plain"})

    service = AemetService(_settings(), transport=httpx.MockTransport(handler))
    try:
        forecast = await service.forecast("Madrid")
    finally:
        await service.aclose()

    assert forecast.available is False
    assert "no hourly temperatures" in forecast.error
