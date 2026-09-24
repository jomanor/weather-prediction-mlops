"""AEMET OpenData client for the hourly municipal forecast.

Real two-step flow, as documented by AEMET:

1. ``GET {base}/prediccion/especifica/municipio/horaria/{ine_code}`` with the
   ``api_key`` header. The body is ``{"datos": "<url>", "metadatos": "<url>"}``
   (plus ``estado`` / ``descripcion``).
2. ``GET <datos url>`` — the signed URL frequently answers 404 or an empty body
   on the first attempt, so we retry with backoff. The payload is served as
   ``text/plain`` and encoded in ISO-8859-15, so it is decoded explicitly
   instead of trusting ``response.json()``.

Successful forecasts are cached in-process for ``aemet_cache_ttl_seconds``.
Failures are never cached and never turned into numbers: the caller gets
``available=False`` plus an explanation.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from datetime import time as dtime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import Request

from app.core.coerce import to_float, to_int
from app.core.config import Settings

logger = logging.getLogger(__name__)

#: INE municipality codes for the 14 producer cities.
AEMET_INE_CODES: dict[str, str] = {
    "Madrid": "28079",
    "Barcelona": "08019",
    "Valencia": "46250",
    "Sevilla": "41091",
    "Zaragoza": "50297",
    "Malaga": "29067",
    "Murcia": "30030",
    "Palma": "07040",
    "Bilbao": "48020",
    "Alicante": "03014",
    "Granada": "18087",
    "Almería": "04013",
    "Paterna": "46190",
    "El Ejido": "04079",
}

#: AEMET municipal forecasts are published in official Spanish local time.
AEMET_TZ = ZoneInfo("Europe/Madrid")


@dataclass(slots=True)
class AemetForecast:
    available: bool
    error: str | None = None
    issued_at: datetime | None = None
    temps: dict[datetime, float] = field(default_factory=dict)


def _parse_period_start(value: Any) -> int | None:
    """``"07"``, ``"07-13"`` or ``7`` -> 7."""
    if isinstance(value, str):
        head = value.strip().split("-")[0]
        return to_int(head)
    return to_int(value)


def _decode(response: httpx.Response) -> str:
    """AEMET serves ISO-8859-15 as text/plain; decode it explicitly."""
    raw = response.content
    for encoding in ("iso-8859-15", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("iso-8859-15", errors="replace")


def _hour_labels(block: dict[str, Any], count: int) -> list[int | None]:
    """Hour index for each value: ``hora`` / ``periodo`` metadata, else 0..23."""
    horas = block.get("hora")
    if isinstance(horas, list) and len(horas) == count:
        return [_parse_period_start(hour) for hour in horas]

    periodos = block.get("periodo")
    if isinstance(periodos, list) and len(periodos) == count:
        return [_parse_period_start(periodo) for periodo in periodos]

    return list(range(count))


def _extract_temperatures(block: Any) -> tuple[list[Any], list[int | None]]:
    """Accept both payload variants: ``{"dato": [...], "hora": [...]}`` and
    ``[{"periodo": 7, "value": "18"}, ...]``."""
    if isinstance(block, dict):
        values = block.get("dato")
        if not isinstance(values, list):
            return [], []
        return values, _hour_labels(block, len(values))

    if isinstance(block, list):
        values = [item.get("value") for item in block if isinstance(item, dict)]
        hours = [
            _parse_period_start(item.get("periodo")) for item in block if isinstance(item, dict)
        ]
        return values, hours

    return [], []


def _parse_local_timestamp(value: Any) -> datetime | None:
    """AEMET timestamps are local Spanish time without an offset."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=AEMET_TZ)
    return parsed.astimezone(timezone.utc)


def parse_forecast(payload: Any) -> tuple[datetime | None, dict[datetime, float]]:
    """Parse a ``datos`` payload into ``(issued_at, {utc_timestamp: °C})``.

    Empty strings and non-numeric values are dropped, never coerced to zero.
    """
    if not isinstance(payload, list) or not payload:
        return None, {}

    first = payload[0]
    if not isinstance(first, dict):
        return None, {}

    issued_at = _parse_local_timestamp(first.get("elaborado"))

    temps: dict[datetime, float] = {}
    days = (first.get("prediccion") or {}).get("dia") or []
    for day in days:
        if not isinstance(day, dict):
            continue
        fecha = day.get("fecha")
        if not fecha:
            continue
        try:
            day_date = date.fromisoformat(str(fecha)[:10])
        except ValueError:
            continue

        values, hours = _extract_temperatures(day.get("temperatura"))
        for value, hour in zip(values, hours):
            if hour is None or not 0 <= hour <= 23:
                continue
            temperature = to_float(value)
            if temperature is None:
                continue
            local = datetime.combine(day_date, dtime(hour=hour), tzinfo=AEMET_TZ)
            temps[local.astimezone(timezone.utc)] = temperature

    return issued_at, temps


class AemetService:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = settings.aemet_api_key
        self._base_url = settings.aemet_base_url.rstrip("/")
        self._retries = max(1, settings.aemet_retries)
        self._backoff = max(0.0, settings.aemet_backoff_seconds)
        self._cache_ttl = settings.aemet_cache_ttl_seconds
        self._cache: dict[str, tuple[float, AemetForecast]] = {}
        self._client = httpx.AsyncClient(
            timeout=settings.aemet_timeout_seconds,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def aclose(self) -> None:
        await self._client.aclose()

    def clear_cache(self) -> None:
        self._cache.clear()

    async def forecast(self, city: str) -> AemetForecast:
        """Hourly forecast for *city* (cached). Never raises."""
        if not self._api_key:
            return AemetForecast(available=False, error="AEMET_API_KEY is not configured")

        code = AEMET_INE_CODES.get(city)
        if code is None:
            return AemetForecast(
                available=False, error=f"No AEMET municipality code known for city '{city}'"
            )

        cached = self._cache.get(city)
        if cached is not None and (time.monotonic() - cached[0]) < self._cache_ttl:
            return cached[1]

        try:
            forecast = await self._fetch(code)
        except Exception as exc:  # noqa: BLE001 - surfaced as `error`, never fabricated
            logger.warning("AEMET forecast failed for %s: %s", city, exc)
            return AemetForecast(available=False, error=f"AEMET request failed: {exc}")

        self._cache[city] = (time.monotonic(), forecast)
        return forecast

    async def _fetch(self, code: str) -> AemetForecast:
        url = f"{self._base_url}/prediccion/especifica/municipio/horaria/{code}"
        response = await self._client.get(url, headers={"api_key": self._api_key or ""})
        response.raise_for_status()

        metadata = response.json()
        estado = metadata.get("estado")
        if estado != 200:
            raise RuntimeError(f"estado={estado}: {metadata.get('descripcion')}")

        datos_url = metadata.get("datos")
        if not datos_url:
            raise RuntimeError("response did not include a 'datos' URL")

        payload = await self._fetch_datos(datos_url)
        issued_at, temps = parse_forecast(payload)
        if not temps:
            raise RuntimeError("payload contained no hourly temperatures")

        return AemetForecast(available=True, issued_at=issued_at, temps=temps)

    async def _fetch_datos(self, url: str) -> Any:
        """The signed datos URL is flaky on first hit: retry with backoff."""
        last_error = "no attempt made"
        for attempt in range(self._retries):
            if attempt:
                await asyncio.sleep(self._backoff * 2 ** (attempt - 1))
            try:
                response = await self._client.get(url, headers={"Accept": "text/plain"})
            except httpx.HTTPError as exc:
                last_error = str(exc)
                continue

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                continue
            if not response.content.strip():
                last_error = "empty body"
                continue
            try:
                return json.loads(_decode(response))
            except ValueError as exc:
                last_error = f"invalid JSON: {exc}"

        raise RuntimeError(f"datos fetch failed after {self._retries} attempts ({last_error})")


def get_aemet_service(request: Request) -> AemetService:
    return request.app.state.aemet
