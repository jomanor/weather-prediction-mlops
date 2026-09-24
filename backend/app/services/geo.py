"""Open-Meteo geocoding client (no API key).

Thin proxy over ``https://geocoding-api.open-meteo.com/v1/search`` used by the
station picker. Results are returned exactly as the upstream orders them and
never fabricated: an unreachable or failing upstream raises ``GeocodingError``
so the route can surface a 502.
"""

import logging
from typing import Any

import httpx
from fastapi import Request

from app.core.coerce import to_float
from app.core.config import Settings
from app.schemas.city import GeoResult

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


class GeocodingError(RuntimeError):
    """The upstream geocoding API could not be reached or parsed."""


def _text(value: Any) -> str | None:
    """Non-empty string or None; anything else is missing."""
    return value if isinstance(value, str) and value else None


def _map_result(item: dict[str, Any]) -> GeoResult | None:
    """One upstream result -> GeoResult, or None when name/coords are missing."""
    name = _text(item.get("name"))
    latitude = to_float(item.get("latitude"))
    longitude = to_float(item.get("longitude"))
    if name is None or latitude is None or longitude is None:
        return None
    return GeoResult(
        name=name,
        latitude=latitude,
        longitude=longitude,
        country=_text(item.get("country")),
        admin1=_text(item.get("admin1")),
    )


class GeocodingService:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            timeout=settings.geo_timeout_seconds,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str) -> list[GeoResult]:
        """Geocode *query*. Raises ``GeocodingError`` on upstream failure."""
        query = query.strip()
        if not query:
            return []

        try:
            response = await self._client.get(
                GEOCODING_URL,
                params={"name": query, "count": 8, "language": "es", "format": "json"},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Geocoding request failed for %r: %s", query, exc)
            raise GeocodingError(f"Geocoding request failed: {exc}") from exc

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            return []
        return [
            result
            for result in (_map_result(item) for item in results if isinstance(item, dict))
            if result is not None
        ]


def get_geo_service(request: Request) -> GeocodingService:
    return request.app.state.geo
