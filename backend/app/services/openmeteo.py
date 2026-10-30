"""Open-Meteo client for the on-demand refresh endpoint.

Fetches exactly what ``scripts/ingest_weather.py`` fetches (same URL, same
params, same ``CURRENT_VARIABLES``) and builds the same ``raw_weather``
document, so the endpoint and the scheduled ingestion stay interchangeable.

Only ``payload.current`` is requested: that is all the Spark batch job reads,
and ``raw_weather`` is the largest collection on a memory-constrained Atlas
tier. See the script's docstring for the full rationale.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import Request

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
#: Bounded so one slow upstream cannot pin a request (and its rate-limit slot).
OPEN_METEO_TIMEOUT_SECONDS = 15.0

#: Identical to ``scripts/ingest_weather.CURRENT_VARIABLES`` (asserted by a
#: test): the backend must request the same current-conditions variables the
#: writer stores.
CURRENT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "is_day",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "rain",
    "showers",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "snowfall",
    "visibility",
    "shortwave_radiation",
    "dew_point_2m",
    "uv_index",
    "cape",
    "wind_speed_80m",
    "wind_direction_80m",
]


class OpenMeteoError(RuntimeError):
    """The upstream call failed or did not return a usable JSON object."""


def build_document(data: dict[str, Any]) -> dict[str, Any] | None:
    """Open-Meteo payload -> ``raw_weather`` document, or None if unusable.

    Deliberately identical to ``scripts/ingest_weather.build_document``: the
    ``_id`` is ``{city}_{observation epoch}`` so a re-fetch of the same
    observation upserts instead of appending a duplicate.
    """
    current = data.get("current") or {}
    unix_time = current.get("time")
    if not isinstance(unix_time, (int, float)) or isinstance(unix_time, bool):
        return None

    timestamp = datetime.fromtimestamp(int(unix_time), tz=timezone.utc)
    return {
        "_id": f"{data['city']}_{int(timestamp.timestamp())}",
        "city": data["city"],
        "fetched_at": datetime.now(tz=timezone.utc),
        "timestamp": timestamp,
        # The whole response, untouched apart from the city tag.
        "payload": data,
    }


class OpenMeteoService:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # ``transport`` is the test seam: httpx.MockTransport keeps the suite offline.
        self._client = httpx.AsyncClient(
            timeout=OPEN_METEO_TIMEOUT_SECONDS,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch_current(self, city: str, latitude: float, longitude: float) -> dict[str, Any]:
        """One Open-Meteo call, tagged with the city name. Raises ``OpenMeteoError``."""
        try:
            response = await self._client.get(
                OPEN_METEO_URL,
                params={
                    "wind_speed_unit": "ms",
                    "timeformat": "unixtime",
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": CURRENT_VARIABLES,
                    "timezone": "auto",
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OpenMeteoError(f"Open-Meteo request failed: {exc}") from exc

        if not isinstance(payload, dict):
            raise OpenMeteoError("Open-Meteo response was not a JSON object")
        payload["city"] = city
        return payload


def get_openmeteo_service(request: Request) -> OpenMeteoService:
    return request.app.state.openmeteo
