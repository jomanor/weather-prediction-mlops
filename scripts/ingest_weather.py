"""Ingest current observations from Open-Meteo into ``raw_weather``.

Locally the data path is Kafka producer -> consumer -> ``raw_weather``. Neither
a broker nor a long-lived worker fits the free hosting tiers the app deploys
to, so this script stands in for both: it fetches the same Open-Meteo payload
and writes the same document the consumer writes, straight to Mongo.

Only ``payload.current`` is fetched. That is all the Spark batch job reads
(``payload.current.*`` plus ``payload.latitude`` / ``payload.longitude``); the
producer additionally ships 24h of hourly and six pressure levels, which made
each document roughly fifty times larger and would push ``raw_weather`` through
Atlas's 512 MB free tier within months.

The document shape below is deliberately identical to
``kafka/kafka-consumer/consumer.py`` so both writers stay interchangeable.

Usage::

    MONGO_URI="mongodb+srv://..." python scripts/ingest_weather.py
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import requests
from pymongo import MongoClient

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DB_NAME = "weather_db"
RAW_COLLECTION = "raw_weather"
CITIES_COLLECTION = "cities"
MAX_WORKERS = 5

#: Current-conditions variables: what the API maps for the UI plus the fields
#: the Spark batch job reads. Mirrors the producer's list.
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
    # Present in the historical backfill's `current` block, so keep fetching
    # them to hold the schema steady across both writers.
    "cape",
    "wind_speed_80m",
    "wind_direction_80m",
]


def mongo_uri() -> str | None:
    return os.getenv("MONGO_URI") or os.getenv("MONGO_URL")


def load_cities(db) -> list[tuple[str, float, float]]:
    """The station registry, which the API seeds on startup."""
    cities: list[tuple[str, float, float]] = []
    for doc in db[CITIES_COLLECTION].find({}, {"name": 1, "latitude": 1, "longitude": 1}):
        name = doc.get("name")
        latitude, longitude = doc.get("latitude"), doc.get("longitude")
        if name and latitude is not None and longitude is not None:
            cities.append((name, float(latitude), float(longitude)))
    return sorted(cities)


def fetch_current(session: requests.Session, city: str, latitude: float, longitude: float) -> dict:
    """One Open-Meteo call, tagged with the city name."""
    response = session.get(
        OPEN_METEO_URL,
        params={
            "wind_speed_unit": "ms",
            "timeformat": "unixtime",
            "latitude": latitude,
            "longitude": longitude,
            "current": CURRENT_VARIABLES,
            "timezone": "auto",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    payload["city"] = city
    return payload


def build_document(data: dict) -> dict | None:
    """Open-Meteo payload -> ``raw_weather`` document, or None if unusable."""
    current = data.get("current") or {}
    unix_time = current.get("time")
    if unix_time is None:
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


def ingest() -> int:
    uri = mongo_uri()
    if not uri:
        print("MONGO_URI is not set.", file=sys.stderr)
        return 1

    db = MongoClient(uri, serverSelectionTimeoutMS=15000)[DB_NAME]
    cities = load_cities(db)
    if not cities:
        print(
            f"No stations in {DB_NAME}.{CITIES_COLLECTION}. Start the API once so it "
            "seeds the registry, then re-run.",
            file=sys.stderr,
        )
        return 1

    timestamp = datetime.now(tz=timezone.utc)
    print(f"Ingesting {len(cities)} station(s) at {timestamp.isoformat()}...", flush=True)

    stored = 0
    failures: list[str] = []

    with requests.Session() as session, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_current, session, city, latitude, longitude): city
            for city, latitude, longitude in cities
        }
        documents = []
        for future, city in ((f, futures[f]) for f in futures):
            try:
                documents.append((city, build_document(future.result())))
            except Exception as error:  # noqa: BLE001 - report and keep going
                failures.append(f"{city}: {error}")

        for city, document in documents:
            if document is None:
                failures.append(f"{city}: response had no current.timestamp")
                continue
            try:
                db[RAW_COLLECTION].replace_one({"_id": document["_id"]}, document, upsert=True)
                stored += 1
            except Exception as error:  # noqa: BLE001
                failures.append(f"{city}: {error}")

    print(f"Stored {stored}/{len(cities)} observation(s) in {RAW_COLLECTION}.", flush=True)
    for failure in failures:
        print(f"  failed - {failure}", file=sys.stderr)

    # A partial outage should not fail the run; a total one must be visible.
    return 0 if stored else 1


if __name__ == "__main__":
    sys.exit(ingest())
