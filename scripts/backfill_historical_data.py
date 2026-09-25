"""
backfill_historical_data.py
============================
Standalone script that backfills historical weather data from the
Open-Meteo Historical API directly into MongoDB (raw_weather collection),
bypassing Kafka.

Usage
-----
    python scripts/backfill_historical_data.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
    python scripts/backfill_historical_data.py --days 90 --recent 5

Options
-------
    --days   : backfill the last N days (overrides --start); the range the app
               queries is only the last few days, so 90 keeps the free tier far
               below its 512 MB limit while still giving training a long series.
    --recent : additionally fill the last N days from the forecast API. The
               archive (ERA5) lags real time by a few days, so without this the
               most recent days — the ones the UI charts — stay missing.

Defaults
--------
    --start : 4 years before today
    --end   : yesterday (the Historical API does not serve today)

Cities
------
    Read from the ``cities`` registry when present so the station set matches
    the deployed app; otherwise the bundled list is used.

Idempotency
-----------
    Before inserting a date-city pair the script checks whether a document
    with that _id already exists in raw_weather and skips it.  Running the
    script multiple times is safe.

Environment
-----------
    MONGO_URL  mongodb connection string (same as the rest of the stack)
"""

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import requests
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
#: Serves the most recent days, which the archive API does not cover yet.
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_DELAY_SECONDS = 0.5  # Be polite to the free API
BATCH_UPSERT_SIZE = 100  # MongoDB upsert batch size

CITIES = {
    "El Ejido": (36.7756, -2.8144),
    "Almería": (36.8381, -2.4597),
    "Granada": (37.1773, -3.5986),
    "Paterna": (39.5028, -0.4408),
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3851, 2.1734),
    "Valencia": (39.4699, -0.3763),
    "Sevilla": (37.3891, -5.9845),
    "Zaragoza": (41.6488, -0.8891),
    "Malaga": (36.7213, -4.4214),
    "Murcia": (37.9922, -1.1307),
    "Palma": (39.5696, 2.6502),
    "Bilbao": (43.2630, -2.9350),
    "Alicante": (38.3452, -0.4810),
}

# Hourly variables requested from the historical archive
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "cape",
    "wind_speed_80m",
    "wind_direction_80m",
    "apparent_temperature",
]

# Pressure-level variables (upper-air)
PRESSURE_LEVEL_VARIABLES = [
    "geopotential_height",
    "temperature",
    "relative_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

PRESSURE_LEVELS = [200, 500, 700, 850, 925, 1000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_historical_chunk(
    city: str, lat: float, lon: float, start: date, end: date
) -> dict | None:
    """
    Fetch one city's historical data for [start, end] from Open-Meteo.
    Returns the parsed JSON or None on error.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "pressure_level": ",".join(str(p) for p in PRESSURE_LEVELS),
        "hourly_pressure_level": ",".join(PRESSURE_LEVEL_VARIABLES),
        "wind_speed_unit": "ms",
        "timeformat": "unixtime",
        "timezone": "auto",
    }

    try:
        resp = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            data["city"] = city
            return data
        else:
            print(
                f"  [WARN] HTTP {resp.status_code} for {city} ({start} – {end}): {resp.text[:200]}"
            )
    except Exception as exc:
        print(f"  [ERROR] Request failed for {city}: {exc}")

    return None


def fetch_recent_chunk(city: str, lat: float, lon: float, past_days: int) -> dict | None:
    """
    Fetch the most recent *past_days* of hourly data from the forecast API,
    which extends closer to real time than the archive. Returns the parsed
    JSON (same ``hourly`` shape) or None on error.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": 1,
        "hourly": ",".join(HOURLY_VARIABLES),
        "wind_speed_unit": "ms",
        "timeformat": "unixtime",
        "timezone": "auto",
    }

    try:
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            data["city"] = city
            return data
        print(
            f"  [WARN] HTTP {resp.status_code} for {city} (recent {past_days}d): {resp.text[:200]}"
        )
    except Exception as exc:
        print(f"  [ERROR] Recent request failed for {city}: {exc}")

    return None


def load_cities(db) -> dict[str, tuple[float, float]]:
    """The station registry the API seeds, as {name: (lat, lon)}."""
    cities: dict[str, tuple[float, float]] = {}
    for doc in db["cities"].find({}, {"name": 1, "latitude": 1, "longitude": 1}):
        name, lat, lon = doc.get("name"), doc.get("latitude"), doc.get("longitude")
        if name and lat is not None and lon is not None:
            cities[name] = (float(lat), float(lon))
    return cities


def upsert_documents(collection, docs: list[dict]) -> tuple[int, int]:
    """Upsert *docs* in fixed-size batches. Returns (inserted, skipped)."""
    inserted = 0
    skipped = 0
    for i in range(0, len(docs), BATCH_UPSERT_SIZE):
        ins, skp = upsert_batch(collection, docs[i : i + BATCH_UPSERT_SIZE])
        inserted += ins
        skipped += skp
    return inserted, skipped


def already_exists(collection, city: str, unix_ts: int) -> bool:
    """Check whether a specific city+timestamp document is already in MongoDB."""
    return collection.count_documents({"_id": f"{city}_{unix_ts}"}, limit=1) > 0


def build_documents(city: str, payload: dict) -> list[dict]:
    """
    Expand an Open-Meteo historical response into one document per hour.
    Each document mirrors the same shape used by the live consumer so that
    batch_processing.py can read from a single collection.
    """
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        return []

    docs = []
    for i, unix_ts in enumerate(times):
        # Build a dict with all hourly scalar fields at this index
        hourly_point = {}
        for var, values in hourly.items():
            if var == "time":
                continue
            if isinstance(values, list) and i < len(values):
                hourly_point[var] = values[i]

        timestamp_dt = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
        doc_id = f"{city}_{int(unix_ts)}"

        doc = {
            "_id": doc_id,
            "city": city,
            "timestamp": timestamp_dt,
            "fetched_at": datetime.now(tz=timezone.utc),
            "source": "historical_backfill",
            "payload": {
                "city": city,
                "latitude": payload.get("latitude"),
                "longitude": payload.get("longitude"),
                "utc_offset_seconds": payload.get("utc_offset_seconds"),
                "timezone": payload.get("timezone"),
                # Store the per-hour snapshot under "current" so that
                # downstream code can treat it the same way as live data.
                "current": {
                    "time": unix_ts,
                    **hourly_point,
                },
                # Keep a reference to the pressure-level payload for the
                # upper-air features without duplicating the full timeseries.
                "hourly_meta": {k: v for k, v in payload.items() if k not in ("hourly", "city")},
            },
        }
        docs.append(doc)

    return docs


def upsert_batch(collection, docs: list[dict]) -> tuple[int, int]:
    """Bulk-upsert a list of documents. Returns (inserted, skipped)."""
    if not docs:
        return 0, 0

    ops = [UpdateOne({"_id": d["_id"]}, {"$setOnInsert": d}, upsert=True) for d in docs]

    inserted = 0
    skipped = 0
    try:
        result = collection.bulk_write(ops, ordered=False)
        inserted = result.upserted_count
        skipped = len(docs) - inserted
    except BulkWriteError as bwe:
        # Duplicate key errors are expected and safe to ignore
        inserted = bwe.details.get("nUpserted", 0)
        skipped = len(docs) - inserted

    return inserted, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    today = date.today()
    four_years_ago = date(today.year - 4, today.month, today.day)
    yesterday = today - timedelta(days=1)

    parser = argparse.ArgumentParser(description="Backfill historical weather data")
    parser.add_argument(
        "--start",
        default=four_years_ago.isoformat(),
        help="Start date YYYY-MM-DD (default: 4 years ago)",
    )
    parser.add_argument(
        "--end",
        default=yesterday.isoformat(),
        help="End date YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Backfill the last N days (overrides --start)",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=0,
        help="Also fill the last N days from the forecast API (default: 0)",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=180,
        help="Days per API request chunk (default: 180)",
    )
    default_mongo = os.getenv("MONGO_URI") or os.getenv(
        "MONGO_URL", "mongodb://admin:weatherpass123@localhost:27017/weather_db?authSource=admin"
    )
    if "@mongodb:27017" in default_mongo and not os.path.exists("/.dockerenv"):
        default_mongo = default_mongo.replace("@mongodb:27017", "@localhost:27017")

    parser.add_argument(
        "--mongo-url",
        default=default_mongo,
        help="MongoDB connection string (or set MONGO_URI env var)",
    )
    args = parser.parse_args()
    if args.days is not None:
        args.start = (today - timedelta(days=args.days)).isoformat()
    return args


def main():
    args = parse_args()

    if not args.mongo_url:
        print("[ERROR] MONGO_URI is not set. Pass --mongo-url or export the env var.")
        sys.exit(1)

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    if start_date > end_date:
        print(f"[ERROR] start ({start_date}) must be before end ({end_date})")
        sys.exit(1)

    client = MongoClient(args.mongo_url)
    db = client["weather_db"]
    collection = db["raw_weather"]

    # Match the deployed station registry when it exists; fall back to the
    # bundled list on a brand-new database the API has not seeded yet.
    cities = load_cities(db) or CITIES

    print(f"Backfill range : {start_date} → {end_date}")
    print(f"Recent fill    : {args.recent} day(s) from the forecast API")
    print(f"Cities         : {len(cities)}")
    print(f"Chunk size     : {args.chunk_days} days")
    print(f"MongoDB        : {args.mongo_url.split('@')[-1]}")  # hide credentials
    print()

    # Create a compound index to speed up existence checks
    collection.create_index([("city", 1), ("timestamp", -1)], background=True)

    total_inserted = 0
    total_skipped = 0

    for city, (lat, lon) in cities.items():
        print(f"── {city} ({lat}, {lon})")

        chunk_start = start_date
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=args.chunk_days - 1), end_date)

            print(f"   Fetching {chunk_start} → {chunk_end} … ", end="", flush=True)
            payload = fetch_historical_chunk(city, lat, lon, chunk_start, chunk_end)

            if payload is None:
                print("FAILED — skipping chunk")
                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            docs = build_documents(city, payload)
            print(f"{len(docs)} hourly records … ", end="", flush=True)

            city_inserted, city_skipped = upsert_documents(collection, docs)
            print(f"inserted={city_inserted}, skipped(already existed)={city_skipped}")
            total_inserted += city_inserted
            total_skipped += city_skipped

            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(REQUEST_DELAY_SECONDS)

    # The archive lags real time by a few days; fill the window the UI charts.
    if args.recent > 0:
        now = datetime.now(tz=timezone.utc)
        print(f"\n── Recent fill (last {args.recent} days, forecast API)")
        for city, (lat, lon) in cities.items():
            print(f"   {city} … ", end="", flush=True)
            payload = fetch_recent_chunk(city, lat, lon, args.recent)
            if payload is None:
                print("FAILED — skipping")
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            # The response also carries tomorrow's forecast; keep only what
            # has already happened.
            docs = [doc for doc in build_documents(city, payload) if doc["timestamp"] <= now]
            city_inserted, city_skipped = upsert_documents(collection, docs)
            print(f"inserted={city_inserted}, skipped={city_skipped}")
            total_inserted += city_inserted
            total_skipped += city_skipped
            time.sleep(REQUEST_DELAY_SECONDS)

    print()
    print("Backfill complete.")
    print(f"  Total inserted : {total_inserted}")
    print(f"  Total skipped  : {total_skipped}")

    client.close()


if __name__ == "__main__":
    main()
