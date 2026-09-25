"""One-screen, read-only data snapshot for the ML pipeline summary.

Prints a short markdown block (``weather_features`` freshness, the latest
``model_registry`` entry, and the newest prediction age) so a failed or partial
run still leaves a diagnosable trace in the job summary. Uses ``pymongo`` only;
nothing here writes, and it never raises — any error is printed as a single
line so the caller can keep the pipeline moving.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from pymongo import MongoClient

DB_NAME = "weather_db"


def _as_utc(value):
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value) -> str:
    moment = _as_utc(value)
    return moment.isoformat().replace("+00:00", "Z") if moment else "n/a"


def _age(value) -> str:
    moment = _as_utc(value)
    if moment is None:
        return "n/a"
    hours = (datetime.now(timezone.utc) - moment).total_seconds() / 3600.0
    return f"{hours:.1f} h"


def summary_lines(db) -> list[str]:
    features = db["weather_features"]
    count = features.count_documents({})
    latest = features.find_one({}, sort=[("timestamp", -1)], projection={"timestamp": 1}) or {}
    lines = [
        f"- `weather_features`: {count} rows, latest {_iso(latest.get('timestamp'))} "
        f"(age {_age(latest.get('timestamp'))})"
    ]

    model = db["model_registry"].find_one({}, sort=[("timestamp", -1)])
    if model:
        lines.append(
            "- `model_registry` (latest): "
            f"`{model.get('model_name', 'unknown')}` @ {_iso(model.get('timestamp'))} | "
            f"interval={'yes' if model.get('interval') else 'no'} | "
            f"split={'yes' if model.get('split') else 'no'} | "
            f"commit={model.get('commit', 'unknown')}"
        )
    else:
        lines.append("- `model_registry`: no documents")

    newest = (
        db["weather_predictions"].find_one(
            {}, sort=[("source_timestamp", -1)], projection={"source_timestamp": 1}
        )
        or {}
    )
    lines.append(
        f"- `weather_predictions`: newest source_timestamp "
        f"{_iso(newest.get('source_timestamp'))} (age {_age(newest.get('source_timestamp'))})"
    )
    return lines


def main() -> int:
    try:
        uri = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
        if not uri:
            print("- pipeline summary: MONGO_URI not set")
            return 0
        client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
        try:
            lines = summary_lines(client[DB_NAME])
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 - a summary must never fail the pipeline
        print(f"- pipeline summary error: {type(exc).__name__}: {exc}")
        return 0

    print("### Pipeline data snapshot")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
