"""One-screen, read-only data snapshot for the ML pipeline summary.

Prints a short markdown block (``weather_features`` freshness, the latest
``model_registry`` entry, the newest prediction age, and any M4 model drift
above PSI 0.25) so a failed or partial run still leaves a diagnosable trace in
the job summary. Uses ``pymongo`` only; nothing here writes, and it never raises
— any error is printed as a single line so the caller can keep the pipeline
moving. Drift is advisory: it emits a ``::warning`` on stderr and never changes
the exit code.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone

from pymongo import MongoClient

DB_NAME = "weather_db"

#: M4: a model's ``maxDriftPsi`` (max finite ``diagnostics.drift_psi`` value)
#: above this gets a ``::warning``. The threshold is advisory: drift never fails
#: the pipeline, it only annotates the run.
DRIFT_PSI_WARN_THRESHOLD = 0.25
#: Cap on registry docs scanned when resolving the current model per family.
DRIFT_MODEL_LIMIT = 200


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


def max_drift_psi(model) -> float | None:
    """Max finite ``diagnostics.drift_psi`` value, or ``None``.

    Legacy registry docs (no ``diagnostics``) and non-finite/absent PSI values
    are skipped, never an error — the summary must not fail on old data.
    """
    diagnostics = model.get("diagnostics") if isinstance(model, dict) else None
    if not isinstance(diagnostics, dict):
        return None
    drift = diagnostics.get("drift_psi")
    if not isinstance(drift, dict):
        return None
    finite = [
        value
        for value in drift.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    ]
    return max(finite) if finite else None


def _model_label(model: dict) -> str:
    model_type = model.get("model_type", "?")
    horizon = model.get("horizon_hours", "?")
    name = model.get("model_name", "unknown")
    return f"{model_type} {horizon}h {name}"


def drift_report(models) -> tuple[list[str], str | None]:
    """Summarize registry drift.

    Returns ``(summary_lines, warning)`` where ``warning`` is the body of a
    ``::warning`` to print on stderr (``None`` when nothing is over threshold).
    *models* is an iterable of registry documents; each model's ``maxDriftPsi``
    is the max over its finite ``diagnostics.drift_psi`` values.
    """
    scored = []
    for model in models:
        if not isinstance(model, dict):
            continue
        psi = max_drift_psi(model)
        if psi is not None:
            scored.append((model, psi))
    drifted = sorted(
        (pair for pair in scored if pair[1] > DRIFT_PSI_WARN_THRESHOLD),
        key=lambda pair: pair[1],
        reverse=True,
    )

    lines = [
        f"- `model_registry` drift: {len(scored)} model(s) scored, "
        f"{len(drifted)} above PSI {DRIFT_PSI_WARN_THRESHOLD:.2f}"
    ]
    warning = None
    if drifted:
        listing = "; ".join(
            f"{_model_label(model)} maxDriftPsi={psi:.3f}" for model, psi in drifted
        )
        lines.append(f"- `model_registry` drift warning: {listing}")
        warning = f"model drift PSI > {DRIFT_PSI_WARN_THRESHOLD:.2f}: {listing}"
    return lines, warning


def latest_registry_models(db) -> list[dict]:
    """Newest registry document per ``(model_type, horizon_hours)`` with drift data."""
    cursor = db["model_registry"].find(
        {"diagnostics.drift_psi": {"$exists": True}},
        sort=[("timestamp", -1)],
        limit=DRIFT_MODEL_LIMIT,
    )
    latest: dict[tuple, dict] = {}
    for doc in cursor:
        key = (doc.get("model_type"), doc.get("horizon_hours"))
        latest.setdefault(key, doc)
    return list(latest.values())


def summary_and_warning(db) -> tuple[list[str], str | None]:
    """All summary lines, plus the drift ``::warning`` body when over threshold."""
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

    drift_lines, warning = drift_report(latest_registry_models(db))
    lines.extend(drift_lines)
    return lines, warning


def summary_lines(db) -> list[str]:
    """Backwards-compatible summary-only view (the warning goes to stderr)."""
    return summary_and_warning(db)[0]


def main() -> int:
    warning = None
    try:
        uri = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
        if not uri:
            print("- pipeline summary: MONGO_URI not set")
            return 0
        client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
        try:
            lines, warning = summary_and_warning(client[DB_NAME])
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 - a summary must never fail the pipeline
        print(f"- pipeline summary error: {type(exc).__name__}: {exc}")
        return 0

    print("### Pipeline data snapshot")
    for line in lines:
        print(line)
    if warning:
        # stderr, not stdout: the workflow redirects stdout to $GITHUB_STEP_SUMMARY,
        # and only the step log surfaces ::warning as an annotation.
        print(f"::warning title=Model drift::{warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
