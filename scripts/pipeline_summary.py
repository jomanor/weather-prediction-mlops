"""One-screen, read-only data snapshot for the ML pipeline summary.

Prints a short markdown block (``weather_features`` freshness, the latest
``model_registry`` entry, the newest prediction age, the per-model top drifting
features, and any model whose test ``skill_score`` has fallen to (or below) the
persistence baseline) so a failed or partial run still leaves a diagnosable
trace in the job summary. Uses ``pymongo`` only; nothing here writes, and it
never raises — any error is printed as a single line so the caller can keep the
pipeline moving. Drift is advisory: it emits a ``::warning`` on stderr and never
changes the exit code.

Drift PSI is *reported*, not blanket-warned. ``diagnostics.drift_psi`` is a
train→test PSI over 10 equal-width bins (Contract 3). On the multimodal
``pressure`` feature (15 city modes at different altitudes, each narrow) that
metric reads ~0.51 for **every** model and horizon as the temporal split crosses
the summer→September seasonal pressure rise — measured 2026-09-28, spread
0.505–0.518 across 11 models × 5 horizons, while temperature is ~0.26 and every
other feature is <0.19. A ``maxDriftPsi`` ceiling cannot separate that seasonal
shift from a genuine break, so the actionable ``::warning`` keys on skill
degradation instead: a model no better than "predict today".
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone

from pymongo import MongoClient

DB_NAME = "weather_db"

#: M4: a model's test ``metrics.skill_score`` at or below this value is flagged.
#: ``skill_score = 1 - model/baseline`` (persistence for both temperature and
#: rain), so ``<= 0.0`` means the model is no better than the "predict the
#: current observation" baseline — always actionable (roll back / retrain).
#: This replaces the old ``maxDriftPsi > 0.25`` blanket warning, which fired on
#: every run because the equal-width-binned ``pressure`` PSI saturates at ~0.51
#: for every model and horizon as the split crosses the summer→September
#: seasonal pressure rise (measured 2026-09-28, spread 0.505–0.518 across 11
#: models × 5 horizons; temperature ~0.26, all other features <0.19).
SKILL_SCORE_WARN_THRESHOLD = 0.0
#: How many top per-feature ``drift_psi`` values to list per model. Informational
#: only — these never raise the warning.
TOP_DRIFT_FEATURES = 3
#: Cap on registry docs scanned when resolving the current model per family.
DRIFT_MODEL_LIMIT = 200
#: Prediction coverage below this fraction of the cities that have features is
#: reported as a ``::warning`` (never a failure). Mirrors
#: ``inference.COVERAGE_WARN_RATIO``; by the time the summary runs the write has
#: already happened, so this is the only signal that a run scored only part of
#: the fleet (2026-10-30: 1 of 15 cities, silently).
PREDICTION_COVERAGE_WARN_RATIO = 0.9


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


def _finite_drift(model) -> dict[str, float] | None:
    """Finite ``diagnostics.drift_psi`` values, or ``None``.

    Legacy registry docs (no ``diagnostics``) and non-finite/absent PSI values
    are skipped, never an error — the summary must not fail on old data.
    """
    diagnostics = model.get("diagnostics") if isinstance(model, dict) else None
    if not isinstance(diagnostics, dict):
        return None
    drift = diagnostics.get("drift_psi")
    if not isinstance(drift, dict):
        return None
    finite = {
        key: value
        for key, value in drift.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    }
    return finite or None


def max_drift_psi(model) -> float | None:
    """Max finite ``diagnostics.drift_psi`` value, or ``None``."""
    drift = _finite_drift(model)
    return max(drift.values()) if drift else None


def top_drift_features(model, n: int = TOP_DRIFT_FEATURES) -> list[tuple[str, float]]:
    """Top *n* ``(feature, psi)`` pairs by PSI, descending.

    Informational only: listed in the summary, never used to raise a warning
    (see the module docstring for why a PSI ceiling is not discriminating).
    """
    drift = _finite_drift(model)
    if not drift:
        return []
    ordered = sorted(drift.items(), key=lambda item: item[1], reverse=True)
    return ordered[:n]


def skill_score(model) -> float | None:
    """Finite test ``metrics.skill_score``, or ``None`` when absent/non-finite."""
    metrics = model.get("metrics") if isinstance(model, dict) else None
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("skill_score")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return float(value)


def _model_label(model: dict) -> str:
    """Human label ``temperature 24h (LinearRegression)`` when ``target`` is
    present, else the legacy ``model_type horizon model_name`` form."""
    target = model.get("target")
    horizon = model.get("horizon_hours", "?")
    model_type = model.get("model_type", "?")
    if target:
        return f"{target} {horizon}h ({model_type})"
    name = model.get("model_name", "unknown")
    return f"{model_type} {horizon}h {name}"


def drift_report(models) -> tuple[list[str], str | None]:
    """Summarize registry drift and skill degradation.

    Returns ``(summary_lines, warning)`` where ``warning`` is the body of a
    ``::warning`` to print on stderr (``None`` when nothing is actionable).
    *models* is an iterable of registry documents.

    PSI is reported per model (top features), never warned: the equal-width
    ``pressure`` PSI sits at ~0.51 for every model, so a ceiling cannot
    discriminate. The ``::warning`` fires only when a model's test
    ``skill_score`` has fallen to (or below) the persistence baseline — the one
    condition that maps to a concrete action.
    """
    scored = []
    degraded = []
    for model in models:
        if not isinstance(model, dict):
            continue
        top = top_drift_features(model)
        if top:
            scored.append((model, top))
        skill = skill_score(model)
        if skill is not None and skill <= SKILL_SCORE_WARN_THRESHOLD:
            degraded.append((model, skill))

    lines = [f"- `model_registry` drift: {len(scored)} model(s) scored"]
    for model, top in sorted(scored, key=lambda pair: _model_label(pair[0])):
        listing = ", ".join(f"{feature}={psi:.3f}" for feature, psi in top)
        lines.append(f"  - {_model_label(model)}: {listing}")

    warning = None
    if degraded:
        ordered = sorted(degraded, key=lambda pair: pair[1])
        listing = "; ".join(
            f"{_model_label(model)} skill_score={skill:.3f}" for model, skill in ordered
        )
        lines.append(
            f"- `model_registry` skill warning: {len(ordered)} model(s) at or below "
            "the persistence baseline"
        )
        warning = (
            f"model skill_score <= {SKILL_SCORE_WARN_THRESHOLD:.0f} "
            f"(no better than persistence baseline): {listing}"
        )
    return lines, warning


def prediction_horizon_stats(db) -> dict[int, dict]:
    """Per-horizon prediction rows/cities, plus the newest-batch city count.

    One aggregation, no sort (hard rule: Mongo sorts are city-prefixed) and
    ``allowDiskUse=True``. Grouping by ``(horizon, city)`` bounds the result at
    #horizons × #cities documents, so the roll-up below is trivial.

    ``cities_at_newest`` is how many cities have a prediction for that horizon's
    newest ``source_timestamp`` — the number that collapsed to 1 during the
    2026-10-30 incident, which a lifetime city count would have hidden.

    Any failure returns ``{}``: the summary must never fail the pipeline.
    """
    pipeline = [
        {
            "$group": {
                "_id": {"horizon": "$horizon_hours", "city": "$city"},
                "rows": {"$sum": 1},
                "oldest": {"$min": "$source_timestamp"},
                "newest": {"$max": "$source_timestamp"},
            }
        }
    ]
    try:
        groups = list(db["weather_predictions"].aggregate(pipeline, allowDiskUse=True))
    except Exception:  # noqa: BLE001 - a summary must never fail the pipeline
        return {}

    stats: dict[int, dict] = {}
    per_horizon_newest: dict[int, list[tuple]] = {}
    for group in groups:
        key = group.get("_id") or {}
        horizon = key.get("horizon")
        if horizon is None:
            continue  # legacy row without horizon_hours: not attributable
        horizon = int(horizon)
        entry = stats.setdefault(horizon, {"rows": 0, "cities": 0, "oldest": None, "newest": None})
        entry["rows"] += int(group.get("rows") or 0)
        entry["cities"] += 1
        oldest = _as_utc(group.get("oldest"))
        newest = _as_utc(group.get("newest"))
        if oldest and (entry["oldest"] is None or oldest < entry["oldest"]):
            entry["oldest"] = oldest
        if newest and (entry["newest"] is None or newest > entry["newest"]):
            entry["newest"] = newest
        per_horizon_newest.setdefault(horizon, []).append((key.get("city"), newest))

    for horizon, entry in stats.items():
        entry["cities_at_newest"] = sum(
            1
            for _city, newest in per_horizon_newest[horizon]
            if newest is not None and newest == entry["newest"]
        )
    return stats


def available_cities(db) -> int:
    """Distinct cities with stored features — the population inference could score."""
    try:
        return len([city for city in db["weather_features"].distinct("city") if city])
    except Exception:  # noqa: BLE001 - a summary must never fail the pipeline
        return 0


def coverage_warning(scored_cities: int, available: int) -> str | None:
    """``::warning`` body when *scored_cities* is under 90 % of *available*.

    Pure. Returns ``None`` when coverage is fine or *available* is unknown (0),
    so an empty/fresh ``weather_features`` cannot produce a false alarm.
    """
    if not available or scored_cities >= PREDICTION_COVERAGE_WARN_RATIO * available:
        return None
    share = 100.0 * scored_cities / available
    return (
        f"prediction coverage {scored_cities}/{available} cities ({share:.0f}%) at the "
        f"newest source_timestamp is below {PREDICTION_COVERAGE_WARN_RATIO * 100:.0f}%"
    )


def prediction_coverage_lines(stats: dict[int, dict], available: int) -> list[str]:
    """Summary lines: rows/cities per horizon, oldest source timestamp, coverage."""
    if not stats:
        return ["- `weather_predictions` coverage: no prediction rows"]

    per_horizon = "; ".join(
        f"{horizon}h {entry['rows']} rows / {entry['cities']} cities"
        for horizon, entry in sorted(stats.items())
    )
    oldest_values = [entry["oldest"] for entry in stats.values() if entry["oldest"]]
    oldest = min(oldest_values) if oldest_values else None
    scored = min(entry["cities_at_newest"] for entry in stats.values())
    share = f" ({100.0 * scored / available:.0f}%)" if available else ""
    return [
        f"- `weather_predictions` per horizon: {per_horizon}",
        f"- `weather_predictions` oldest source_timestamp {_iso(oldest)} (age {_age(oldest)})",
        f"- `weather_predictions` coverage: {scored}/{available or '?'} cities{share} "
        "at the newest source_timestamp",
    ]


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
    """All summary lines, plus the skill-degradation ``::warning`` body."""
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


def summary_and_warnings(db) -> tuple[list[str], list[tuple[str, str]]]:
    """All summary lines plus every ``(title, body)`` annotation to emit.

    Existing sections are unchanged; the prediction-coverage block follows the
    prediction line, and its ``::warning`` (below 90 % city coverage) joins the
    skill-degradation one. Neither ever fails the run.
    """
    lines, skill_warning = summary_and_warning(db)
    stats = prediction_horizon_stats(db)
    available = available_cities(db)
    lines.extend(prediction_coverage_lines(stats, available))

    warnings: list[tuple[str, str]] = []
    if skill_warning:
        warnings.append(("Model skill degradation", skill_warning))
    if stats:
        coverage = coverage_warning(min(e["cities_at_newest"] for e in stats.values()), available)
        if coverage:
            warnings.append(("Prediction coverage", coverage))
    return lines, warnings


def summary_lines(db) -> list[str]:
    """Backwards-compatible summary-only view (warnings go to stderr)."""
    return summary_and_warning(db)[0]


def main() -> int:
    warnings: list[tuple[str, str]] = []
    try:
        uri = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
        if not uri:
            print("- pipeline summary: MONGO_URI not set")
            return 0
        client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
        try:
            lines, warnings = summary_and_warnings(client[DB_NAME])
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 - a summary must never fail the pipeline
        print(f"- pipeline summary error: {type(exc).__name__}: {exc}")
        return 0

    print("### Pipeline data snapshot")
    for line in lines:
        print(line)
    for title, body in warnings:
        # stderr, not stdout: the workflow redirects stdout to $GITHUB_STEP_SUMMARY,
        # and only the step log surfaces ::warning as an annotation.
        print(f"::warning title={title}::{body}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
