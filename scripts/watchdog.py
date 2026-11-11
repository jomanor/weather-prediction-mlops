"""Dead-man's switch for the ingest -> features -> predictions pipeline.

Repo-local replacement for the user-side Healthchecks.io ping. It reads the
**public read API only** (no Mongo credentials, no ``gh`` CLI) and turns stale
data into a single rolling GitHub issue in this repository, authenticated with
the automatic ``GITHUB_TOKEN``. Nothing here writes to the database or to the
pipeline: the only side effects are GitHub issue create/comment/close.

Signals (evidence gathered from ``/api/v1``):

1. **ingest/features staleness** -- age of the newest ``observed_at`` across
   stations in ``GET /weather/current``, plus ``GET /weather/quality?days=7``
   completeness of the worst city.
2. **inference staleness** -- age of the newest ``source_timestamp`` in
   ``GET /predictions/latest``, plus prediction coverage (distinct cities with a
   prediction / cities in ``/weather/current``).
3. **model degradation** -- entries in ``GET /models`` still in production whose
   skill score is at or below zero. **Advisory only**: two production models
   already sit below zero, so it is printed in the issue body but never triggers
   an issue or a comment on its own.

Exit codes -- a red run always means the *watchdog* is broken, never that the
pipeline is stale or the API is down:

* ``0`` = a verdict was produced and handled. That covers the healthy case, the
  alert case, and **an unreachable/erroring API**: a timeout, a connection fault
  or a 5xx/502 from the read API after the retries below is an *availability
  signal*, not a broken switch. It becomes the ``api_down`` verdict, opens or
  updates the same rolling issue (the error text is in the body, never
  swallowed) and the run still exits ``0``. A failing models endpoint is the same
  idea, scoped to the advisory: it records ``advisory_error`` and the graded
  signals still decide the verdict.
* ``1`` = the watchdog itself failed: an unreadable/absent ``GITHUB_TOKEN`` (or
  ``--repo``), a GitHub REST error, or an internal exception -- including a
  payload that breaks the documented API shape (that is a contract failure in
  this file, not an availability one). ``argparse`` exits ``2`` on a bad flag.

So a green run with an open issue is the pipeline alert; a red run is the
alarm about the alarm.

Stdlib only (``urllib.request``/``json``/``argparse``/``dataclasses``): this job
must be able to run before, and independently of, any dependency install. The one
repo import is ``app.core.thresholds`` -- plain constants, no imports of its own --
so sharing the numbers costs nothing at runtime (see the threshold block below).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# Thresholds -- imported from their ONE authority, ``backend/app/core/
# thresholds.py`` (plain constants, stdlib-importable), shared with the API,
# ``inference.py`` and ``pipeline_summary.py``. Nothing is mirrored here any
# more: to retune a gate, change that module and every reader follows (2026-10-31
# audit, finding 5: four copies, three different numbers). The age gates are
# calibrated to the ~6 h feature-rebuild cycle (~2x ok, ~3x warn = one missed
# rebuild) and that calibration belongs to the shared module too.
# ``scripts/`` runs from the repo root, so derive ``<repo_root>/backend`` from
# this file rather than from the cwd, and keep this import stdlib-only.
# ---------------------------------------------------------------------------
# ``SKILL_SCORE_ADVISORY_AT_OR_BELOW`` is advisory only (never a trigger):
# production models scoring at/below it are printed, not paged.
_BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.core.thresholds import (  # noqa: E402 - path is set just above
    PREDICTION_COVERAGE_WARN_RATIO,
    PREDICTION_OK_AGE_HOURS,
    PREDICTION_WARN_AGE_HOURS,
    QUALITY_BAD_COMPLETENESS_BELOW,
    QUALITY_OK_AGE_HOURS,
    QUALITY_WARN_AGE_HOURS,
    SKILL_SCORE_ADVISORY_AT_OR_BELOW,
)

#: Local aliases so the rest of this file keeps reading the same names. Both
#: *are* the shared values -- a retune upstream moves them, a stale copy cannot.
PREDICTION_BAD_AGE_HOURS = PREDICTION_WARN_AGE_HOURS
PREDICTION_BAD_COVERAGE_BELOW = PREDICTION_COVERAGE_WARN_RATIO

# ---------------------------------------------------------------------------
# Fixed identities -- one rolling issue, so the switch cannot fan out.
# ---------------------------------------------------------------------------
ISSUE_TITLE = "[ops] Pipeline de datos desactualizado"
ISSUE_LABEL = "ops-alert"
ISSUE_LABEL_COLOR = "b60205"
ISSUE_LABEL_DESCRIPTION = "Alerta automática del dead-man's switch del pipeline"

DEFAULT_API_BASE = "https://weather-prediction-mlops.onrender.com"
DEFAULT_GITHUB_API = "https://api.github.com"

#: Retry each read-API call 3 times with 5 s / 10 s backoff (intent of
#: spark/spark-jobs/mongo_retry.py): retry the fault, then give up loudly.
API_ATTEMPTS = 3
API_BACKOFF_SECONDS: tuple[float, ...] = (5.0, 10.0)

#: 120 s, not 30 s: the read API runs on Render's free tier, which spins down
#: after ~15 min idle. Nothing keeps the instance warm between the 3-hourly runs,
#: so attempt 1 usually hits a cold start that takes 30-60 s to answer -- and
#: sometimes longer. 30 s timed out the first attempt on every scheduled run;
#: this is a boot allowance, not a latency budget.
HTTP_TIMEOUT_SECONDS = 120.0

#: Wall-clock ceiling for the whole evidence-gathering phase (all endpoints, all
#: attempts, all backoff). Without it the worst case is 3 endpoints x (3 x 120 s +
#: 15 s backoff) = 18.75 min, which blows the workflow timeout. With it: at most
#: ``budget + HTTP_TIMEOUT_SECONDS`` = 8 min, after which the remaining calls fail
#: instantly as ``api_down``/``advisory_error``.
API_TOTAL_BUDGET_SECONDS = 360.0

#: GitHub is a different beast: no cold start, so keep the original 30 s and do
#: NOT inherit the Render boot allowance above.
GITHUB_TIMEOUT_SECONDS = 30.0

#: GitHub's maximum page size for the issues list, and a hard page bound for the
#: open-issue scan (5 000 open issues is far beyond any real repo). Scanning only
#: the first page used to hide the rolling issue once a repo passed 100 open
#: issues, and every later run then opened a duplicate (audit finding 20).
GITHUB_PAGE_SIZE = 100
GITHUB_MAX_PAGES = 50

#: API endpoints used as evidence. Public read routes only.
PATH_CURRENT = "/api/v1/weather/current"
PATH_QUALITY = "/api/v1/weather/quality?days=7"
PATH_PREDICTIONS = "/api/v1/predictions/latest"
PATH_MODELS = "/api/v1/models"

#: Machine-readable verdict embedded in every issue body / comment. Formatting
#: must stay byte-stable: it is what makes "unchanged verdict -> no comment" work.
VERDICT_MARKER = "watchdog-verdict"
_VERDICT_RE = re.compile(r"watchdog-verdict\s*(\{.*?\})\s*-->", re.DOTALL)

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_BAD = "bad"
#: Availability verdict: the read API could not be read. Alerts like a bad signal.
STATUS_API_DOWN = "api_down"
#: Key the availability verdict is recorded under in the byte-stable marker block.
API_STATUS_KEY = "api"
_STATUS_RANK = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_BAD: 2}

Fetcher = Callable[[str], Any]


class WatchdogError(RuntimeError):
    """The check itself failed (transport after retries, unusable payload)."""


class GitHubError(RuntimeError):
    """The GitHub REST call failed."""


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime; ``None`` if absent/garbled."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def age_hours(timestamp: datetime, now: datetime) -> float:
    """Hours between ``timestamp`` and ``now`` (negative if the clock disagrees)."""
    return (now - timestamp).total_seconds() / 3600.0


def grade_age(hours: float, ok_hours: float, warn_hours: float) -> str:
    """Grade an age: ``<= ok`` is ok, ``<= warn`` is warn, beyond is bad."""
    if hours <= ok_hours:
        return STATUS_OK
    if hours <= warn_hours:
        return STATUS_WARN
    return STATUS_BAD


def _fmt_hours(hours: float | None) -> str:
    return "n/d" if hours is None else f"{hours:.1f} h"


def _fmt_ratio(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.3f}"


def _worst(statuses: Sequence[str]) -> str:
    return max(statuses, key=_STATUS_RANK.__getitem__) if statuses else STATUS_OK


# ---------------------------------------------------------------------------
# HTTP (stdlib) with the retry policy above
# ---------------------------------------------------------------------------


def fetch_json(
    url: str,
    *,
    timeout: float = HTTP_TIMEOUT_SECONDS,
    attempts: int = API_ATTEMPTS,
    sleep: Callable[[float], Any] = time.sleep,
    open_url: Callable[..., Any] = urllib.request.urlopen,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """GET ``url`` and decode its JSON body, retrying transport/payload faults.

    Backoff is 5 s then 10 s (``API_BACKOFF_SECONDS``). ``deadline`` is a
    ``monotonic()`` instant (``API_TOTAL_BUDGET_SECONDS``) that also caps each
    attempt's socket timeout and the backoff, so a hung API cannot outlive the
    workflow timeout. When the budget or the attempts are exhausted the last
    error is re-raised as :class:`WatchdogError`; the caller turns that into the
    ``api_down`` verdict, so the failure is reported rather than swallowed.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        budget = float("inf") if deadline is None else deadline - clock()
        if budget <= 0:
            raise WatchdogError(
                f"{url} sin leer: presupuesto de {API_TOTAL_BUDGET_SECONDS:.0f} s agotado "
                f"tras {attempt - 1} intento(s); último error: {last_error}"
            ) from last_error
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with open_url(request, timeout=min(timeout, budget)) as response:
                raw = response.read()
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 -- every fault is retried, then surfaced
            last_error = exc
            if attempt < attempts:
                backoff = API_BACKOFF_SECONDS[min(attempt, len(API_BACKOFF_SECONDS)) - 1]
                sleep(backoff if deadline is None else max(0.0, min(backoff, deadline - clock())))
    raise WatchdogError(f"{url} falló tras {attempts} intentos: {last_error}") from last_error


def api_url(api_base: str, path: str) -> str:
    return api_base.rstrip("/") + path


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """One graded evidence line; a ``bad`` status on a signal triggers the alert."""

    key: str
    label: str
    status: str
    value: str
    threshold: str

    @property
    def is_bad(self) -> bool:
        return self.status == STATUS_BAD

    def as_dict(self) -> dict[str, str]:
        return {
            "signal": self.key,
            "label": self.label,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
        }


AGE_THRESHOLD_TEXT = (
    f"ok ≤ {QUALITY_OK_AGE_HOURS:.0f} h / warn ≤ {QUALITY_WARN_AGE_HOURS:.0f} h / bad > "
    f"{QUALITY_WARN_AGE_HOURS:.0f} h"
)
PREDICTION_AGE_THRESHOLD_TEXT = (
    f"ok ≤ {PREDICTION_OK_AGE_HOURS:.0f} h / bad > {PREDICTION_BAD_AGE_HOURS:.0f} h"
)
COMPLETENESS_THRESHOLD_TEXT = f"bad < {QUALITY_BAD_COMPLETENESS_BELOW:.2f} (peor ciudad)"
COVERAGE_THRESHOLD_TEXT = f"bad < {PREDICTION_BAD_COVERAGE_BELOW:.2f}"


def _payload_dict(payload: Any, *, what: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise WatchdogError(f"{what}: payload inesperado ({type(payload).__name__})")
    return payload


def _payload_list(payload: dict[str, Any], key: str, *, what: str) -> list[dict[str, Any]]:
    raw = payload.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise WatchdogError(f"{what}: '{key}' no es una lista")
    return [item for item in raw if isinstance(item, dict)]


def _newest_timestamp(rows: Sequence[dict[str, Any]], key: str) -> datetime | None:
    """Newest parsed ``key`` across rows; unparseable/missing values are skipped."""
    stamps = [stamp for stamp in (parse_timestamp(row.get(key)) for row in rows) if stamp]
    return max(stamps) if stamps else None


def check_ingest(current: Any, quality: Any, *, now: datetime) -> list[Signal]:
    """Signal 1: observation/feature freshness + worst-city completeness."""
    current_payload = _payload_dict(current, what="weather/current")
    stations = _payload_list(current_payload, "stations", what="weather/current")
    newest = _newest_timestamp(stations, "observed_at")
    if newest is None:
        age_signal = Signal(
            "ingest_age",
            "Observaciones (observed_at más reciente)",
            STATUS_BAD,
            "sin observed_at",
            AGE_THRESHOLD_TEXT,
        )
    else:
        hours = age_hours(newest, now)
        age_signal = Signal(
            "ingest_age",
            "Observaciones (observed_at más reciente)",
            grade_age(hours, QUALITY_OK_AGE_HOURS, QUALITY_WARN_AGE_HOURS),
            _fmt_hours(hours),
            AGE_THRESHOLD_TEXT,
        )

    quality_payload = _payload_dict(quality, what="weather/quality")
    cities = _payload_list(quality_payload, "cities", what="weather/quality")
    completeness_values = [
        float(city["completeness"])
        for city in cities
        if isinstance(city.get("completeness"), (int, float))
        and not isinstance(city.get("completeness"), bool)
    ]
    if not completeness_values:
        completeness_signal = Signal(
            "ingest_completeness",
            "Completitud features 7 d (peor ciudad)",
            STATUS_BAD,
            "sin datos",
            COMPLETENESS_THRESHOLD_TEXT,
        )
    else:
        worst = min(completeness_values)
        completeness_signal = Signal(
            "ingest_completeness",
            "Completitud features 7 d (peor ciudad)",
            STATUS_BAD if worst < QUALITY_BAD_COMPLETENESS_BELOW else STATUS_OK,
            _fmt_ratio(worst),
            COMPLETENESS_THRESHOLD_TEXT,
        )
    return [age_signal, completeness_signal]


def check_predictions(predictions: Any, current: Any, *, now: datetime) -> list[Signal]:
    """Signal 2: inference freshness (newest ``source_timestamp``) + city coverage."""
    rows = _payload_list(
        _payload_dict(predictions, what="predictions/latest"),
        "predictions",
        what="predictions/latest",
    )
    newest = _newest_timestamp(rows, "source_timestamp")
    if newest is None:
        age_signal = Signal(
            "prediction_age",
            "Predicciones (source_timestamp más reciente)",
            STATUS_BAD,
            "sin source_timestamp",
            PREDICTION_AGE_THRESHOLD_TEXT,
        )
    else:
        hours = age_hours(newest, now)
        age_signal = Signal(
            "prediction_age",
            "Predicciones (source_timestamp más reciente)",
            STATUS_BAD if hours > PREDICTION_BAD_AGE_HOURS else STATUS_OK,
            _fmt_hours(hours),
            PREDICTION_AGE_THRESHOLD_TEXT,
        )

    predicted_cities = {row.get("city") for row in rows if isinstance(row.get("city"), str)}
    stations = _payload_list(
        _payload_dict(current, what="weather/current"), "stations", what="weather/current"
    )
    known_cities = {row.get("city") for row in stations if isinstance(row.get("city"), str)}
    if not known_cities:
        coverage_signal = Signal(
            "prediction_coverage",
            "Cobertura de predicciones",
            STATUS_BAD,
            "sin estaciones",
            COVERAGE_THRESHOLD_TEXT,
        )
    else:
        coverage = len(predicted_cities & known_cities) / len(known_cities)
        coverage_signal = Signal(
            "prediction_coverage",
            "Cobertura de predicciones",
            STATUS_BAD if coverage < PREDICTION_BAD_COVERAGE_BELOW else STATUS_OK,
            f"{_fmt_ratio(coverage)} ({len(predicted_cities & known_cities)}/{len(known_cities)})",
            COVERAGE_THRESHOLD_TEXT,
        )
    return [age_signal, coverage_signal]


def _skill_score(entry: dict[str, Any]) -> float | None:
    """Skill score from ``diagnostics`` first, then ``metrics`` (the served shape).

    The API contract puts honest metrics under ``metrics.skill_score`` and
    ``diagnostics`` carries only slices/drift, so both are read: neither key is
    invented when absent, and a null score is "unknown", never "degraded".
    """
    for container in ("diagnostics", "metrics"):
        block = entry.get(container)
        if not isinstance(block, dict):
            continue
        score = block.get("skill_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
    return None


def check_models(models: Any) -> list[str]:
    """Signal 3, advisory only: production models scoring at/below zero."""
    rows = _payload_list(_payload_dict(models, what="models"), "models", what="models")
    advisories: list[str] = []
    for entry in rows:
        stage = entry.get("status") or entry.get("stage")
        if not isinstance(stage, str) or stage.lower() != "production":
            continue
        score = _skill_score(entry)
        if score is None or score > SKILL_SCORE_ADVISORY_AT_OR_BELOW:
            continue
        name = entry.get("name") or "(sin nombre)"
        target = entry.get("target") or "?"
        horizon = entry.get("horizon_hours")
        label = f"{name} ({target}"
        label += f", {horizon} h)" if isinstance(horizon, int) else ")"
        advisories.append(
            f"{label}: skill_score {score:.3f} ≤ {SKILL_SCORE_ADVISORY_AT_OR_BELOW:.1f}"
        )
    return advisories


@dataclass(frozen=True)
class Verdict:
    """Graded evidence for one run. Only signals 1 and 2 can trigger an alert.

    ``api_error`` (the graded endpoints were unreadable) and ``advisory_error``
    (only ``/models`` was unreadable) are carried, not raised: both are reported
    in the issue body and in ``--json``, and only ``api_error`` forces a verdict.
    """

    signals: tuple[Signal, ...]
    advisories: tuple[str, ...] = ()
    advisory_error: str | None = None
    api_error: str | None = None
    api_base: str = DEFAULT_API_BASE
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def statuses(self) -> dict[str, str]:
        statuses = {signal.key: signal.status for signal in self.signals}
        if self.api_error is not None:
            statuses[API_STATUS_KEY] = STATUS_API_DOWN
        return statuses

    @property
    def status(self) -> str:
        if self.api_error is not None:
            return STATUS_API_DOWN
        return _worst([signal.status for signal in self.signals])

    @property
    def triggered(self) -> bool:
        """An unreachable API is an availability alert too, not a silent run."""
        if self.api_error is not None:
            return True
        return any(signal.is_bad for signal in self.signals)

    def statuses_json(self) -> str:
        """Compact, byte-stable block embedded in the issue/comment."""
        return json.dumps(self.statuses, sort_keys=True, separators=(",", ":"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "status": self.status,
            "checked_at": self.checked_at.isoformat(),
            "api_base": self.api_base,
            "signals": [signal.as_dict() for signal in self.signals],
            "advisories": list(self.advisories),
            "advisory_error": self.advisory_error,
            "api_error": self.api_error,
            "verdict_block": self.statuses,
        }


def gather_advisories(api_base: str, fetch: Fetcher) -> tuple[tuple[str, ...], str | None]:
    """Best-effort ``/models`` read: advisory evidence, never a run-killer.

    A broken advisory payload (transport fault, 5xx, unexpected shape) becomes an
    ``advisory_error`` string that is printed in the issue body; the graded
    signals still decide the verdict, so a stale pipeline is never masked by a
    models endpoint that is itself sick.
    """
    try:
        return tuple(check_models(fetch(api_url(api_base, PATH_MODELS)))), None
    except WatchdogError as exc:
        return (), str(exc)


def gather_verdict(*, api_base: str, fetch: Fetcher, now: datetime) -> Verdict:
    """Collect the evidence from the public read API and grade it.

    A failed read of a *graded* endpoint (``current``/``quality``/``predictions``)
    after the retries is an availability signal -- :data:`STATUS_API_DOWN` -- not
    an exception: it must open/update the rolling issue and exit 0. A payload that
    parses but breaks the documented shape still raises :class:`WatchdogError`
    (exit 1): that is this file's contract failing, not the API's availability.
    """
    try:
        current = fetch(api_url(api_base, PATH_CURRENT))
        quality = fetch(api_url(api_base, PATH_QUALITY))
        predictions = fetch(api_url(api_base, PATH_PREDICTIONS))
    except WatchdogError as exc:
        return Verdict(signals=(), api_base=api_base, checked_at=now, api_error=str(exc))

    signals = [
        *check_ingest(current, quality, now=now),
        *check_predictions(predictions, current, now=now),
    ]
    advisories, advisory_error = gather_advisories(api_base, fetch)
    return Verdict(
        signals=tuple(signals),
        advisories=advisories,
        advisory_error=advisory_error,
        api_base=api_base,
        checked_at=now,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _verdict_marker(verdict: Verdict) -> str:
    return f"<!-- {VERDICT_MARKER} {verdict.statuses_json()} -->"


def render_table(verdict: Verdict) -> str:
    lines = ["| Señal | Valor | Umbral | Estado |", "| --- | --- | --- | --- |"]
    for signal in verdict.signals:
        lines.append(f"| {signal.label} | {signal.value} | {signal.threshold} | {signal.status} |")
    return "\n".join(lines)


def render_availability(verdict: Verdict) -> list[str]:
    """Report what could not be read: failures are printed, never swallowed."""
    lines: list[str] = []
    if verdict.api_error is not None:
        lines.extend(
            [
                "",
                f"**API no disponible** (`{STATUS_API_DOWN}`): no pude leer el pipeline tras "
                "los reintentos. Esto es una señal de disponibilidad, no un fallo del switch.",
                "",
                f"- Error: `{verdict.api_error}`",
            ]
        )
    if verdict.advisory_error is not None:
        lines.extend(
            [
                "",
                f"**Aviso no disponible**: falló la lectura de `{PATH_MODELS}` "
                f"(`{verdict.advisory_error}`). Las señales graduadas siguen siendo válidas.",
            ]
        )
    return lines


def render_advisories(verdict: Verdict) -> list[str]:
    if not verdict.advisories:
        return []
    lines = ["", "**Aviso (no dispara alerta): modelos en producción con skill ≤ 0**", ""]
    lines.extend(f"- {advisory}" for advisory in verdict.advisories)
    return lines


def render_footer(verdict: Verdict) -> list[str]:
    return [
        "",
        f"- Comprobado: `{verdict.checked_at.isoformat()}` (UTC)",
        f"- API: {verdict.api_base}",
        f"- Umbrales: edad {QUALITY_OK_AGE_HOURS:.0f} h ok / {QUALITY_WARN_AGE_HOURS:.0f} h warn, "
        f"completitud < {QUALITY_BAD_COMPLETENESS_BELOW:.2f}, cobertura < "
        f"{PREDICTION_BAD_COVERAGE_BELOW:.2f}",
        "- Switch: `scripts/watchdog.py`, workflow `watchdog.yml`.",
        "",
        _verdict_marker(verdict),
    ]


def render_issue_body(verdict: Verdict) -> str:
    if verdict.api_error is not None:
        intro = "El dead-man's switch no pudo leer el pipeline: la API de lectura está caída."
    else:
        intro = "El dead-man's switch encontró datos del pipeline fuera de umbral."
    lines = [
        intro,
        *render_availability(verdict),
        *([render_table(verdict)] if verdict.signals else []),
        *render_advisories(verdict),
        *render_footer(verdict),
    ]
    return "\n".join(lines)


def render_comment_body(verdict: Verdict) -> str:
    lines = [
        "Verdicto actualizado.",
        *render_availability(verdict),
        *([render_table(verdict)] if verdict.signals else []),
        *render_advisories(verdict),
        *render_footer(verdict),
    ]
    return "\n".join(lines)


def render_recovery_body(verdict: Verdict) -> str:
    lines = [
        "Señales de datos recuperadas: cierro la incidencia. "
        "El switch sigue programado; si vuelve a estar fuera de umbral abrirá una nueva.",
        *render_availability(verdict),
        *([render_table(verdict)] if verdict.signals else []),
        *render_advisories(verdict),
        *render_footer(verdict),
    ]
    return "\n".join(lines)


def extract_verdict(text: str | None) -> dict[str, str] | None:
    """Parse the embedded verdict block back out of an issue body / comment."""
    match = _VERDICT_RE.search(text or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Decision table (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """What to do this run. ``comment`` is only meaningful with an open issue."""

    kind: str  # "create" | "comment" | "close" | "none"
    comment: bool = False
    reason: str = ""


def plan_action(
    *,
    triggered: bool,
    issue_number: int | None,
    last_verdict: dict[str, str] | None,
    current_verdict: dict[str, str],
) -> Action:
    """Exactly one rolling issue; at most one comment per run.

    * triggered + no issue -> create
    * triggered + issue + changed verdict -> comment
    * triggered + issue + unchanged verdict -> none
    * not triggered + issue -> close (comment the recovery unless it is already
      the last recorded verdict, i.e. a previous run recorded it)
    * not triggered + no issue -> none
    """
    if triggered:
        if issue_number is None:
            return Action("create", reason="señal bad y no hay incidencia abierta")
        if last_verdict == current_verdict:
            return Action("none", reason="veredicto sin cambios: no se comenta")
        return Action("comment", comment=True, reason="señal bad y el veredicto cambió")
    if issue_number is not None:
        if last_verdict == current_verdict:
            return Action("close", comment=False, reason="recuperado; la recuperación ya consta")
        return Action("close", comment=True, reason="recuperado: se comenta y se cierra")
    return Action("none", reason="todo en umbral y no hay incidencia abierta")


# ---------------------------------------------------------------------------
# GitHub REST client (issues only)
# ---------------------------------------------------------------------------


class GitHubClient:
    """Minimal issues-only client; the token is never logged nor echoed."""

    def __init__(
        self,
        *,
        repo: str,
        token: str | None,
        base: str = DEFAULT_GITHUB_API,
        sleep: Callable[[float], Any] = time.sleep,
        open_url: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.repo = repo
        self._token = token
        self.base = base.rstrip("/")
        self._sleep = sleep
        self._open_url = open_url

    # -- transport ---------------------------------------------------------
    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[int, Any]:
        url = path if path.startswith("http") else self.base + path
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "weather-pipeline-watchdog",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        last_error: Exception | None = None
        for attempt in range(1, API_ATTEMPTS + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method=method)
                with self._open_url(request, timeout=GITHUB_TIMEOUT_SECONDS) as response:
                    text = response.read().decode("utf-8")
                    return response.status, (json.loads(text) if text.strip() else None)
            except urllib.error.HTTPError as exc:
                # A 4xx is an answer, not a transient fault: surface it now.
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")
                except Exception:  # pragma: no cover - best effort only
                    detail = ""
                if exc.code < 500:
                    return exc.code, detail
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - transport faults are retried
                last_error = exc
            if attempt < API_ATTEMPTS:
                self._sleep(API_BACKOFF_SECONDS[min(attempt, len(API_BACKOFF_SECONDS)) - 1])
        raise GitHubError(f"{method} {path} falló tras {API_ATTEMPTS} intentos: {last_error}")

    # -- reads -------------------------------------------------------------
    def find_open_issue(self) -> dict[str, Any] | None:
        # Match on the title, not the label: the title is the issue's identity, so
        # a hand-removed label cannot fork a second rolling issue.
        # Paginated until a short page: the rolling issue is not necessarily on
        # the first 100 (audit finding 20).
        for page in range(1, GITHUB_MAX_PAGES + 1):
            query = f"?state=open&per_page={GITHUB_PAGE_SIZE}&page={page}"
            status, payload = self._request("GET", f"/repos/{self.repo}/issues{query}")
            if status == 404:
                raise GitHubError(f"repo {self.repo} no encontrado")
            if status >= 300:
                raise GitHubError(f"listar incidencias falló (HTTP {status}): {payload}")
            issues = [item for item in payload or [] if isinstance(item, dict)]
            for issue in issues:
                if issue.get("title") == ISSUE_TITLE:
                    return issue
            if len(issues) < GITHUB_PAGE_SIZE:
                return None
        return None

    def comments(self, number: int) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for page in range(1, 6):  # 500 comments is far beyond any real history
            status, payload = self._request(
                "GET",
                f"/repos/{self.repo}/issues/{number}/comments?per_page=100&page={page}",
            )
            if status >= 300:
                raise GitHubError(f"listar comentarios falló (HTTP {status}): {payload}")
            chunk = [item for item in payload or [] if isinstance(item, dict)]
            collected.extend(chunk)
            if len(chunk) < 100:
                break
        return collected

    def last_recorded_verdict(self, issue: dict[str, Any]) -> dict[str, str] | None:
        """Verdict block of the newest comment, else the issue body."""
        number = issue.get("number")
        if not isinstance(number, int):
            return None
        comments = self.comments(number)
        for comment in reversed(comments):
            parsed = extract_verdict(comment.get("body"))
            if parsed is not None:
                return parsed
        return extract_verdict(issue.get("body"))

    # -- writes ------------------------------------------------------------
    def create_issue(self, body: str) -> dict[str, Any]:
        status, payload = self._request(
            "POST",
            f"/repos/{self.repo}/issues",
            {"title": ISSUE_TITLE, "body": body, "labels": [ISSUE_LABEL]},
        )
        if status == 422:
            # Most likely the label does not exist yet: create it, retry once.
            self.ensure_label()
            status, payload = self._request(
                "POST",
                f"/repos/{self.repo}/issues",
                {"title": ISSUE_TITLE, "body": body, "labels": [ISSUE_LABEL]},
            )
        if status >= 300:
            raise GitHubError(f"crear incidencia falló (HTTP {status}): {payload}")
        return payload or {}

    def ensure_label(self) -> None:
        status, payload = self._request(
            "POST",
            f"/repos/{self.repo}/labels",
            {
                "name": ISSUE_LABEL,
                "color": ISSUE_LABEL_COLOR,
                "description": ISSUE_LABEL_DESCRIPTION,
            },
        )
        if status >= 300 and status != 422:  # 422 = already exists, which is fine
            raise GitHubError(f"crear etiqueta falló (HTTP {status}): {payload}")

    def comment(self, number: int, body: str) -> dict[str, Any]:
        status, payload = self._request(
            "POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body}
        )
        if status >= 300:
            raise GitHubError(f"comentar la incidencia falló (HTTP {status}): {payload}")
        return payload or {}

    def close(self, number: int) -> dict[str, Any]:
        status, payload = self._request(
            "PATCH", f"/repos/{self.repo}/issues/{number}", {"state": "closed"}
        )
        if status >= 300:
            raise GitHubError(f"cerrar la incidencia falló (HTTP {status}): {payload}")
        return payload or {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watchdog",
        description="Dead-man's switch: stale ingest/features/predictions -> one GitHub issue.",
    )
    parser.add_argument("--api", default=DEFAULT_API_BASE, help="read-API base URL")
    parser.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY", ""), help="OWNER/NAME of this repo"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token (default $GITHUB_TOKEN)",
    )
    parser.add_argument("--dry-run", action="store_true", help="compute and print; never write")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def run(
    args: argparse.Namespace,
    *,
    fetch: Fetcher | None = None,
    client_factory: Callable[..., GitHubClient] | None = None,
    now: datetime | None = None,
    out: Callable[[str], None] = print,
    err: Callable[[str], None] | None = None,
) -> int:
    err = err or (lambda line: print(line, file=sys.stderr))
    now = now or datetime.now(timezone.utc)
    if fetch is None:
        # One wall-clock budget for every endpoint of this run (see
        # API_TOTAL_BUDGET_SECONDS): a hung API can never outlive the workflow.
        deadline = time.monotonic() + API_TOTAL_BUDGET_SECONDS

        def bounded_fetch(url: str) -> Any:
            return fetch_json(url, deadline=deadline)

        fetch = bounded_fetch

    try:
        verdict = gather_verdict(api_base=args.api, fetch=fetch, now=now)
    except WatchdogError as exc:
        err(f"watchdog: no pude leer el pipeline (fallo del switch): {exc}")
        return 1

    # Availability is a verdict, not an exception: report it, then alert like
    # any other bad signal (same rolling issue) and exit 0.
    if verdict.api_error is not None:
        err(f"watchdog: API no disponible ({STATUS_API_DOWN}): {verdict.api_error}")
    if verdict.advisory_error is not None:
        err(f"watchdog: aviso de modelos no disponible: {verdict.advisory_error}")

    if not args.repo and not args.dry_run:
        err("watchdog: falta --repo o $GITHUB_REPOSITORY (el switch no puede escribir sin repo)")
        return 1
    if not args.token and not args.dry_run:
        err("watchdog: falta --token o $GITHUB_TOKEN (el switch no puede escribir sin token)")
        return 1

    client_factory = client_factory or GitHubClient
    issue: dict[str, Any] | None = None
    last_verdict: dict[str, str] | None = None
    client: GitHubClient | None = None
    issue_state_known = False
    if args.repo:
        try:
            client = client_factory(repo=args.repo, token=args.token or None)
            issue = client.find_open_issue()
            last_verdict = client.last_recorded_verdict(issue) if issue else None
            issue_state_known = True
        except GitHubError as exc:
            client = None
            if not args.dry_run:
                err(f"watchdog: no pude leer el estado de la incidencia: {exc}")
                return 1
            err(f"watchdog: estado de la incidencia desconocido ({exc}); plan orientativo")

    number = issue.get("number") if isinstance(issue, dict) else None
    number = number if isinstance(number, int) else None
    action = plan_action(
        triggered=verdict.triggered,
        issue_number=number,
        last_verdict=last_verdict,
        current_verdict=verdict.statuses,
    )

    if args.dry_run:
        if args.json:
            payload = {**verdict.as_dict(), "action": action.kind, "action_reason": action.reason}
            out(json.dumps(payload))
        else:
            out(f"veredicto: {verdict.status} (triggered={verdict.triggered})")
            for line in render_availability(verdict):
                if line.strip():
                    out(line)
            if verdict.signals:
                out(render_table(verdict))
            for advisory in verdict.advisories:
                out(f"aviso: {advisory}")
            if not issue_state_known:
                out(
                    "plan: create/comment/close según el estado real del issue "
                    "(desconocido aquí: falta --repo o falló la lectura); nada escrito"
                )
            else:
                out(
                    f"plan: {action.kind}{' + comentario' if action.comment else ''} "
                    f"({action.reason})"
                )
            out("dry-run: no se ha escrito nada")
        return 0

    assert client is not None  # guaranteed: a real run without a repo/token returns 1 above
    try:
        if action.kind == "create":
            created = client.create_issue(render_issue_body(verdict))
            out(f"watchdog: incidencia creada #{created.get('number', '?')}")
        elif action.kind == "comment" and number is not None:
            client.comment(number, render_comment_body(verdict))
            out(f"watchdog: comentario añadido a #{number}")
        elif action.kind == "close" and number is not None:
            if action.comment:
                client.comment(number, render_recovery_body(verdict))
            client.close(number)
            out(f"watchdog: incidencia #{number} cerrada")
        else:
            out(f"watchdog: sin cambios ({action.reason})")
    except GitHubError as exc:
        err(f"watchdog: no pude actualizar la incidencia: {exc}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
