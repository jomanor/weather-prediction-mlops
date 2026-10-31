"""``GET /metrics``: process-local request counters and latency estimates.

JSON, not Prometheus text: the API runs on a single free-tier instance with no
scraper, and the access log is already JSON, so an operator (or the watchdog)
reads this with ``curl | jq`` and no new dependency is needed.

The counters live in the process and reset on every restart. On the free Render
tier the instance also sleeps when idle, so waking up starts them from zero
again: these numbers describe the current instance's life, **not a 28-day
window**, and must not be reported as one.

One key is not a counter: ``database.indexes`` carries the outcome of the
startup ``ensure_indexes`` call (see ``main._prepare_database``), so a process
serving every read as a collection scan is visible instead of merely slow.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.metrics import RequestMetrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics(request: Request) -> JSONResponse:
    """Totals, per-route percentiles and the p95 SLO verdict for this process.

    ``routes`` is keyed by ``(method, route template)`` — the template the app
    declares, never the served path — with an unmatched request (404 before
    routing) in ``<unmatched>``. The method column is normalized to
    GET/HEAD/POST/PUT/PATCH/DELETE/OPTIONS; any other token is ``<other>``, and
    the table is capped at ``MAX_ROUTES`` entries in total, with everything past
    the cap folded into one ``(<other>, <other>)`` row. Query strings, city
    names, request ids and client-supplied method text are never recorded, so
    the payload is safe to serve publicly. ``error_count`` counts 5xx only.

    Percentiles are **bucket upper bounds** (no interpolation): 5/10/25/50/100/
    250/500/1000/2500/5000/10000 ms, so a 30 ms request is reported as 50 ms and
    the estimate is never below the true value. One above 10 000 ms reports the
    largest duration seen on that route.

    ``/metrics`` is counted like every other route; the request reading this
    payload is counted once its own response is complete, so it shows up from
    the next call on. The response is ``Cache-Control: no-store``.

    ``database.indexes`` is the one non-counter key: ``app.state.index_status``
    (see ``main._prepare_database``) as ``{"ok": bool|null, "error": str|null}``,
    ``null`` while the background index task has not run. It is reported here and
    not in ``/health/ready`` on purpose: a missing index slows reads down, it does
    not make the instance unable to serve, and readiness failure would have Render
    restart a working process.
    """
    recorder: RequestMetrics = request.app.state.metrics
    payload = recorder.snapshot()
    status = request.app.state.index_status
    payload["database"] = {
        "indexes": {
            "ok": status["ok"] if status is not None else None,
            "error": status["error"] if status is not None else None,
        }
    }
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})
