"""Process-lifetime request metrics behind ``GET /metrics``.

The read API already emitted one JSON access line per request
(``app.core.logging``) but nothing aggregated it, so there was no latency or
availability surface an SLO could be measured against. This module is that
aggregator: in-memory, no new dependency (no Prometheus client, no
OpenTelemetry), no background task, no new collection, and bounded memory.

Four properties are deliberate, not incidental:

* **Bucket counters, never per-request samples.** Latency is counted into the
  fixed buckets of ``LATENCY_BUCKETS_MS`` so memory is O(routes x buckets) and
  cannot grow with traffic. A list of durations would grow without bound on a
  memory-constrained free-tier instance.
* **Route templates and a fixed method set only.** The recorded key is
  ``(normalized method, matched route template)`` — ``GET`` ×
  ``/api/v1/weather/history/{city}`` — never the served path and never the raw
  method token, so no city name, query string, request id or crafted method can
  reach the payload: it is safe to serve publicly.
* **The table is bounded by ``MAX_ROUTES`` entries, full stop.** The key
  universe is finite by construction (``KNOWN_METHODS`` plus ``<other>``, times
  app-defined route templates plus ``<unmatched>``) and the cap applies to the
  whole table, reserved labels included: every request past the cap is folded
  into the single ``(<other>, <other>)`` entry. Neither method cardinality nor
  path cardinality can grow the map.
* **Per process, reset on restart.** The counters live in this object. On the
  free Render tier the instance sleeps when idle and the numbers restart from
  zero when it wakes, so they describe the current instance's life only — not a
  28-day window. Anything longer needs an exporter or a store; deliberately out
  of scope here.

Counters are mutated without an ``await`` and FastAPI runs every handler for an
app on one event loop, so no lock is needed (the same reasoning as
``app.core.cache``). ``record`` is the only writer and the caller wraps it in
``try/except`` so a metrics failure can never fail a request.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

#: Upper bound, in ms, of each latency bucket. Bucket ``i`` counts the durations
#: in ``(LATENCY_BUCKETS_MS[i-1], LATENCY_BUCKETS_MS[i]]`` — a duration equal to
#: a bound lands in the bucket *ending* at it. The extra counter past the last
#: bound is the ``+Inf`` overflow bucket.
LATENCY_BUCKETS_MS: tuple[float, ...] = (
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
)
#: Number of counters per route: one per finite bound plus the ``+Inf`` bucket.
BUCKET_COUNT = len(LATENCY_BUCKETS_MS) + 1

#: Quantiles reported per route. ``0.95`` is the SLO quantile.
PERCENTILES: tuple[tuple[str, float], ...] = (("p50_ms", 0.50), ("p95_ms", 0.95), ("p99_ms", 0.99))

#: Target for ``GET /metrics``'s ``slo.met`` verdict (roadmap: p95 < 2 s).
SLO_LATENCY_P95_MS_TARGET = 2000

#: Label for a request the router never matched (a 404 raised before routing).
UNMATCHED_ROUTE = "<unmatched>"
#: Label for the overflow entry and for methods outside ``KNOWN_METHODS``.
OTHER_ROUTE = "<other>"

#: HTTP methods that keep their own row. A method outside this set — lowercase
#: ``get``, ``TRACE``, a crafted ``M4711`` — is reported as ``<other>``, so the
#: method column never carries client text and the key universe stays finite.
#: Membership is exact: no case folding, because a lowercase token is not a
#: method this API serves (Starlette answers it 405) and echoing it back would
#: reintroduce the reflection this guards against.
KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
#: Reporting label for any method outside ``KNOWN_METHODS``.
OTHER_METHOD = OTHER_ROUTE
#: The single entry every request past the cap folds into. Both columns are
#: ``<other>``, so overflow can add at most one entry whatever the client sends.
SINK_KEY = (OTHER_METHOD, OTHER_ROUTE)
#: Hard bound on table entries — every ``(method, route template)`` combination,
#: reserved labels included. The app declares ~54 mount+path combinations and a
#: real client uses one method on most of them (plus ``POST /refresh``,
#: ``POST/DELETE /cities`` and the odd preflight), so a healthy table is far
#: below this; 256 leaves room for that and for a scanner's method sweep, and
#: ``test_default_cap_leaves_room_for_every_documented_route`` trips if a route
#: explosion ever makes real traffic compete for slots.
MAX_ROUTES = 256


def normalize_method(method: str) -> str:
    """The reporting label for the raw client method token.

    Exact uppercase membership only: anything else becomes ``<other>``, which is
    what makes the method column finite and free of client text.
    """
    return method if method in KNOWN_METHODS else OTHER_METHOD


def latency_bucket_index(duration_ms: float) -> int:
    """Index of the bucket that counts ``duration_ms`` (``BUCKET_COUNT`` is the overflow).

    Negative and non-finite inputs fold into the first bucket rather than
    raising: a broken duration must not cost a counter update.
    """
    if not duration_ms >= 0:  # False for NaN too
        return 0
    # Bounds are ascending; ``<= bound`` means the bucket ending at that bound.
    for index, bound in enumerate(LATENCY_BUCKETS_MS):
        if duration_ms <= bound:
            return index
    return BUCKET_COUNT - 1


def bucket_percentile_ms(
    buckets: Sequence[int],
    *,
    count: int,
    quantile: float,
    overflow_ms: float,
) -> float | None:
    """Upper-bound percentile estimate from fixed-bucket counters.

    **Convention: no interpolation, bucket upper bound.** The result is the upper
    bound of the first bucket whose cumulative count covers the quantile's rank,
    i.e. the true percentile rounded *up* to the next bucket edge
    (5/10/25/50/100/250/500/1000/2500/5000/10000 ms). A duration of 30 ms is
    reported as 50 ms and one of 120 ms as 250 ms; the estimate is therefore
    conservative (never below the true value), which is what an SLO verdict
    wants. A rank that lands in the ``+Inf`` bucket reports ``overflow_ms``, the
    largest duration observed on that route.

    ``count`` is the number of samples already summed into ``buckets``; with no
    samples there is nothing to estimate and the result is ``None``.
    """
    if count <= 0:
        return None
    seen = 0
    for index, observed in enumerate(buckets):
        seen += observed
        # ``seen / count >= quantile`` is the rank rule (smallest k with
        # P(X <= bound_k) >= q) written without float rounding on ``q * count``.
        if seen / count >= quantile:
            if index < len(LATENCY_BUCKETS_MS):
                return float(LATENCY_BUCKETS_MS[index])
            return float(overflow_ms)
    return float(overflow_ms)


def route_template(template: str | None, path: str, prefixes: Sequence[str]) -> str | None:
    """The matched route template with its mount prefix, e.g. ``/api/v1/cities``.

    ``None`` when no route matched (the request 404s before routing), which the
    caller records as ``<unmatched>``.

    The prefix cannot come from the route object: FastAPI mounts one aggregate
    router twice (``/api/v1`` and the deprecated ``/api`` alias) and both mounts
    share the same route objects, so ``scope["route"].path`` is the *unprefixed*
    template. The served path picks the mount instead — longest prefix first, on
    a segment boundary — and only the template is returned: the path itself is
    user input (the city name lives in it) and is never recorded.
    """
    if not template:
        return None
    for prefix in sorted(prefixes, key=len, reverse=True):
        if path == prefix or path.startswith(f"{prefix}/"):
            return f"{prefix}{template}"
    return template


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


class _Stats:
    """Counters for one route, or for the whole process.

    One instance per ``(method, route template)`` key plus one for the totals,
    so the payload's per-route numbers and its process-wide numbers come from
    the same arithmetic.
    """

    __slots__ = ("buckets", "cache_hits", "count", "error_count", "max_ms")

    def __init__(self) -> None:
        self.count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.buckets: list[int] = [0] * BUCKET_COUNT
        self.max_ms = 0.0

    def observe(self, duration_ms: float, *, error: bool, cache_hit: bool) -> None:
        self.count += 1
        if error:
            self.error_count += 1
        if cache_hit:
            self.cache_hits += 1
        self.buckets[latency_bucket_index(duration_ms)] += 1
        if duration_ms > self.max_ms:
            self.max_ms = duration_ms

    def percentile_ms(self, quantile: float) -> float | None:
        return bucket_percentile_ms(
            self.buckets,
            count=self.count,
            quantile=quantile,
            overflow_ms=self.max_ms,
        )


class RequestMetrics:
    """Process-lifetime request counters, keyed by ``(method, route template)``.

    ``mount_prefixes`` are the prefixes the app mounts its routes under (see
    ``main.API_V1_PREFIX``); they are passed in so this module owns no knowledge
    of the app's URL layout.

    **Invariant: ``len(routes) <= max_routes`` at all times**, whatever the
    client sends. The method is normalized to ``KNOWN_METHODS`` (everything else
    is ``<other>``) and one slot is always kept free for the ``<other>``/``<other>``
    sink, so at most ``max_routes - 1`` real keys plus the single overflow entry
    exist — method cardinality and path cardinality are both irrelevant, and
    every label in the payload is either an app-defined route template or one of
    ``<unmatched>``/``<other>``.
    """

    def __init__(
        self,
        *,
        mount_prefixes: Sequence[str] = (),
        max_routes: int = MAX_ROUTES,
        clock: Callable[[], float] = time.monotonic,
        started_at: datetime | None = None,
    ) -> None:
        self._prefixes = tuple(mount_prefixes)
        # At least one slot is needed: the sink always has to fit.
        self._max_routes = max(1, max_routes)
        self._clock = clock
        started = started_at or datetime.now(timezone.utc)
        self.started_at = (
            started.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        )
        self._start = clock()
        self._routes: dict[tuple[str, str], _Stats] = {}
        self._admitted = 0
        self._all = _Stats()
        self._by_status_class: dict[str, int] = {}

    def record(
        self,
        *,
        method: str,
        route: str | None,
        status_code: int,
        duration_ms: float,
        cache_hit: bool,
    ) -> None:
        """Count one finished request. Never awaits, so it cannot interleave."""
        if not duration_ms >= 0:  # NaN or a clock glitch must not poison max_ms
            duration_ms = 0.0
        key = self._key(method, route or UNMATCHED_ROUTE)
        self._all.observe(duration_ms, error=status_code >= 500, cache_hit=cache_hit)
        status_class = f"{status_code // 100}xx"
        self._by_status_class[status_class] = self._by_status_class.get(status_class, 0) + 1
        stats = self._routes.get(key)
        if stats is None:
            stats = _Stats()
            self._routes[key] = stats
        stats.observe(duration_ms, error=status_code >= 500, cache_hit=cache_hit)

    def _key(self, method: str, route: str) -> tuple[str, str]:
        key = (normalize_method(method), route)
        if key in self._routes or key == SINK_KEY:
            return key
        # One slot stays free for the sink, so overflow always fits and the
        # table never exceeds ``max_routes`` entries.
        if self._admitted >= self._max_routes - 1:
            return SINK_KEY
        self._admitted += 1
        return key

    @staticmethod
    def _route_entry(method: str, route: str, stats: _Stats) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "method": method,
            "route": route,
            "count": stats.count,
            # 5xx only: a 4xx is a client error, not an availability failure.
            "error_count": stats.error_count,
            "cache_hit_rate": _rate(stats.cache_hits, stats.count),
        }
        for name, quantile in PERCENTILES:
            value = stats.percentile_ms(quantile)
            entry[name] = None if value is None else round(value, 1)
        entry["max_ms"] = round(stats.max_ms, 1)
        return entry

    def snapshot(self) -> dict[str, Any]:
        """The ``GET /metrics`` body: totals, per-route percentiles, the SLO verdict."""
        p95 = self._all.percentile_ms(0.95)
        routes = [
            self._route_entry(method, route, stats)
            for (method, route), stats in sorted(
                self._routes.items(),
                key=lambda item: (-item[1].count, item[0][1], item[0][0]),
            )
        ]
        return {
            "started_at": self.started_at,
            "uptime_seconds": round(self._clock() - self._start, 3),
            "requests": {
                "total": self._all.count,
                "by_status_class": dict(sorted(self._by_status_class.items())),
                "errors_5xx": self._all.error_count,
                "cache_hits": self._all.cache_hits,
                "cache_hit_rate": _rate(self._all.cache_hits, self._all.count),
            },
            "routes": routes,
            "slo": {
                "latency_p95_ms_target": SLO_LATENCY_P95_MS_TARGET,
                "p95_ms": None if p95 is None else round(p95, 1),
                # No traffic means nothing has violated the target yet.
                "met": p95 is None or p95 <= SLO_LATENCY_P95_MS_TARGET,
            },
        }


__all__ = [
    "KNOWN_METHODS",
    "LATENCY_BUCKETS_MS",
    "MAX_ROUTES",
    "OTHER_METHOD",
    "OTHER_ROUTE",
    "PERCENTILES",
    "SINK_KEY",
    "SLO_LATENCY_P95_MS_TARGET",
    "UNMATCHED_ROUTE",
    "RequestMetrics",
    "bucket_percentile_ms",
    "latency_bucket_index",
    "normalize_method",
    "route_template",
]
