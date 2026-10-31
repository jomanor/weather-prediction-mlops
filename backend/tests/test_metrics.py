"""``GET /metrics``: bucket maths, route templates, the cap, and the wiring.

The recorder is in-memory and bucketed, so these tests pin the two things that
can silently go wrong: the percentile convention at the bucket edges, and what
the route key is allowed to contain (a template — never a city, a query string
or a request id, because the payload is served publicly).
"""

import json
from types import SimpleNamespace

import pytest

from app.core.logging import RequestContextMiddleware
from app.core.metrics import (
    KNOWN_METHODS,
    LATENCY_BUCKETS_MS,
    MAX_ROUTES,
    OTHER_METHOD,
    OTHER_ROUTE,
    SINK_KEY,
    SLO_LATENCY_P95_MS_TARGET,
    UNMATCHED_ROUTE,
    RequestMetrics,
    bucket_percentile_ms,
    latency_bucket_index,
    normalize_method,
    route_template,
)
from app.main import API_LEGACY_PREFIX, API_V1_PREFIX

PREFIXES = (API_V1_PREFIX, API_LEGACY_PREFIX)
EMPTY_BUCKETS = [0] * (len(LATENCY_BUCKETS_MS) + 1)


def _recorder(**kwargs) -> RequestMetrics:
    return RequestMetrics(mount_prefixes=PREFIXES, **kwargs)


def _route(body: dict, method: str, route: str) -> dict:
    return next(
        entry for entry in body["routes"] if entry["method"] == method and entry["route"] == route
    )


def _hit(metrics: RequestMetrics, route: str | None, duration_ms: float, *, status: int = 200):
    metrics.record(
        method="GET",
        route=route,
        status_code=status,
        duration_ms=duration_ms,
        cache_hit=False,
    )


async def _metrics_body(client) -> dict:
    response = await client.get("/api/v1/metrics")
    assert response.status_code == 200
    return response.json()


# --- bucket -> percentile maths ------------------------------------------------


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [
        (0.0, 0),
        # A duration equal to a bound is in the bucket that *ends* at it.
        (5.0, 0),
        (5.1, 1),
        (10.0, 1),
        (100.0, 4),
        (2500.0, 8),
        (10000.0, 10),
        (10000.1, 11),  # the +Inf overflow counter
        (-1.0, 0),  # a broken duration folds into the first bucket
        (float("nan"), 0),
    ],
)
def test_latency_bucket_index_at_the_edges(duration_ms: float, expected: int):
    assert latency_bucket_index(duration_ms) == expected


def test_percentiles_of_a_known_distribution():
    metrics = _recorder()
    for _ in range(95):
        _hit(metrics, "/api/v1/cities", 100.0)
    for _ in range(5):
        _hit(metrics, "/api/v1/cities", 1000.0)

    entry = _route(metrics.snapshot(), "GET", "/api/v1/cities")

    assert entry["count"] == 100
    # 95 of 100 samples are <= 100 ms, exactly the p95 rank: it stays in that bucket.
    assert entry["p50_ms"] == 100.0
    assert entry["p95_ms"] == 100.0
    assert entry["p99_ms"] == 1000.0
    assert entry["max_ms"] == 1000.0


def test_percentile_is_the_bucket_upper_bound_not_an_interpolation():
    metrics = _recorder()
    for duration in (1.0, 3.0, 7.0):
        _hit(metrics, "/api/v1/x", duration)

    entry = _route(metrics.snapshot(), "GET", "/api/v1/x")

    # The median sample is 3 ms and the convention rounds up to the bucket edge,
    # so the estimate is never below the true value.
    assert entry["p50_ms"] == 5.0
    assert entry["max_ms"] == 7.0


def test_single_sample_route():
    metrics = _recorder()
    _hit(metrics, "/api/v1/x", 7.0)

    entry = _route(metrics.snapshot(), "GET", "/api/v1/x")

    assert entry["count"] == 1
    assert (entry["p50_ms"], entry["p95_ms"], entry["p99_ms"]) == (10.0, 10.0, 10.0)
    assert entry["max_ms"] == 7.0


def test_overflow_bucket_reports_the_observed_maximum():
    metrics = _recorder()
    _hit(metrics, "/api/v1/slow", 12000.0)

    entry = _route(metrics.snapshot(), "GET", "/api/v1/slow")

    assert entry["p99_ms"] == 12000.0
    assert entry["max_ms"] == 12000.0
    assert metrics.snapshot()["slo"]["met"] is False


def test_empty_recorder_has_no_routes_and_no_percentiles():
    body = _recorder().snapshot()

    assert body["routes"] == []
    assert body["requests"] == {
        "total": 0,
        "by_status_class": {},
        "errors_5xx": 0,
        "cache_hits": 0,
        "cache_hit_rate": None,
    }
    assert body["slo"] == {
        "latency_p95_ms_target": SLO_LATENCY_P95_MS_TARGET,
        "p95_ms": None,
        "met": True,
    }
    assert body["uptime_seconds"] >= 0.0
    assert body["started_at"].endswith("Z")
    # An empty distribution has nothing to estimate.
    assert bucket_percentile_ms(EMPTY_BUCKETS, count=0, quantile=0.95, overflow_ms=0.0) is None


def test_slo_verdict_uses_the_whole_process_p95():
    metrics = _recorder()
    for _ in range(19):
        _hit(metrics, "/api/v1/fast", 10.0)
    _hit(metrics, "/api/v1/fast", 3000.0)

    # 19 of 20 samples are <= 10 ms, so the process-wide p95 sits under the target
    # even though the route saw one 3 s request.
    assert metrics.snapshot()["slo"] == {
        "latency_p95_ms_target": SLO_LATENCY_P95_MS_TARGET,
        "p95_ms": 10.0,
        "met": True,
    }

    metrics = _recorder()
    for _ in range(18):
        _hit(metrics, "/api/v1/fast", 10.0)
    for _ in range(2):
        _hit(metrics, "/api/v1/fast", 3000.0)

    slo = metrics.snapshot()["slo"]

    # 10 % of the samples are slow, so the p95 lands in the 2500-5000 ms bucket
    # and the upper-bound convention breaks the 2 s target.
    assert slo["p95_ms"] == 5000.0
    assert slo["met"] is False


# --- route templates -----------------------------------------------------------


@pytest.mark.parametrize(
    ("template", "path", "expected"),
    [
        (
            "/weather/history/{city}",
            "/api/v1/weather/history/Madrid",
            "/api/v1/weather/history/{city}",
        ),
        ("/weather/history/{city}", "/api/weather/history/Alicante", "/api/weather/history/{city}"),
        ("/metrics", "/api/v1/metrics", "/api/v1/metrics"),
        # No configured mount matches: keep the bare template rather than guess.
        ("/metrics", "/elsewhere/metrics", "/metrics"),
        # "/api" only matches on a segment boundary.
        ("/x", "/apix/x", "/x"),
        (None, "/api/v1/nope", None),
    ],
)
def test_route_template_resolves_the_mount_and_never_the_served_path(
    template: str | None, path: str, expected: str | None
):
    resolved = route_template(template, path, PREFIXES)

    assert resolved == expected
    assert "Madrid" not in str(resolved)


# --- caps ----------------------------------------------------------------------


# --- the cap and the finite key universe ---------------------------------------


def test_route_table_is_capped_at_max_routes_and_overflows_into_other():
    # One slot is always kept free for the sink: 2 real keys + <other>/<other>.
    metrics = _recorder(max_routes=3)

    for index in range(3):
        _hit(metrics, f"/api/v1/r{index}", 1.0)

    body = metrics.snapshot()
    assert len(body["routes"]) == 3
    assert {(entry["method"], entry["route"]) for entry in body["routes"]} == {
        ("GET", "/api/v1/r0"),
        ("GET", "/api/v1/r1"),
        (OTHER_METHOD, OTHER_ROUTE),
    }
    assert _route(body, OTHER_METHOD, OTHER_ROUTE)["count"] == 1

    for index in range(3, 10):
        _hit(metrics, f"/api/v1/r{index}", 1.0)

    body = metrics.snapshot()
    assert len(body["routes"]) == 3
    assert _route(body, OTHER_METHOD, OTHER_ROUTE)["count"] == 8
    assert body["requests"]["total"] == 10


def test_default_cap_leaves_room_for_every_documented_route(app):
    """Real traffic must never compete with a scanner's overflow for slots."""
    # OpenAPI spells methods lowercase and mixes non-method keys into the item.
    method_keys = {method.lower() for method in KNOWN_METHODS}
    documented = {
        (method.upper(), path)
        for path, item in app.openapi()["paths"].items()
        for method in item
        if method in method_keys
    }

    assert len(documented) > 0
    assert len(documented) < MAX_ROUTES // 2


def test_payload_size_is_bounded_by_the_table_cap():
    metrics = _recorder()
    for index in range(5000):
        metrics.record(
            method=f"M{index}",
            route=f"/api/v1/route/{index}",
            status_code=500,
            duration_ms=9000.0 + index,
            cache_hit=True,
        )

    body = metrics.snapshot()

    assert len(body["routes"]) == MAX_ROUTES
    # Stated bound: the body cannot exceed the cap times one entry (~200 B).
    assert len(json.dumps(body).encode()) < 64 * 1024


def test_reserved_labels_are_not_exempt_from_the_cap():
    metrics = _recorder(max_routes=2)
    _hit(metrics, "/api/v1/r0", 1.0)

    # A third distinct key, even an unmatched one, folds into the sink.
    _hit(metrics, None, 1.0, status=404)
    _hit(metrics, None, 1.0, status=404)

    body = metrics.snapshot()

    assert len(body["routes"]) == 2
    assert _route(body, OTHER_METHOD, OTHER_ROUTE)["count"] == 2
    assert UNMATCHED_ROUTE not in {entry["route"] for entry in body["routes"]}


def test_table_size_is_bounded_by_max_routes_for_any_input():
    """The documented invariant: method and path cardinality cannot grow the table."""
    metrics = _recorder(max_routes=8)

    for index in range(5000):
        metrics.record(
            method=f"M{index}",
            route=None,
            status_code=404,
            duration_ms=1.0,
            cache_hit=False,
        )
        # No client value can reach `route` (it is a template or None), but a
        # route explosion is simulated anyway to bound path cardinality too.
        metrics.record(
            method="GET",
            route=f"/api/v1/r{index}",
            status_code=200,
            duration_ms=1.0,
            cache_hit=False,
        )

    body = metrics.snapshot()
    labels = {(entry["method"], entry["route"]) for entry in body["routes"]}

    assert len(body["routes"]) <= 8
    assert body["requests"]["total"] == 10000
    assert sum(entry["count"] for entry in body["routes"]) == 10000
    # No client text anywhere: every label is a known method or a reserved one.
    assert {method for method, _route in labels} <= KNOWN_METHODS | {OTHER_METHOD}
    assert len([key for key in labels if key == SINK_KEY]) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GET", "GET"),
        ("HEAD", "HEAD"),
        ("POST", "POST"),
        ("OPTIONS", "OPTIONS"),
        # Not in the fixed set: never echoed back to the payload.
        ("get", OTHER_METHOD),
        ("TRACE", OTHER_METHOD),
        ("M4711", OTHER_METHOD),
        ("GET ", OTHER_METHOD),
        ("", OTHER_METHOD),
    ],
)
def test_normalize_method(raw: str, expected: str):
    assert normalize_method(raw) == expected


def test_error_count_counts_5xx_only():
    metrics = _recorder()
    _hit(metrics, "/api/v1/x", 1.0, status=404)
    _hit(metrics, "/api/v1/x", 1.0, status=500)

    entry = _route(metrics.snapshot(), "GET", "/api/v1/x")

    assert entry["count"] == 2
    assert entry["error_count"] == 1


# --- the recorder never breaks a request ---------------------------------------


async def test_broken_recorder_still_serves_200(app, client, monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("metrics are broken")

    monkeypatch.setattr(app.state.metrics, "record", boom)

    response = await client.get("/api/v1/cities")

    assert response.status_code == 200
    assert response.json()
    assert (await client.get("/api/v1/metrics")).status_code == 200


async def test_unhandled_exception_is_counted_as_a_5xx():
    recorder = _recorder()

    async def explode(scope, receive, send):
        raise RuntimeError("kaboom")

    async def ignore(*_args):
        return None

    middleware = RequestContextMiddleware(explode, metrics=recorder, route_prefixes=PREFIXES)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/cities",
        "headers": [],
        "route": SimpleNamespace(path="/cities"),
    }

    with pytest.raises(RuntimeError):
        await middleware(scope, ignore, ignore)

    body = recorder.snapshot()
    assert body["requests"]["errors_5xx"] == 1
    assert body["requests"]["by_status_class"] == {"5xx": 1}
    assert _route(body, "GET", "/api/v1/cities")["error_count"] == 1


async def test_non_http_scope_is_ignored():
    recorder = _recorder()
    seen: list[str] = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    async def ignore(*_args):
        return None

    middleware = RequestContextMiddleware(inner, metrics=recorder, route_prefixes=PREFIXES)

    await middleware({"type": "websocket", "path": "/api/v1/ws"}, ignore, ignore)

    assert seen == ["websocket"]
    assert recorder.snapshot()["requests"]["total"] == 0


# --- the endpoint --------------------------------------------------------------


async def test_metrics_is_served_on_both_prefixes_with_the_documented_shape(client):
    await client.get("/api/v1/health")

    response = await client.get("/api/v1/metrics")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert set(body) == {"started_at", "uptime_seconds", "requests", "routes", "slo"}
    assert set(body["requests"]) == {
        "total",
        "by_status_class",
        "errors_5xx",
        "cache_hits",
        "cache_hit_rate",
    }
    assert set(body["slo"]) == {"latency_p95_ms_target", "p95_ms", "met"}
    assert body["slo"]["latency_p95_ms_target"] == 2000
    assert body["started_at"].endswith("Z")
    assert body["uptime_seconds"] >= 0
    route_keys = {
        "method",
        "route",
        "count",
        "error_count",
        "cache_hit_rate",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
    }
    for entry in body["routes"]:
        assert set(entry) == route_keys
    counts = [entry["count"] for entry in body["routes"]]
    assert counts == sorted(counts, reverse=True)
    # The totals are the sum of the table: nothing is counted twice.
    assert body["requests"]["total"] == sum(counts)
    assert _route(body, "GET", "/api/v1/health")["count"] == 1
    assert body["requests"]["by_status_class"] == {"2xx": 1}

    legacy = await client.get("/api/metrics")

    assert legacy.status_code == 200
    assert set(legacy.json()) == set(body)
    # /metrics counts itself: the request reading the payload is counted once its
    # own response is complete, so it shows up from the next call on.
    assert _route(legacy.json(), "GET", "/api/v1/metrics")["count"] == 1


async def test_city_paths_collapse_into_one_template_entry(client):
    assert (await client.get("/api/v1/weather/history/Madrid")).status_code == 200
    # Alicante is a registered station with no fake rows: a *routed* 404, so it
    # keeps the template instead of landing in <unmatched>.
    assert (await client.get("/api/v1/weather/history/Alicante")).status_code == 404

    response = await client.get("/api/v1/metrics")

    entry = _route(response.json(), "GET", "/api/v1/weather/history/{city}")
    assert entry["count"] == 2
    assert "Madrid" not in response.text
    assert "Alicante" not in response.text


async def test_unmatched_request_lands_in_the_unmatched_bucket(client):
    assert (await client.get("/api/v1/nope")).status_code == 404

    body = await _metrics_body(client)

    entry = _route(body, "GET", UNMATCHED_ROUTE)
    assert entry["count"] == 1
    assert entry["error_count"] == 0  # a 404 is a client error, not an outage
    assert body["requests"]["by_status_class"]["4xx"] == 1
    assert "nope" not in json.dumps(body)


async def test_cors_preflight_is_counted_as_unmatched(client):
    response = await client.options(
        "/api/v1/cities",
        headers={"origin": "http://localhost:5173", "access-control-request-method": "GET"},
    )
    assert response.status_code == 200

    body = await _metrics_body(client)

    # The CORS middleware answers a preflight without reaching a route, so there
    # is no template to key it on.
    assert _route(body, "OPTIONS", UNMATCHED_ROUTE)["count"] == 1


async def test_get_head_and_post_stay_distinct(client):
    assert (await client.get("/api/v1/health")).status_code == 200
    # Neither HEAD nor POST is served on this route (405), but both are known
    # methods, so each keeps its own row instead of folding into <other>.
    assert (await client.head("/api/v1/health")).status_code == 405
    assert (await client.post("/api/v1/health")).status_code == 405

    body = await _metrics_body(client)

    methods = {entry["method"] for entry in body["routes"] if entry["route"] == "/api/v1/health"}
    assert methods == {"GET", "HEAD", "POST"}


async def test_crafted_method_on_a_matched_route_folds_into_other(client):
    assert (await client.get("/api/v1/health")).status_code == 200
    for _ in range(3):
        assert (await client.request("m9", "/api/v1/health")).status_code == 405

    body = await _metrics_body(client)

    # The real GET entry keeps its row; the crafted methods share one <other> row.
    assert _route(body, "GET", "/api/v1/health")["count"] == 1
    assert _route(body, OTHER_METHOD, "/api/v1/health")["count"] == 3
    assert "M9" not in json.dumps(body)


async def test_500_crafted_methods_stay_bounded_and_echo_no_client_text(client):
    for index in range(500):
        assert (await client.request(f"m{index}", "/api/v1/nope")).status_code == 404

    response = await client.get("/api/v1/metrics")
    body = response.json()

    assert len(body["routes"]) <= MAX_ROUTES
    assert _route(body, OTHER_METHOD, UNMATCHED_ROUTE)["count"] == 500
    # The whole state is one entry: bounded table, tiny payload, and not one of
    # the 500 crafted tokens survives into the body.
    assert len(body["routes"]) == 1
    assert len(response.content) < 4096
    for token in (b"M0", b"M250", b"M499"):
        assert token not in response.content


async def test_method_mismatch_keeps_the_route_template(client):
    assert (await client.post("/api/v1/health")).status_code == 405

    body = await _metrics_body(client)

    assert _route(body, "POST", "/api/v1/health")["count"] == 1
    assert UNMATCHED_ROUTE not in {entry["route"] for entry in body["routes"]}


async def test_query_string_and_request_id_are_never_recorded(client):
    response = await client.get(
        "/api/v1/weather/history/Madrid",
        params={"token": "SECRET-abc123"},
        headers={"x-request-id": "req-secret-42"},
    )
    assert response.status_code == 200

    payload = json.dumps(await _metrics_body(client))

    assert "SECRET-abc123" not in payload
    assert "req-secret-42" not in payload
    assert "token=" not in payload


async def test_cache_hit_rate_is_recorded_for_a_cached_route(client):
    await client.get("/api/v1/cities")
    await client.get("/api/v1/cities")

    body = await _metrics_body(client)

    entry = _route(body, "GET", "/api/v1/cities")
    assert entry["count"] == 2
    assert entry["cache_hit_rate"] == 0.5
    assert body["requests"]["cache_hits"] == 1


async def test_post_requests_are_their_own_route_entry(client):
    response = await client.post("/api/v1/refresh", json={"city": "Madrid"})
    assert response.status_code == 200

    body = await _metrics_body(client)

    assert _route(body, "POST", "/api/v1/refresh")["count"] == 1
    assert body["requests"]["by_status_class"] == {"2xx": 1}
