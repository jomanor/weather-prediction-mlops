"""The single structured access-log line emitted per request.

``RequestContextMiddleware`` is the only component that logs a request, and it
writes exactly one JSON object to the ``app.request`` logger. These tests pin
the line's fields, the deliberate absence of the query string (user input, may
carry secrets), the cache-hit flag, and that an inbound ``X-Request-Id`` is only
echoed when it is safe.
"""

import json
import logging

import pytest

from app.core.logging import REQUEST_LOGGER

FIELDS = {
    "ts",
    "level",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "cache_hit",
}


class _ListHandler(logging.Handler):
    """Collect the raw log lines the middleware emits."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def access_lines(app):
    """Raw ``app.request`` log lines for one app instance.

    ``configure_logging`` runs inside ``create_app`` (the ``app`` fixture), so
    the handler is attached afterwards. It goes on the logger itself because the
    access logger does not propagate to root.
    """
    handler = _ListHandler()
    REQUEST_LOGGER.addHandler(handler)
    try:
        yield handler.lines
    finally:
        REQUEST_LOGGER.removeHandler(handler)


async def test_one_json_line_per_request_with_the_documented_fields(client, access_lines):
    response = await client.get("/api/v1/cities")

    assert response.status_code == 200
    assert len(access_lines) == 1
    line = json.loads(access_lines[0])
    assert set(line) == FIELDS
    assert line["level"] == "info"
    assert line["request_id"] == response.headers["x-request-id"]
    assert line["method"] == "GET"
    assert line["path"] == "/api/v1/cities"
    assert line["status_code"] == 200
    assert line["duration_ms"] >= 0
    assert line["cache_hit"] is False
    assert line["ts"].endswith("Z")


async def test_query_string_is_absent_from_the_logged_path(client, access_lines):
    response = await client.get("/api/v1/cities", params={"token": "SECRET-abc123"})

    assert response.status_code == 200
    assert len(access_lines) == 1
    assert json.loads(access_lines[0])["path"] == "/api/v1/cities"
    assert "SECRET-abc123" not in access_lines[0]
    assert "token=" not in access_lines[0]


async def test_cached_second_request_logs_cache_hit_true(client, access_lines):
    await client.get("/api/v1/cities")
    await client.get("/api/v1/cities")

    assert len(access_lines) == 2
    assert [json.loads(line)["cache_hit"] for line in access_lines] == [False, True]


async def test_valid_inbound_request_id_is_echoed_and_logged(client, access_lines):
    response = await client.get("/api/v1/health", headers={"x-request-id": "req-abc.123"})

    assert response.headers["x-request-id"] == "req-abc.123"
    assert json.loads(access_lines[0])["request_id"] == "req-abc.123"


async def test_invalid_inbound_request_id_is_replaced_not_reflected(client, access_lines):
    invalid = "bad id!\r\nx-injected: 1"
    response = await client.get("/api/v1/health", headers={"x-request-id": invalid})

    echoed = response.headers["x-request-id"]
    assert echoed != invalid
    assert "x-injected" not in echoed
    assert json.loads(access_lines[0])["request_id"] == echoed
    assert invalid not in access_lines[0]
