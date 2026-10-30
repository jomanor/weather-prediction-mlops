"""The stable error envelope: shape, request-id correlation, and no input echo.

Every 4xx/5xx body is ``{"error": {"code", "message", "request_id"}}`` and the
``request_id`` is the same value as the ``X-Request-Id`` response header, so a
user-reported failure maps to exactly one log line. The one documented
exception is ``/health/ready``, an operational probe that keeps its own body.
"""

from httpx import ASGITransport, AsyncClient

from tests.fakes import FakeDatabase


def _assert_envelope(response, status_code: int, code: str) -> dict:
    """Assert the envelope shape and the request-id correlation; return the error."""
    assert response.status_code == status_code
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    assert error["request_id"] == response.headers["x-request-id"]
    assert error["request_id"]
    return error


async def test_unknown_path_is_a_not_found_envelope(client):
    response = await client.get("/api/v1/nope")

    _assert_envelope(response, 404, "not_found")


async def test_validation_error_is_an_envelope_without_the_submitted_value(client, write_auth):
    # ``longitude`` is missing, so the whole submitted body is pydantic's
    # ``input``: it must not be reflected back to the caller. The registry write
    # is gated, so the caller is authorised here on purpose: it is the body that
    # must be rejected, not the token.
    response = await client.post(
        "/api/v1/cities",
        headers=write_auth,
        json={"name": "SECRET-abc123", "latitude": 40.4},
    )

    error = _assert_envelope(response, 422, "validation_error")
    assert "SECRET-abc123" not in response.text
    assert any("longitude" in field["loc"] for field in error["fields"])


async def test_unhandled_exception_is_an_internal_error_envelope(app):
    async def boom() -> None:
        raise RuntimeError("SECRET-detail-that-must-not-leak")

    app.add_api_route("/api/v1/boom", boom, methods=["GET"])

    # ``raise_app_exceptions=False`` lets the ASGI stack render the 500 the way
    # a real server would instead of re-raising into the test.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/boom")

    error = _assert_envelope(response, 500, "internal_error")
    assert error["message"] == "Internal server error"
    assert "SECRET-detail-that-must-not-leak" not in response.text


async def test_health_ready_keeps_its_documented_503_body(client, app):
    app.state.db = FakeDatabase(error=RuntimeError("no mongod"))

    response = await client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "weather_db"}
    assert response.headers["x-request-id"]
