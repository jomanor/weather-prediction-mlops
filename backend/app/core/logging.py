"""Structured logging, request ids, and the single per-request access log.

``RequestContextMiddleware`` writes exactly one JSON object per request to
stdout and is the only component that logs a request:

    {"ts": "...Z", "level": "info", "request_id": "...", "method": "GET",
     "path": "/api/v1/cities", "status_code": 200, "duration_ms": 3.2,
     "cache_hit": false}

Deliberate omissions, not oversights:

* **No query string** — ``scope["path"]`` only. Query values are user input and
  can carry secrets (tokens, emails); they are never logged.
* **No client address** — an IP is personal data under the GDPR, and behind the
  Netlify edge proxy it would be a proxy hop rather than the caller anyway.
* **No traceback in the access line** — the exception is re-raised so uvicorn
  logs the stack trace; a traceback spliced into the line would make it
  unparseable as JSON. The response body never carries one (``app.core.errors``).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
REQUEST_LOGGER = logging.getLogger("app.request")

#: Response header carrying the correlation id (also in every error envelope).
REQUEST_ID_HEADER = "X-Request-Id"
#: Inbound ids longer than this are replaced rather than echoed.
REQUEST_ID_MAX_LENGTH = 64

#: An inbound ``X-Request-Id`` is only echoed when it matches: the value is
#: reflected into a response header and into error bodies, so control
#: characters, spaces and separators must not survive (header/JSON injection).
REQUEST_ID_PATTERN = re.compile(rf"^[A-Za-z0-9._-]{{1,{REQUEST_ID_MAX_LENGTH}}}$")


def configure_logging(level: str = "INFO") -> None:
    """Root config, plus a raw-line stdout handler for the access log.

    The access logger gets its own handler and stops propagating: the only
    thing it ever emits is a JSON object, which must not be decorated with the
    human-readable ``LOG_FORMAT`` prefix.
    """
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT)
    # uvicorn already logs every request; keep our structured line only.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    REQUEST_LOGGER.handlers = [handler]
    REQUEST_LOGGER.propagate = False


def new_request_id() -> str:
    """A short id for a request that arrived without a usable one."""
    return uuid4().hex[:16]


def request_id_from(value: str | None) -> str:
    """Echo a valid inbound id, otherwise mint one."""
    if value and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return new_request_id()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _log_request(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    started: float,
    cache_hit: bool,
) -> None:
    level = "error" if status_code >= 500 else "info"
    payload = {
        "ts": _timestamp(),
        "level": level,
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "cache_hit": cache_hit,
    }
    REQUEST_LOGGER.log(
        logging.ERROR if status_code >= 500 else logging.INFO,
        json.dumps(payload, separators=(",", ":")),
    )


class RequestContextMiddleware:
    """Assign/echo the request id and emit the access log line.

    Plain ASGI (not ``BaseHTTPMiddleware``) so the same ``scope`` — and its
    ``state`` dict, which carries the id to the error handlers and the TTL-cache
    hit flag back out — is reachable on both sides of the call.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = request_id_from(Headers(scope=scope).get("x-request-id"))
        # Same dict FastAPI exposes as ``request.state``; written before the
        # call so an error handler reached from outside this middleware still
        # finds the id.
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["cache_hit"] = False

        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Every response carries it, errors included: the 5xx path goes
                # through the exception handler, which adds the same header.
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            _log_request(
                request_id=request_id,
                method=scope["method"],
                path=scope["path"],
                status_code=500,
                started=started,
                cache_hit=bool(state.get("cache_hit")),
            )
            # Re-raised: uvicorn/ServerErrorMiddleware owns the traceback, and
            # the registered ``Exception`` handler builds the 500 envelope.
            raise

        _log_request(
            request_id=request_id,
            method=scope["method"],
            path=scope["path"],
            status_code=status_code,
            started=started,
            cache_hit=bool(state.get("cache_hit")),
        )
