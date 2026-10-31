"""The stable error envelope for every 4xx/5xx body.

Shape, for every failure that reaches the handlers below::

    {"error": {"code": "<snake_case>", "message": "<human readable>",
               "request_id": "<id>"}}

``request_id`` is the same value returned in the ``X-Request-Id`` header and
written to the access-log line, so a user-reported failure maps to exactly one
log entry. The HTTP status is never rewritten.

What is deliberately not in the body: stack traces, driver/exceptions messages,
connection strings, and — for validation errors — the offending ``input`` and
the pydantic ``ctx``, both of which echo submitted values. A client that sends a
secret in a query string must not see it reflected back (or logged).

One documented exception to the envelope: ``GET /health/ready`` keeps returning
``{"status": "unavailable", "database": "<db>"}`` with a 503. It is an
operational probe, not an API resource: an operator needs the failing database
name, and nothing outside this repository parses that body.
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import REQUEST_ID_HEADER, new_request_id

logger = logging.getLogger(__name__)

#: HTTP status -> stable machine code. Frozen: clients may branch on these, so
#: an existing mapping is never reworded (additions are fine).
STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    406: "not_acceptable",
    408: "request_timeout",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    # Both FastAPI's own validation failures and our explicit 422s (bad query
    # windows, out-of-range cursor) share one code: the client fix is the same.
    422: "validation_error",
    429: "too_many_requests",
    500: "internal_error",
    501: "not_implemented",
    502: "bad_gateway",
    503: "service_unavailable",
    504: "gateway_timeout",
}

#: Fallback for a status with no explicit mapping.
UNMAPPED_STATUS_CODE = "http_error"

INTERNAL_ERROR_MESSAGE = "Internal server error"

#: Bound on the 422 detail: a hostile payload must not turn a validation error
#: into a multi-megabyte response.
MAX_VALIDATION_FIELDS = 20
MAX_VALIDATION_MESSAGE_LENGTH = 200


class APIError(StarletteHTTPException):
    """An ``HTTPException`` that carries an explicit stable envelope code.

    The status -> code map below covers the generic failures; routes whose
    contract names a more specific code (``city_not_found``,
    ``city_writes_disabled``, ...) raise this instead of a bare HTTPException.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code


def request_id_of(request: Request) -> str:
    """The id assigned by ``RequestContextMiddleware``, or a fresh one.

    The fallback matters for a request that never reached the middleware (an
    error raised while building the middleware stack itself).
    """
    return getattr(request.state, "request_id", None) or new_request_id()


def error_body(
    code: str,
    message: str,
    request_id: str,
    fields: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if fields is not None:
        error["fields"] = fields
    return {"error": error}


def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    *,
    fields: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_body(code, message, request_id, fields),
        headers={REQUEST_ID_HEADER: request_id, **(headers or {})},
    )


def _default_message(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Request failed"


def _validation_fields(exc: RequestValidationError) -> list[dict[str, Any]]:
    """pydantic errors reduced to location/type/message — no ``input``, no ``ctx``."""
    fields: list[dict[str, Any]] = []
    for error in exc.errors()[:MAX_VALIDATION_FIELDS]:
        fields.append(
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": str(error.get("msg", ""))[:MAX_VALIDATION_MESSAGE_LENGTH],
                "type": str(error.get("type", "")),
            }
        )
    return fields


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Any ``HTTPException`` raised by a handler or by Starlette's router."""
    status_code = exc.status_code
    # ``APIError`` names its own code; anything else falls back to the status map.
    code = getattr(exc, "code", None) or STATUS_CODES.get(status_code, UNMAPPED_STATUS_CODE)
    # Our own handlers pass human-readable strings; anything else (a status
    # raised by the framework) falls back to the status phrase.
    message = (
        exc.detail if isinstance(exc.detail, str) and exc.detail else _default_message(status_code)
    )
    return error_response(
        status_code,
        code,
        message,
        request_id_of(request),
        # 405 carries Allow, 401 may carry WWW-Authenticate: never drop them.
        headers=dict(exc.headers) if exc.headers else None,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return error_response(
        422,
        STATUS_CODES[422],
        "Request validation failed",
        request_id_of(request),
        fields=_validation_fields(exc),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """500 for anything unhandled: generic message, no detail, no traceback.

    No logging here on purpose — ``RequestContextMiddleware`` already wrote the
    one access line for this request.
    """
    return error_response(500, STATUS_CODES[500], INTERNAL_ERROR_MESSAGE, request_id_of(request))


def register_error_handlers(app: FastAPI) -> None:
    """Install the envelope handlers over FastAPI's defaults."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
