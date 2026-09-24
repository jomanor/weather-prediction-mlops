"""Logging setup and the per-request context middleware."""

import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
REQUEST_LOGGER = logging.getLogger("app.request")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT)
    # uvicorn already logs every request; keep our structured line only.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, echo it back, and log one structured line per request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            REQUEST_LOGGER.exception(
                "method=%s path=%s status=500 duration_ms=%.1f request_id=%s",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        REQUEST_LOGGER.info(
            "method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response
