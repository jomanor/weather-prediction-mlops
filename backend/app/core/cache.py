"""Single-process TTL cache + HTTP validators (ETag / Cache-Control / 304).

The API runs on exactly one Render instance, so there is no second process to
coordinate with: the cache is a plain ``dict`` with an expiry epoch per entry
and no eviction, locking or invalidation bus. Explicitness matters here — the
absence of thread-safety is deliberate, not an oversight. FastAPI executes all
handlers for a given app on a single event loop, so no lock is needed; if the
deployment ever grows a second worker or a thread pool, this has to change.

Cold misses are single-flighted per key: concurrent callers of ``cached()`` for
the same key share one loader task instead of each hitting Mongo. The in-flight
map lives on the ``TTLCache`` instance (not the module) so separate apps/tests
cannot leak work into each other.

ETags are derived from the serialized payload. A cache hit therefore never
touches Mongo to recompute its validator.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from fastapi import Request, Response
from pydantic import BaseModel

T = TypeVar("T")


@dataclass
class _Entry:
    expires_at: float
    value: Any
    etag: str


class TTLCache:
    """dict of key -> (expiry_epoch, value, etag). Single-process, no coordination."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        #: key -> running cold-miss loader task, so concurrent callers coalesce.
        self._inflight: dict[str, asyncio.Task[Any]] = {}

    def get(self, key: str) -> tuple[Any, str] | None:
        """Return ``(value, etag)`` when fresh, else None (expired entries are dropped)."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            del self._entries[key]
            return None
        return entry.value, entry.etag

    def set(self, key: str, value: Any, etag: str, ttl_seconds: float) -> None:
        self._entries[key] = _Entry(time.monotonic() + ttl_seconds, value, etag)

    def clear(self) -> None:
        self._entries.clear()


def _serialize(payload: Any) -> str:
    """Deterministic JSON for the supported payload shapes (pydantic / list / dict)."""
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    if isinstance(payload, (list, tuple)):
        return "[" + ",".join(_serialize(item) for item in payload) + "]"
    return json.dumps(payload, default=str, sort_keys=True)


def etag_for(payload: Any) -> str:
    """Strong ETag: sha256 of the serialized payload, so it changes with the data."""
    digest = hashlib.sha256(_serialize(payload).encode("utf-8")).hexdigest()
    return f'"{digest}"'


def _if_none_match(request: Request, etag: str) -> bool:
    """RFC 7232 §3.2: ``*`` matches any representation; otherwise any listed tag.

    The header may carry a comma-separated list of validators, and the client may
    send weak tags (``W/"..."``); If-None-Match uses the weak comparison, so the
    ``W/`` prefix is stripped before comparing.
    """
    value = request.headers.get("if-none-match")
    if not value:
        return False
    for candidate in value.split(","):
        candidate = candidate.strip()
        if candidate == "*":
            return True
        if candidate.startswith("W/"):
            candidate = candidate[2:].strip()
        if candidate == etag:
            return True
    return False


def cache_headers(ttl_seconds: int, etag: str) -> dict[str, str]:
    return {
        "ETag": etag,
        "Cache-Control": (f"public, max-age={ttl_seconds}, stale-while-revalidate={ttl_seconds}"),
    }


def _consume_result(task: asyncio.Task[Any]) -> None:
    """Retrieve an otherwise-unobserved loader exception (no asyncio warning)."""
    if not task.cancelled():
        task.exception()


async def _run_loader(
    cache: TTLCache,
    key: str,
    ttl_seconds: int,
    loader: Callable[[], Awaitable[T]],
) -> tuple[T, str]:
    """The single runner for a cold miss: run, store in the TTL cache, unregister."""
    try:
        value = await loader()
        etag = etag_for(value)
        cache.set(key, value, etag, ttl_seconds)
        return value, etag
    finally:
        # Always unregister, success or failure: a raising loader must not
        # poison the key for later callers.
        cache._inflight.pop(key, None)


async def _load_once(
    cache: TTLCache,
    key: str,
    ttl_seconds: int,
    loader: Callable[[], Awaitable[T]],
) -> tuple[T, str]:
    """Return the in-flight cold-miss loader for ``key``, or start one.

    Callers await a ``shield``ed task, so a waiter being cancelled never cancels
    the shared computation for the others.
    """
    task = cache._inflight.get(key)
    if task is None:
        task = asyncio.ensure_future(_run_loader(cache, key, ttl_seconds, loader))
        task.add_done_callback(_consume_result)
        cache._inflight[key] = task
    return await asyncio.shield(task)


async def cached(
    request: Request,
    response: Response,
    cache: TTLCache,
    key: str,
    ttl_seconds: int,
    loader: Callable[[], Awaitable[T]],
) -> T | Response:
    """Serve ``loader`` through ``cache`` with ETag/Cache-Control and If-None-Match -> 304.

    On a hit the loader (and therefore Mongo) is not called: the ETag comes from
    the cached payload. A cold miss is single-flighted per key: concurrent
    callers share one loader execution and all get its value. Returns a bare
    ``Response(304)`` on a validator match, otherwise the value, with headers
    written onto the injected ``response``.
    """
    hit = cache.get(key)
    if hit is None:
        value, etag = await _load_once(cache, key, ttl_seconds, loader)
    else:
        value, etag = hit

    headers = cache_headers(ttl_seconds, etag)
    if _if_none_match(request, etag):
        return Response(status_code=304, headers=headers)

    response.headers.update(headers)
    return value


def get_cache(request: Request) -> TTLCache:
    """The process-wide cache, stored on ``app.state`` by ``create_app``."""
    return request.app.state.ttl_cache
