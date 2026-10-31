"""Single-process TTL cache + HTTP validators (ETag / Cache-Control / 304).

The API runs on exactly one Render instance, so there is no second process to
coordinate with: the cache is an ``OrderedDict`` with an expiry epoch per entry
and no locking or invalidation bus. Explicitness matters here — the absence of
thread-safety is deliberate, not an oversight. FastAPI executes all handlers for
a given app on a single event loop, so no lock is needed; if the deployment ever
grows a second worker or a thread pool, this has to change.

Eviction is LRU with a hard entry bound. Keys are partly caller-controlled (a
``city`` path segment), so without the bound a client walking names would grow
the process map without limit on a memory-constrained free instance. Both a
``get`` hit and a ``set`` mark an entry as most recently used.

Cold misses are single-flighted per key: concurrent callers of ``cached()`` for
the same key share one loader task instead of each hitting Mongo. The in-flight
map lives on the ``TTLCache`` instance (not the module) so separate apps/tests
cannot leak work into each other. ``delete()`` drops both the cached value and
the registration of a load racing with it, and a loader only stores its result
while it is still the registered load for its key — so an invalidation cannot be
undone by a loader that started before it.

ETags are derived from the serialized payload. A cache hit therefore never
touches Mongo to recompute its validator.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from fastapi import Request, Response
from pydantic import BaseModel

T = TypeVar("T")

#: Entry bound for the process-wide cache. Every cached route keys on a bounded
#: input (window days, a fixed route name) except the per-city ones; 256 covers
#: all of them several times over at the 15-station registry size, so eviction
#: is a safety valve rather than a normal path.
DEFAULT_MAX_ENTRIES = 256


@dataclass
class _Entry:
    expires_at: float
    value: Any
    etag: str


class TTLCache:
    """``OrderedDict`` of key -> (expiry_epoch, value, etag), LRU-bounded.

    Insertion order is recency: the oldest entry is at the front and is the one
    evicted when ``max_entries`` is exceeded.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        #: key -> running cold-miss loader task, so concurrent callers coalesce.
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self._max_entries = max_entries

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> tuple[Any, str] | None:
        """Return ``(value, etag)`` when fresh, else None (expired entries are dropped)."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return entry.value, entry.etag

    def set(self, key: str, value: Any, etag: str, ttl_seconds: float) -> None:
        self._entries[key] = _Entry(time.monotonic() + ttl_seconds, value, etag)
        # Overwriting an existing key keeps its position unless we move it; a
        # fresh load is by definition the most recently used entry.
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def delete(self, key: str) -> None:
        """Drop ``key``: the cached entry and the registration of any cold load for it.

        Used by the registry writes to make ``cities:list`` honest immediately
        instead of up to one TTL later. Unregistering the in-flight load matters
        as much as dropping the value: a load that started *before* the write
        would otherwise store the pre-write payload and hide the write for a full
        TTL (see :func:`_run_loader`, which only stores while it is still the
        registered load). Callers already awaiting that load still receive its
        result; it is just not cached.
        """
        self._entries.pop(key, None)
        self._inflight.pop(key, None)


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
    task = asyncio.current_task()
    try:
        value = await loader()
        etag = etag_for(value)
        # Store only while this task is still the registered load for ``key``:
        # ``TTLCache.delete`` unregisters it, and a load that started before the
        # write which invalidated the key must not re-cache what it replaced.
        if cache._inflight.get(key) is task:
            cache.set(key, value, etag, ttl_seconds)
        return value, etag
    finally:
        # Always unregister, success or failure: a raising loader must not poison
        # the key for later callers. Guarded on identity so this task cannot
        # unregister a newer load that ``delete`` let start in the meantime.
        if cache._inflight.get(key) is task:
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

    Either way ``request.state.cache_hit`` records whether this request was
    served from the cache, which is the value the access-log line reports. A
    304 derived from a cached validator counts as a hit, because no loader ran.
    """
    hit = cache.get(key)
    request.state.cache_hit = hit is not None
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
