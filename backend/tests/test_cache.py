"""TTL cache and ETag helpers (single-process, payload-derived validators)."""

import asyncio

import pytest
from starlette.requests import Request
from starlette.responses import Response

from app.core.cache import TTLCache, cached, etag_for
from app.schemas.city import City

MADRID = City(name="Madrid", latitude=40.4168, longitude=-3.7038)
ALICANTE = City(name="Alicante", latitude=38.3452, longitude=-0.481)


def _request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "headers": []})


async def _value() -> City:
    return MADRID


def test_set_then_get_returns_value_and_etag():
    cache = TTLCache()
    cache.set("k", MADRID, "etag-1", 60)

    assert cache.get("k") == (MADRID, "etag-1")


def test_expired_entry_is_dropped(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr("app.core.cache.time.monotonic", lambda: clock["now"])

    cache = TTLCache()
    cache.set("k", MADRID, "etag-1", 60)

    clock["now"] = 159.0
    assert cache.get("k") == (MADRID, "etag-1")  # still fresh

    clock["now"] = 161.0
    assert cache.get("k") is None  # expired and evicted
    assert cache.get("k") is None  # missing, not stale


def test_etag_changes_with_payload_and_matches_for_lists():
    madrid_etag = etag_for(MADRID)
    assert etag_for(MADRID) == madrid_etag  # deterministic
    assert etag_for(ALICANTE) != madrid_etag

    list_etag = etag_for([MADRID, ALICANTE])
    assert etag_for([MADRID, ALICANTE]) == list_etag
    assert etag_for([ALICANTE, MADRID]) != list_etag  # order-sensitive body
    assert list_etag.startswith('"') and list_etag.endswith('"')


# ---------------------------------------------------------------------------
# Single-flight coalescing: concurrent cold misses share one loader execution
# ---------------------------------------------------------------------------


async def test_cached_coalesces_concurrent_cold_calls():
    cache = TTLCache()
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def loader() -> City:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return MADRID

    async def call() -> City | Response:
        return await cached(_request(), Response(), cache, "k", 60, loader)

    tasks = [asyncio.create_task(call()) for _ in range(5)]
    await started.wait()
    # One event-loop turn lets every waiter reach the shared in-flight task;
    # the loader only resumes on the next turn, so no wall-clock sleep is needed.
    await asyncio.sleep(0)
    assert calls == 1

    release.set()
    results = await asyncio.gather(*tasks)

    assert calls == 1  # five cold callers, one loader execution
    assert all(result is MADRID for result in results)

    # The value was stored, so a later call is a hit, not a re-run.
    assert await call() is MADRID
    assert calls == 1


async def test_cached_failing_loader_errors_waiters_and_does_not_poison_key():
    cache = TTLCache()
    attempts = 0

    async def failing() -> City:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    async def call(loader) -> City | Response:
        return await cached(_request(), Response(), cache, "k", 60, loader)

    results = await asyncio.gather(*(call(failing) for _ in range(3)), return_exceptions=True)

    assert attempts == 1  # the three callers shared one failing execution
    assert all(isinstance(result, RuntimeError) for result in results)

    # ``finally`` unregistered the key: a later call re-runs and succeeds.
    assert await call(_value) is MADRID
    assert attempts == 1


async def test_cached_waiter_cancellation_keeps_the_shared_load_alive():
    cache = TTLCache()
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow() -> City:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return MADRID

    async def call() -> City | Response:
        return await cached(_request(), Response(), cache, "k", 60, slow)

    first = asyncio.create_task(call())
    second = asyncio.create_task(call())
    await started.wait()
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second

    release.set()
    assert await first is MADRID
    assert calls == 1
