"""TTL cache and ETag helpers (single-process, payload-derived validators)."""

import asyncio

import pytest
from starlette.requests import Request
from starlette.responses import Response

from app.core.cache import TTLCache, cached, etag_for
from app.schemas.city import City

MADRID = City(name="Madrid", latitude=40.4168, longitude=-3.7038)
ALICANTE = City(name="Alicante", latitude=38.3452, longitude=-0.481)
SEVILLA = City(name="Sevilla", latitude=37.3891, longitude=-5.9845)


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


# ---------------------------------------------------------------------------
# LRU bound: part of every key is caller-controlled, so the map must never grow
# past the configured maximum and a ``get`` must refresh recency.
# ---------------------------------------------------------------------------


def test_map_never_exceeds_the_configured_max_entries():
    cache = TTLCache(max_entries=2)

    for index in range(10):
        cache.set(f"k{index}", MADRID, f"etag-{index}", 60)
        assert len(cache) <= cache.max_entries == 2

    assert len(cache) == 2

    # The two newest survive; the rest were evicted from the front.
    assert cache.get("k8") == (MADRID, "etag-8")
    assert cache.get("k9") == (MADRID, "etag-9")
    assert cache.get("k7") is None


def test_get_marks_an_entry_as_most_recently_used():
    cache = TTLCache(max_entries=2)
    cache.set("a", MADRID, "etag-a", 60)
    cache.set("b", ALICANTE, "etag-b", 60)

    assert cache.get("a") == (MADRID, "etag-a")  # now "a" is the newest
    cache.set("c", MADRID, "etag-c", 60)

    assert cache.get("a") == (MADRID, "etag-a")
    assert cache.get("c") == (MADRID, "etag-c")
    assert cache.get("b") is None  # least recently used, evicted


# ---------------------------------------------------------------------------
# Invalidation: ``delete`` must also stop a racing cold load from re-caching
# ---------------------------------------------------------------------------


def test_delete_drops_the_entry_and_a_later_get_misses():
    cache = TTLCache()
    cache.set("cities:list", [MADRID], "etag-1", 60)
    assert cache.get("cities:list") is not None

    cache.delete("cities:list")

    assert cache.get("cities:list") is None
    assert len(cache) == 0
    cache.delete("never-cached")  # deleting an absent key is a no-op, not an error


async def test_delete_while_a_cold_load_is_in_flight_does_not_re_cache_it():
    """A load that started before the write must not hide the write for a whole TTL."""
    cache = TTLCache()
    started = asyncio.Event()
    release = asyncio.Event()
    loads: list[str] = []

    async def stale_loader() -> list[City]:
        loads.append("stale")
        started.set()
        await release.wait()
        return [MADRID]

    async def fresh_loader() -> list[City]:
        loads.append("fresh")
        return [MADRID, ALICANTE]

    in_flight = asyncio.create_task(
        cached(_request(), Response(), cache, "cities:list", 60, stale_loader)
    )
    await started.wait()

    # The registry write lands while the pre-write load is still running.
    cache.delete("cities:list")
    release.set()
    assert await in_flight == [MADRID]  # the caller already in flight still gets its value

    # The discarded result was not stored: the next request reloads the new state.
    assert await cached(_request(), Response(), cache, "cities:list", 60, fresh_loader) == [
        MADRID,
        ALICANTE,
    ]
    assert loads == ["stale", "fresh"]
    assert cache.get("cities:list") is not None


async def test_delete_unregisters_the_in_flight_load_so_the_next_call_reloads():
    cache = TTLCache()
    started = asyncio.Event()
    release = asyncio.Event()
    loads: list[str] = []

    async def slow() -> City:
        loads.append("slow")
        started.set()
        await release.wait()
        return MADRID

    async def quick() -> City:
        loads.append("quick")
        return ALICANTE

    first = asyncio.create_task(cached(_request(), Response(), cache, "k", 60, slow))
    await started.wait()
    cache.delete("k")

    # Not coalesced onto the pre-delete load: a new loader starts immediately.
    assert await cached(_request(), Response(), cache, "k", 60, quick) is ALICANTE
    release.set()
    assert await first is MADRID
    assert loads == ["slow", "quick"]


async def test_an_older_finishing_load_cannot_unregister_the_newer_in_flight_one():
    """The finally-guard in ``_run_loader``: an old load must not steal the new one's slot.

    Without the identity check, the older task's ``finally`` pops the key that the
    newer, still-running load registered: the newer load then refuses to store its
    value (it is no longer the registered load) and the next request loads a third
    time instead of hitting it.
    """
    cache = TTLCache()
    old_started, old_release = asyncio.Event(), asyncio.Event()
    new_started, new_release = asyncio.Event(), asyncio.Event()
    loads: list[str] = []

    async def old_loader() -> list[City]:
        loads.append("old")
        old_started.set()
        await old_release.wait()
        return [MADRID]

    async def new_loader() -> list[City]:
        loads.append("new")
        new_started.set()
        await new_release.wait()
        return [ALICANTE]

    async def third_loader() -> list[City]:
        loads.append("third")
        return [SEVILLA]

    older = asyncio.create_task(
        cached(_request(), Response(), cache, "cities:list", 60, old_loader)
    )
    await old_started.wait()
    cache.delete("cities:list")
    newer = asyncio.create_task(
        cached(_request(), Response(), cache, "cities:list", 60, new_loader)
    )
    await new_started.wait()

    old_release.set()  # the older load finishes first, while the newer one is still running
    assert await older == [MADRID]
    new_release.set()
    assert await newer == [ALICANTE]

    # The newer value was stored, so this is a hit and no third load runs.
    assert await cached(_request(), Response(), cache, "cities:list", 60, third_loader) == [
        ALICANTE
    ]
    assert loads == ["old", "new"]
