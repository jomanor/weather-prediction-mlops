"""TTL cache and ETag helpers (single-process, payload-derived validators)."""

from app.core.cache import TTLCache, etag_for
from app.schemas.city import City

MADRID = City(name="Madrid", latitude=40.4168, longitude=-3.7038)
ALICANTE = City(name="Alicante", latitude=38.3452, longitude=-0.481)


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
