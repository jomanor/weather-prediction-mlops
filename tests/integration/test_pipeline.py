"""
test_pipeline.py
================
Integration tests for the ingestion pipeline that writes ``raw_weather``.

The live data path is ``scripts/ingest_weather.py``: Open-Meteo JSON -> ``build_document``
-> ``raw_weather`` upsert. These tests drive the real ``ingest()`` function against
mongomock with a fake HTTP session, so the document shape, the ``_id`` deduplication
contract and the exit codes stay pinned to the production writer.
"""

import os
import sys
from unittest.mock import patch

import mongomock

# Ensure scripts/ and the test fixtures are discoverable
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
TESTS_DIR = os.path.join(os.path.dirname(__file__), "..")
for d in (SCRIPTS_DIR, TESTS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import ingest_weather  # noqa: E402
from conftest import make_open_meteo_message  # noqa: E402


class FakeResponse:
    """Minimal ``requests.Response`` stand-in."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    """``requests.Session`` stand-in returning a canned payload per station.

    Cities are matched on the ``latitude`` query parameter, which is how
    ``fetch_current`` distinguishes stations.
    """

    def __init__(self, payloads: dict[str, dict]):
        self.payloads = payloads
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url, params=None, timeout=None):
        latitude = (params or {}).get("latitude")
        for city, payload in self.payloads.items():
            if payload["latitude"] == latitude:
                self.calls.append(city)
                if isinstance(payload, Exception):
                    raise payload
                return FakeResponse(payload)
        raise AssertionError(f"unexpected latitude in request: {latitude!r}")


#: ``(latitude, longitude)`` per test station — the registry the writer reads.
REGISTRY = {
    "Madrid": (40.4168, -3.7038),
    "Granada": (37.1773, -3.5986),
}


def _seeded_db(cities=("Madrid", "Granada")) -> mongomock.MongoClient:
    """mongomock client with the station registry the API seeds on startup."""
    client = mongomock.MongoClient()
    client[ingest_weather.DB_NAME][ingest_weather.CITIES_COLLECTION].insert_many(
        [
            {"name": city, "latitude": REGISTRY[city][0], "longitude": REGISTRY[city][1]}
            for city in cities
        ]
    )
    return client


def _run_ingest(client, payloads: dict[str, dict]) -> int:
    """Run the real ``ingest()`` with Mongo and HTTP patched out."""

    def session_factory():
        return FakeSession(payloads)

    with patch.object(ingest_weather, "MongoClient", return_value=client):
        with patch.object(ingest_weather.requests, "Session", session_factory):
            with patch.dict(os.environ, {"MONGO_URI": "mongodb://localhost:27017"}):
                return ingest_weather.ingest()


def _city(name: str, unix_ts: int = 1_750_000_000) -> dict:
    """An Open-Meteo response for *name* carrying the registry coordinates."""
    payload = make_open_meteo_message(name, unix_ts)
    payload["latitude"] = REGISTRY[name][0]
    payload["longitude"] = REGISTRY[name][1]
    return payload


class TestIngestionPipelineIntegration:
    def test_build_document_maps_the_open_meteo_payload(self):
        """Payload -> ``raw_weather`` document, ``_id`` keyed on city + observation."""
        message = make_open_meteo_message("Madrid", 1_750_000_000)
        document = ingest_weather.build_document(message)

        assert document is not None
        assert document["_id"] == "Madrid_1750000000"
        assert document["city"] == "Madrid"
        assert document["timestamp"].utcoffset().total_seconds() == 0
        assert int(document["timestamp"].timestamp()) == 1_750_000_000
        # The payload is stored untouched, so the Spark job's field paths hold.
        assert document["payload"]["current"]["temperature_2m"] == 22.5
        assert document["payload"]["latitude"] == 40.4168

    def test_build_document_rejects_a_payload_without_a_timestamp(self):
        """No ``current.time`` -> None (the caller reports it as a failure)."""
        assert ingest_weather.build_document({"city": "Madrid", "current": {}}) is None
        assert ingest_weather.build_document({"city": "Madrid"}) is None

    def test_raw_document_is_idempotent_per_observation(self):
        """Re-writing the same observation must not duplicate ``raw_weather`` rows."""
        collection = mongomock.MongoClient()[ingest_weather.DB_NAME][ingest_weather.RAW_COLLECTION]
        message = make_open_meteo_message("Madrid", 1_750_000_000)

        first = ingest_weather.build_document(message)
        collection.replace_one({"_id": first["_id"]}, first, upsert=True)
        assert collection.count_documents({}) == 1

        # Exact duplicate (a re-run within the same hour) -> still one row.
        again = ingest_weather.build_document(message)
        collection.replace_one({"_id": again["_id"]}, again, upsert=True)
        assert collection.count_documents({}) == 1

        # A later observation is a new document.
        later = ingest_weather.build_document(make_open_meteo_message("Madrid", 1_750_003_600))
        collection.replace_one({"_id": later["_id"]}, later, upsert=True)
        assert collection.count_documents({}) == 2
        assert collection.find_one({"_id": "Madrid_1750003600"})["city"] == "Madrid"

    def test_ingest_writes_every_registered_station(self):
        """``ingest()`` reads the registry and stores one document per station."""
        client = _seeded_db()
        payloads = {"Madrid": _city("Madrid"), "Granada": _city("Granada")}

        assert _run_ingest(client, payloads) == 0

        raw = client[ingest_weather.DB_NAME][ingest_weather.RAW_COLLECTION]
        assert raw.count_documents({}) == 2
        stored = raw.find_one({"_id": "Granada_1750000000"})
        assert stored["city"] == "Granada"
        assert stored["payload"]["current"]["temperature_2m"] == 22.5

    def test_ingest_is_idempotent_across_runs(self):
        """A second run over the same observations adds no rows."""
        client = _seeded_db()
        payloads = {"Madrid": _city("Madrid"), "Granada": _city("Granada")}

        assert _run_ingest(client, payloads) == 0
        assert _run_ingest(client, payloads) == 0

        raw = client[ingest_weather.DB_NAME][ingest_weather.RAW_COLLECTION]
        assert raw.count_documents({}) == 2

    def test_ingest_reports_a_partial_failure_but_keeps_storing(self):
        """One bad station must not drop the others (partial outage is a warning)."""
        client = _seeded_db()
        payloads = {
            "Madrid": _city("Madrid"),
            # No ``current`` block -> ``build_document`` returns None.
            "Granada": {
                "city": "Granada",
                "latitude": REGISTRY["Granada"][0],
                "longitude": REGISTRY["Granada"][1],
            },
        }

        assert _run_ingest(client, payloads) == 0

        raw = client[ingest_weather.DB_NAME][ingest_weather.RAW_COLLECTION]
        assert raw.count_documents({}) == 1
        assert raw.find_one()["city"] == "Madrid"

    def test_ingest_fails_loudly_when_nothing_can_be_stored(self):
        """A total outage must exit non-zero instead of looking successful."""
        client = _seeded_db(cities=("Madrid",))
        payloads = {
            "Madrid": {
                "city": "Madrid",
                "latitude": REGISTRY["Madrid"][0],
                "longitude": REGISTRY["Madrid"][1],
            }
        }

        assert _run_ingest(client, payloads) == 1

        raw = client[ingest_weather.DB_NAME][ingest_weather.RAW_COLLECTION]
        assert raw.count_documents({}) == 0

    def test_ingest_refuses_to_run_without_a_registry(self):
        """An empty ``cities`` collection is an explicit error, not a silent no-op."""
        client = mongomock.MongoClient()

        with patch.object(ingest_weather, "MongoClient", return_value=client):
            with patch.dict(os.environ, {"MONGO_URI": "mongodb://localhost:27017"}):
                assert ingest_weather.ingest() == 1
