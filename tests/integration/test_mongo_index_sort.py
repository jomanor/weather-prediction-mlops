"""Real-Mongo guardrails for the city-prefixed sort rule.

This is **not** a mongomock test. It runs against a real ``mongo:6`` (a CI
service container, or a local ``docker run mongo:6`` via ``MONGO_URI``) because
the behaviour under test is the server's query planner:

* both repo sort patterns — single-city ascending and multi-city ``$in``
  descending — must be index-backed: the winning plan has no blocking ``SORT``
  or ``SORT_MERGE`` stage;
* a ``{timestamp: -1}``-only sort must be a blocking sort (``COLLSCAN`` +
  ``SORT``) and, with the data set sized past the server's in-memory sort
  budget, fail with code 292 when disk use is disabled. That is the failure the
  hard rule exists to prevent on Atlas M0.

The sort budget is **read from the server** (``getParameter``) rather than
hardcoded: MongoDB 4.4+ defaults ``internalQueryMaxBlockingSortMemoryUsageBytes``
to 100 MiB, not the historical 32 MiB. The seed is derived from that live value
so the 292 assertion stays real if the default changes.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import OperationFailure

from tests.conftest import REPO_ROOT, eval_python_constant

#: The app declares its indexes here; the guardrails must check that declaration.
MONGO_SOURCE = REPO_ROOT / "backend" / "app" / "db" / "mongo.py"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "weather_ci_sort"
COLLECTION = "weather_data"

CITIES = ("Madrid", "Alicante", "Sevilla")
BASE_TS = datetime(2026, 1, 1)

#: Read at fixture start; server default on MongoDB 4.4+ is 100 MiB.
SORT_LIMIT_PARAM = "internalQueryMaxBlockingSortMemoryUsageBytes"
SORT_LIMIT_FALLBACK_BYTES = 100 * 1024 * 1024
#: Oversubscribe the payload versus the derived limit so the blocking sort has
#: to fail, with headroom for the planner's key overhead.
SEED_OVERSUBSCRIBE = 1.25

#: ``QueryExceededMemoryLimitNoDiskUseAllowed``.
CODE_QUERY_EXCEEDED_MEMORY_LIMIT = 292
#: Stages a city-prefixed plan must never contain.
BLOCKING_SORT_STAGES = ("SORT", "SORT_MERGE")

#: Per-document sortable padding; the row count is derived from it + the limit.
PAD_BYTES = 700
BATCH_SIZE = 5_000

CITY_SORT = [("city", ASCENDING), ("timestamp", DESCENDING)]
TIMESTAMP_SORT = [("timestamp", DESCENDING)]


def _walk_stages(node) -> list[str]:
    """Every ``stage`` name in an ``explain()`` plan tree, in tree order."""
    stages: list[str] = []
    if isinstance(node, dict):
        if "stage" in node:
            stages.append(node["stage"])
        for value in node.values():
            if isinstance(value, (dict, list)):
                stages.extend(_walk_stages(value))
    elif isinstance(node, list):
        for value in node:
            stages.extend(_walk_stages(value))
    return stages


def _read_sort_limit(client) -> int:
    """Server's blocking-sort budget, or the 100 MiB MongoDB 4.4+ default."""
    try:
        value = client.admin.command({"getParameter": 1, SORT_LIMIT_PARAM: 1}).get(SORT_LIMIT_PARAM)
    except OperationFailure:
        value = None
    return int(value) if value else SORT_LIMIT_FALLBACK_BYTES


def _rows_to_exceed(sort_limit_bytes: int, pad_bytes: int) -> int:
    """Rows whose padded payload clears ``sort_limit_bytes`` (ceil division)."""
    target = int(sort_limit_bytes * SEED_OVERSUBSCRIBE)
    return -(-target // pad_bytes)


def _seed(collection, row_count: int) -> None:
    pad = "x" * PAD_BYTES
    batch: list[dict] = []
    for index in range(row_count):
        batch.append(
            {
                "city": CITIES[index % len(CITIES)],
                "timestamp": BASE_TS + timedelta(hours=index),
                "pad": pad,
                "data": {"main": {"temp": 20.0 + (index % 7)}},
            }
        )
        if len(batch) == BATCH_SIZE:
            collection.insert_many(batch, ordered=False)
            batch = []
    if batch:
        collection.insert_many(batch, ordered=False)

    collection.create_index(CITY_SORT)


@pytest.fixture(scope="module")
def mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2_000)
    try:
        client.admin.command("ping")
        sort_limit_bytes = _read_sort_limit(client)
        row_count = _rows_to_exceed(sort_limit_bytes, PAD_BYTES)
        db = client[DB_NAME]
        db.drop_collection(COLLECTION)
        _seed(db[COLLECTION], row_count)
    except Exception as exc:  # noqa: BLE001 - report why the real server is unusable
        client.close()
        message = f"MongoDB at {MONGO_URI} is not usable for the index guardrails: {exc}"
        if os.getenv("GITHUB_ACTIONS") == "true":
            pytest.fail(message + " (the CI mongo:6 service must be healthy and unauthenticated)")
        pytest.skip(message)

    try:
        yield SimpleNamespace(
            collection=db[COLLECTION],
            sort_limit_bytes=sort_limit_bytes,
            row_count=row_count,
        )
    finally:
        client.drop_database(DB_NAME)
        client.close()


def _winning_stages(collection, query: dict, sort: dict) -> list[str]:
    plan = collection.database.command(
        "explain",
        {"find": COLLECTION, "filter": query, "sort": sort},
        verbosity="queryPlanner",
    )
    return _walk_stages(plan["queryPlanner"]["winningPlan"])


def _city_in_query() -> dict:
    return {
        "city": {"$in": ["Madrid", "Alicante"]},
        "timestamp": {
            "$gte": BASE_TS,
            "$lte": BASE_TS + timedelta(hours=1_000_000),
        },
    }


def _single_city_query(city: str) -> dict:
    return {
        "city": city,
        "timestamp": {
            "$gte": BASE_TS,
            "$lte": BASE_TS + timedelta(hours=1_000_000),
        },
    }


def _assert_index_backed(stages: list[str]) -> None:
    assert "IXSCAN" in stages, f"expected an index scan, got: {stages}"
    blocking = [stage for stage in BLOCKING_SORT_STAGES if stage in stages]
    assert not blocking, f"city-prefixed sort blocks ({blocking}): {stages}"


def test_only_the_city_prefixed_index_exists(mongo):
    keys = [tuple(spec["key"].items()) for spec in mongo.collection.list_indexes()]
    assert (("city", ASCENDING), ("timestamp", DESCENDING)) in keys
    # Hard rule: never a global timestamp-only index — that is what would let a
    # global sort silently pass on M0 and hide the blocking-sort trap.
    assert (("timestamp", DESCENDING),) not in keys


def test_app_declares_the_city_prefixed_weather_data_index():
    """The compound index must be the one the app declares, not just the fixture's."""
    declared = eval_python_constant(MONGO_SOURCE, "INDEXES")
    declared_specs = {
        collection: [tuple(tuple(field) for field in spec) for spec in specs]
        for collection, specs in declared.items()
    }

    assert (("city", ASCENDING), ("timestamp", DESCENDING)) in declared_specs["weather_data"]
    # Hard rule: no global {timestamp: -1} index anywhere in the declaration.
    for collection, specs in declared_specs.items():
        assert (("timestamp", DESCENDING),) not in specs, collection


def test_dataset_exceeds_the_in_memory_sort_budget(mongo):
    inserted = mongo.collection.count_documents({})
    assert inserted == mongo.row_count
    assert (
        inserted * PAD_BYTES > mongo.sort_limit_bytes
    ), "seed must clear the server's blocking-sort budget derived at fixture start"


def test_repo_single_city_ascending_sort_is_index_backed(mongo):
    """Mirror the repo exactly: city equality, ascending timestamp (reverse scan)."""
    stages = _winning_stages(
        mongo.collection,
        _single_city_query(CITIES[0]),
        {"city": ASCENDING, "timestamp": ASCENDING},
    )
    _assert_index_backed(stages)

    rows = list(
        mongo.collection.find(_single_city_query(CITIES[0]), allow_disk_use=False)
        .sort([("city", ASCENDING), ("timestamp", ASCENDING)])
        .limit(1)
    )
    assert len(rows) == 1


def test_city_prefixed_descending_bulk_sort_is_index_backed(mongo):
    """The multi-city bulk read (``find_many_in_range``) uses ``$in`` + descending."""
    stages = _winning_stages(
        mongo.collection, _city_in_query(), {"city": ASCENDING, "timestamp": DESCENDING}
    )
    _assert_index_backed(stages)


def test_timestamp_only_sort_is_blocking(mongo):
    stages = _winning_stages(mongo.collection, {}, {"timestamp": DESCENDING})
    assert "COLLSCAN" in stages
    assert "SORT" in stages, f"expected a blocking sort, got: {stages}"
    assert "SORT_MERGE" not in stages


def test_timestamp_only_sort_fails_without_disk_use(mongo):
    with pytest.raises(OperationFailure) as excinfo:
        list(mongo.collection.find({}, allow_disk_use=False).sort(TIMESTAMP_SORT))
    assert excinfo.value.code == CODE_QUERY_EXCEEDED_MEMORY_LIMIT
