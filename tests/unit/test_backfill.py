"""Behavioural tests for ``scripts/backfill_historical_data.py``.

These tests exercise the module through its public functions: bulk-upsert
semantics (``$setOnInsert``/``upsert``, duplicate-key skips, hard failures),
batching, per-document DST offsets, range validation, ``main`` failure
accounting, the ``--recent`` fill write path (including its future-row filter)
and HTTP error handling. Nothing here inspects the script's source
text, reads ``GITHUB_*`` env vars, depends on the local ``TZ``, or touches a
live API/MongoDB.

``FakeCollection`` stands in for a pymongo collection: it records the
``bulk_write`` calls and returns a ``upserted_count`` the test controls.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from pymongo.errors import BulkWriteError, ServerSelectionTimeoutError

from tests.conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import backfill_historical_data as bh  # noqa: E402

UTC = timezone.utc
#: Fixed instants so no assertion depends on the wall clock or the host TZ.
BASE_TS = int(datetime(2026, 1, 1, 0, 0, tzinfo=UTC).timestamp())
WINTER_TS = int(datetime(2024, 1, 15, 12, 0, tzinfo=UTC).timestamp())
SUMMER_TS = int(datetime(2024, 7, 15, 12, 0, tzinfo=UTC).timestamp())
#: Recent-fill rows that are unmistakably in the past / in the future, so the
#: ``doc["timestamp"] <= now`` filter is exercised without a fake clock.
PAST_TS = int(datetime(2020, 1, 1, 12, 0, tzinfo=UTC).timestamp())
FUTURE_TS = int(datetime(2100, 1, 1, 12, 0, tzinfo=UTC).timestamp())


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _BulkResult:
    def __init__(self, upserted_count: int):
        self.upserted_count = upserted_count


class FakeCollection:
    """Minimal pymongo collection: records ops, replays scripted errors."""

    def __init__(self, upserted_count: int | None = None, errors=()):
        #: ``None`` means "every op was inserted".
        self.upserted_count = upserted_count
        #: Per-position exception (``None`` = success) for ``bulk_write`` calls.
        self.errors = list(errors)
        #: ``[(ops, ordered), ...]`` in call order.
        self.calls: list[tuple[list, bool]] = []
        self.indexes: list[tuple[object, dict]] = []

    def bulk_write(self, operations, ordered=True):
        ops = list(operations)
        call_index = len(self.calls)
        self.calls.append((ops, ordered))
        if call_index < len(self.errors) and self.errors[call_index] is not None:
            raise self.errors[call_index]
        inserted = len(ops) if self.upserted_count is None else self.upserted_count
        return _BulkResult(inserted)

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))
        return "city_1_timestamp_-1"


class FakeDatabase:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, name):
        assert name == "raw_weather", f"unexpected collection {name!r}"
        return self.collection


class FakeClient:
    def __init__(self, collection):
        self.db = FakeDatabase(collection)
        self.closed = False

    def __getitem__(self, name):
        assert name == "weather_db", f"unexpected database {name!r}"
        return self.db

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def _docs(count: int = 3) -> list[dict]:
    return [{"_id": f"Madrid_{BASE_TS + i}", "city": "Madrid"} for i in range(count)]


def _payload(rows: int = 2, **extra) -> dict:
    """An Open-Meteo-shaped archive response with *rows* past hourly points."""
    payload = {
        "city": "Madrid",
        "latitude": 40.4168,
        "longitude": -3.7038,
        "timezone": "Europe/Madrid",
        "utc_offset_seconds": 3600,
        "hourly": {
            "time": [BASE_TS + i for i in range(rows)],
            "temperature_2m": [10.0 + i for i in range(rows)],
        },
    }
    payload.update(extra)
    return payload


def _recent_payload(*, past_rows: int = 2, future_rows: int = 1) -> dict:
    """A forecast-API response: *past_rows* elapsed hours plus *future_rows* ahead."""
    times = [PAST_TS + 3600 * i for i in range(past_rows)]
    times += [FUTURE_TS + 3600 * i for i in range(future_rows)]
    return {
        "city": "Madrid",
        "latitude": 40.4168,
        "longitude": -3.7038,
        "timezone": "Europe/Madrid",
        "utc_offset_seconds": 3600,
        "hourly": {
            "time": times,
            "temperature_2m": [1.0] * len(times),
        },
    }


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# upsert_batch
# ---------------------------------------------------------------------------


class TestUpsertBatch:
    def test_empty_batch_returns_zero_and_does_not_write(self):
        collection = FakeCollection()

        assert bh.upsert_batch(collection, []) == (0, 0)
        assert collection.calls == []

    def test_returns_inserted_and_skipped_counts(self):
        collection = FakeCollection(upserted_count=2)

        assert bh.upsert_batch(collection, _docs(3)) == (2, 1)

        ops, ordered = collection.calls[0]
        assert len(ops) == 3
        assert ordered is False

    def test_ops_are_set_on_insert_upserts_keyed_by_id(self):
        docs = _docs(3)
        collection = FakeCollection()

        bh.upsert_batch(collection, docs)

        ops = collection.calls[0][0]
        assert [op._filter for op in ops] == [{"_id": doc["_id"]} for doc in docs]
        assert [op._doc for op in ops] == [{"$setOnInsert": doc} for doc in docs]
        assert all(op._upsert is True for op in ops)

    @pytest.mark.parametrize("n_upserted", [0, 2])
    def test_all_duplicate_bulk_error_is_counted_as_skipped(self, n_upserted):
        docs = _docs(5)
        duplicate_error = BulkWriteError(
            {
                "writeErrors": [{"code": 11000, "errmsg": "duplicate key"} for _ in docs],
                "nUpserted": n_upserted,
            }
        )

        inserted, skipped = bh.upsert_batch(FakeCollection(errors=[duplicate_error]), docs)

        assert inserted == n_upserted
        assert skipped == len(docs) - n_upserted

    @pytest.mark.parametrize(
        "details",
        [
            {"writeErrors": [{"code": 121, "errmsg": "document validation failure"}]},
            {"writeErrors": [{"code": 11000}, {"code": 121}]},
            {"nUpserted": 0},
            {"writeErrors": []},
        ],
        ids=["non-duplicate", "mixed", "no-writeErrors", "empty-writeErrors"],
    )
    def test_non_duplicate_bulk_error_raises_runtime_error(self, details):
        docs = _docs(3)
        original = BulkWriteError(details)

        with pytest.raises(RuntimeError) as raised:
            bh.upsert_batch(FakeCollection(errors=[original]), docs)

        assert raised.value.__cause__ is original

    def test_non_duplicate_bulk_error_message_names_the_code(self):
        original = BulkWriteError({"writeErrors": [{"code": 121}]})

        with pytest.raises(RuntimeError, match="121"):
            bh.upsert_batch(FakeCollection(errors=[original]), _docs(1))


# ---------------------------------------------------------------------------
# upsert_documents batching
# ---------------------------------------------------------------------------


class TestUpsertDocuments:
    def test_splits_into_batches_and_sums_the_counts(self, monkeypatch):
        monkeypatch.setattr(bh, "BATCH_UPSERT_SIZE", 2)
        collection = FakeCollection()

        assert bh.upsert_documents(collection, _docs(5)) == (5, 0)

        assert [len(ops) for ops, _ in collection.calls] == [2, 2, 1]
        assert all(ordered is False for _, ordered in collection.calls)

    def test_empty_document_list_does_not_write(self):
        collection = FakeCollection()

        assert bh.upsert_documents(collection, []) == (0, 0)
        assert collection.calls == []

    def test_runtime_error_from_a_later_batch_propagates(self, monkeypatch):
        monkeypatch.setattr(bh, "BATCH_UPSERT_SIZE", 2)
        original = BulkWriteError({"writeErrors": [{"code": 121}]})
        collection = FakeCollection(errors=[None, original])
        docs = [{"_id": f"doc_{i}"} for i in range(5)]

        with pytest.raises(RuntimeError) as raised:
            bh.upsert_documents(collection, docs)

        assert raised.value.__cause__ is original
        # The first (successful) batch was written; the third was never reached.
        assert [len(ops) for ops, _ in collection.calls] == [2, 2]


# ---------------------------------------------------------------------------
# build_documents: per-document offsets (audit finding 6)
# ---------------------------------------------------------------------------


class TestBuildDocuments:
    def test_utc_offset_follows_dst_within_one_payload(self):
        payload = {
            "timezone": "Europe/Madrid",
            # Open-Meteo's single sampled offset, stale for half the range.
            "utc_offset_seconds": 7200,
            "hourly": {"time": [WINTER_TS, SUMMER_TS], "temperature_2m": [5.0, 30.0]},
        }

        docs = bh.build_documents("Madrid", payload)

        assert [doc["payload"]["utc_offset_seconds"] for doc in docs] == [3600, 7200]
        assert [doc["_id"] for doc in docs] == [
            f"Madrid_{WINTER_TS}",
            f"Madrid_{SUMMER_TS}",
        ]
        assert docs[0]["timestamp"] == datetime(2024, 1, 15, 12, 0, tzinfo=UTC)
        assert docs[1]["timestamp"] == datetime(2024, 7, 15, 12, 0, tzinfo=UTC)

    def test_missing_timezone_falls_back_to_the_payload_offset(self):
        payload = _payload(rows=3)
        payload.pop("timezone")
        payload["utc_offset_seconds"] = 7200

        docs = bh.build_documents("Madrid", payload)

        assert [doc["payload"]["utc_offset_seconds"] for doc in docs] == [7200, 7200, 7200]

    def test_unknown_timezone_falls_back_to_the_payload_offset(self):
        payload = _payload(rows=2, timezone="Nowhere/Nothing", utc_offset_seconds=-3600)

        docs = bh.build_documents("Madrid", payload)

        assert [doc["payload"]["utc_offset_seconds"] for doc in docs] == [-3600, -3600]

    def test_no_timezone_and_no_offset_stores_none(self):
        payload = _payload(rows=2)
        payload.pop("timezone")
        payload.pop("utc_offset_seconds")

        docs = bh.build_documents("Madrid", payload)

        assert [doc["payload"]["utc_offset_seconds"] for doc in docs] == [None, None]

    def test_empty_hourly_window_yields_no_documents(self):
        assert bh.build_documents("Madrid", {"hourly": {"time": []}}) == []


# ---------------------------------------------------------------------------
# resolve_range
# ---------------------------------------------------------------------------


def _range(start: str, end: str) -> argparse.Namespace:
    return argparse.Namespace(start=start, end=end)


class TestResolveRange:
    def test_valid_default_length_range_is_returned(self):
        start, end = bh.resolve_range(_range("2026-01-01", "2026-03-31"))

        assert (start, end) == (date(2026, 1, 1), date(2026, 3, 31))
        assert (end - start).days + 1 == 90

    def test_range_longer_than_max_range_days_exits_1(self):
        with pytest.raises(SystemExit) as raised:
            bh.resolve_range(_range("2026-01-01", "2026-07-20"))  # 201 days

        assert raised.value.code == 1

    def test_range_of_exactly_max_range_days_is_accepted(self):
        start, end = bh.resolve_range(_range("2026-01-01", "2026-06-29"))  # 180 days

        assert (end - start).days + 1 == bh.MAX_RANGE_DAYS

    def test_start_after_end_exits_1(self):
        with pytest.raises(SystemExit) as raised:
            bh.resolve_range(_range("2026-02-01", "2026-01-01"))

        assert raised.value.code == 1

    def test_parse_args_default_is_the_last_default_days(self, monkeypatch):
        class _FixedDate(date):
            @classmethod
            def today(cls):
                return date(2026, 5, 20)

        monkeypatch.setattr(bh, "date", _FixedDate)
        monkeypatch.setattr(sys, "argv", ["prog"])

        args = bh.parse_args()
        start, end = bh.resolve_range(args)

        assert args.days is None
        assert start == date(2026, 5, 20) - timedelta(days=bh.DEFAULT_DAYS)
        assert end == date(2026, 5, 19)  # yesterday
        assert (end - start).days + 1 == bh.DEFAULT_DAYS == 90


# ---------------------------------------------------------------------------
# main() exit accounting
# ---------------------------------------------------------------------------


CHUNK_ARGV = [
    "prog",
    "--start",
    "2026-01-01",
    "--end",
    "2026-01-02",
    "--mongo-url",
    "mongodb://fake",
]


def _patch_main(monkeypatch, collection, fetch, *, recent_fetch=None, argv=None, cities=None):
    """Wire main() to fakes: no network, no Mongo, no sleeps."""
    client = FakeClient(collection)
    registry = dict(cities or {"Madrid": (40.4168, -3.7038)})
    monkeypatch.setattr(bh, "MongoClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(bh, "load_cities", lambda db: registry)
    monkeypatch.setattr(bh, "fetch_historical_chunk", fetch)
    monkeypatch.setattr(bh, "fetch_recent_chunk", recent_fetch or (lambda *a, **k: _payload()))
    monkeypatch.setattr(bh.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", list(argv or CHUNK_ARGV))
    return client


class TestMainExitAccounting:
    def test_chunk_fetch_failure_exits_1(self, monkeypatch):
        collection = FakeCollection()
        _patch_main(monkeypatch, collection, lambda *a, **k: None)

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1
        assert collection.calls == []

    def test_recent_fetch_failure_exits_1(self, monkeypatch):
        collection = FakeCollection()
        argv = [*CHUNK_ARGV, "--recent", "2"]
        _patch_main(
            monkeypatch,
            collection,
            lambda *a, **k: _payload(),
            recent_fetch=lambda *a, **k: None,
            argv=argv,
        )

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1

    def test_accounting_mismatch_exits_1(self, monkeypatch):
        collection = FakeCollection()
        _patch_main(monkeypatch, collection, lambda *a, **k: _payload(rows=2))
        # upsert_batch always derives skipped = len(docs) - inserted, so its
        # arithmetic cannot disagree with len(docs); this guard protects against
        # a writer adapter that under-reports. Simulate that writer here.
        monkeypatch.setattr(bh, "upsert_documents", lambda coll, docs: (0, 1))

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1

    def test_run_where_every_document_already_exists_exits_0(self, monkeypatch, capsys):
        collection = FakeCollection(upserted_count=0)
        _patch_main(monkeypatch, collection, lambda *a, **k: _payload(rows=2))

        with pytest.raises(SystemExit) as raised:
            bh.main()

        # A clean run is an explicit exit 0, not an implicit fall-off-the-end.
        assert raised.value.code == 0
        assert len(collection.calls) == 1
        assert len(collection.calls[0][0]) == 2
        out = capsys.readouterr().out
        assert "Backfill complete." in out
        assert "Total inserted : 0" in out

    def test_run_with_nothing_written_exits_1(self, monkeypatch, capsys):
        collection = FakeCollection()
        _patch_main(
            monkeypatch,
            collection,
            lambda *a, **k: {"hourly": {"time": []}},
        )

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1
        assert collection.calls == []
        out = capsys.readouterr().out
        assert "[ERROR] Nothing was written" in out
        assert "Total inserted : 0" in out
        assert "Total skipped  : 0" in out
        assert "Backfill FAILED." in out

    def test_dry_run_prints_plan_without_fetching_or_writing(self, monkeypatch, capsys):
        collection = FakeCollection()

        def _no_fetch(*args, **kwargs):
            raise AssertionError("dry-run must not call Open-Meteo")

        _patch_main(
            monkeypatch,
            collection,
            _no_fetch,
            recent_fetch=_no_fetch,
            argv=[*CHUNK_ARGV, "--dry-run"],
        )

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 0
        assert collection.calls == []
        assert "[DRY RUN]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main(): API failures vs write failures (the per-city silent-loss defect)
# ---------------------------------------------------------------------------


TWO_CITIES = {"Madrid": (40.4168, -3.7038), "Pasadena": (34.1478, -118.1445)}
#: A 200 response with no hourly rows at all.
EMPTY_PAYLOAD = {"hourly": {"time": []}}


class TestApiFailureVersusWriteFailure:
    def test_empty_archive_payload_is_an_api_failure_not_a_write(self, monkeypatch, capsys):
        collection = FakeCollection()
        _patch_main(monkeypatch, collection, lambda *a, **k: EMPTY_PAYLOAD)

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1
        out = capsys.readouterr().out
        assert "EMPTY — no hourly rows in the response" in out
        assert "API failures   : 1" in out
        assert "Write failures" not in out
        assert "Backfill FAILED." in out
        assert collection.calls == []

    def test_empty_recent_payload_is_an_api_failure(self, monkeypatch, capsys):
        collection = FakeCollection()
        _patch_main(
            monkeypatch,
            collection,
            lambda *a, **k: _payload(rows=2),
            recent_fetch=lambda *a, **k: EMPTY_PAYLOAD,
            argv=[*CHUNK_ARGV, "--recent", "2"],
        )

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1
        out = capsys.readouterr().out
        assert "EMPTY — no past hourly rows in the response" in out
        assert "API failures   : 1" in out
        # The archive chunk was written before the recent fill came back empty.
        assert len(collection.calls) == 1
        assert "Total inserted : 2" in out

    @pytest.mark.parametrize(
        "failure_payload", [None, EMPTY_PAYLOAD], ids=["fetch-returns-none", "empty-payload"]
    )
    @pytest.mark.parametrize("best_effort", [False, True], ids=["strict", "best-effort"])
    def test_one_city_losing_its_payload(self, monkeypatch, capsys, failure_payload, best_effort):
        """Madrid writes fine, Pasadena yields nothing: strict fails, --best-effort warns."""
        collection = FakeCollection()

        def _fetch(city, *args, **kwargs):
            return _payload(rows=2) if city == "Madrid" else failure_payload

        argv = [*CHUNK_ARGV] + (["--best-effort"] if best_effort else [])
        _patch_main(monkeypatch, collection, _fetch, argv=argv, cities=TWO_CITIES)

        with pytest.raises(SystemExit) as raised:
            bh.main()

        out = capsys.readouterr().out
        assert raised.value.code == (0 if best_effort else 1)
        assert "API failures   : 1" in out
        # Madrid's documents are written and reported in both runs.
        assert "Total inserted : 2" in out
        assert len(collection.calls) == 1
        if best_effort:
            assert "::warning title=Backfill partial::" in out
            assert "Backfill complete." in out
            assert "Backfill FAILED." not in out
        else:
            assert "Backfill FAILED." in out

    def test_all_api_failures_with_best_effort_exit_0_with_a_warning(self, monkeypatch, capsys):
        collection = FakeCollection()
        _patch_main(
            monkeypatch,
            collection,
            lambda *a, **k: EMPTY_PAYLOAD,
            argv=[*CHUNK_ARGV, "--best-effort"],
        )

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 0
        out = capsys.readouterr().out
        assert "::warning title=Backfill partial::" in out
        assert "Total inserted : 0" in out
        assert "Backfill complete." in out
        assert collection.calls == []

    @pytest.mark.parametrize("extra", [[], ["--best-effort"]], ids=["strict", "best-effort"])
    def test_write_failure_exits_1_aborts_the_run_and_still_summarises(
        self, monkeypatch, capsys, extra
    ):
        non_duplicate = BulkWriteError({"writeErrors": [{"code": 121}]})
        collection = FakeCollection(errors=[non_duplicate])
        fetched: list[str] = []
        recent: list[str] = []

        def _fetch(city, *args, **kwargs):
            fetched.append(city)
            return _payload(rows=2)

        def _recent_fetch(city, *args, **kwargs):
            recent.append(city)
            return _payload(rows=2)

        argv = [
            "prog",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-02",
            "--chunk-days",
            "1",
            "--recent",
            "2",
            "--mongo-url",
            "mongodb://fake",
            *extra,
        ]
        client = _patch_main(
            monkeypatch,
            collection,
            _fetch,
            recent_fetch=_recent_fetch,
            argv=argv,
            cities=TWO_CITIES,
        )

        # The RuntimeError must not escape main(): it becomes a summary + exit 1.
        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1
        # Aborted on the first failing chunk: no second chunk, no second city,
        # no recent fill - but the client is still closed and the summary printed.
        assert fetched == ["Madrid"]
        assert recent == []
        assert len(collection.calls) == 1
        assert client.closed is True
        out = capsys.readouterr().out
        assert "Write failures : 1" in out
        assert "Backfill FAILED." in out

    def test_accounting_mismatch_is_not_downgraded_by_best_effort(self, monkeypatch):
        collection = FakeCollection()
        _patch_main(
            monkeypatch,
            collection,
            lambda *a, **k: _payload(rows=2),
            argv=[*CHUNK_ARGV, "--best-effort"],
        )
        monkeypatch.setattr(bh, "upsert_documents", lambda coll, docs: (0, 1))

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1

    def test_clean_run_exits_0(self, monkeypatch, capsys):
        collection = FakeCollection()
        _patch_main(monkeypatch, collection, lambda *a, **k: _payload(rows=2))

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 0
        out = capsys.readouterr().out
        assert "Backfill complete." in out
        assert "Backfill FAILED." not in out
        assert "::warning" not in out


# ---------------------------------------------------------------------------
# main(): the --recent fill write path (the mode ml-pipeline.yml runs)
# ---------------------------------------------------------------------------

#: Archive range + the ``--recent`` flag, i.e. what .github/workflows does.
RECENT_ARGV = [*CHUNK_ARGV, "--recent", "2"]


class TestRecentFillWritePath:
    def test_past_rows_are_written_future_rows_filtered_and_exit_0(self, monkeypatch, capsys):
        collection = FakeCollection()
        _patch_main(
            monkeypatch,
            collection,
            lambda *a, **k: _payload(rows=2),
            recent_fetch=lambda *a, **k: _recent_payload(past_rows=2, future_rows=1),
            argv=RECENT_ARGV,
        )

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 0
        # One write for the archive chunk, one for the recent fill.
        assert len(collection.calls) == 2
        assert len(collection.calls[0][0]) == 2
        recent_ids = [op._filter["_id"] for op in collection.calls[1][0]]
        # Elapsed hours are written…
        assert recent_ids == [f"Madrid_{PAST_TS}", f"Madrid_{PAST_TS + 3600}"]
        # …and tomorrow's forecast row is not.
        assert f"Madrid_{FUTURE_TS}" not in recent_ids

        out = capsys.readouterr().out
        assert "Recent fill (last 2 days, forecast API)" in out
        assert "inserted=2, skipped=0" in out
        assert "Total inserted : 4" in out  # 2 archive + 2 recent
        assert "Total skipped  : 0" in out
        assert "Backfill complete." in out
        assert "Backfill FAILED." not in out

    def test_write_failure_exits_1_annotates_and_aborts_the_recent_loop(self, monkeypatch, capsys):
        non_duplicate = BulkWriteError({"writeErrors": [{"code": 121}]})
        # Calls 0 and 1 are the two cities' archive chunks; call 2 is Madrid's
        # recent fill, which fails.
        collection = FakeCollection(errors=[None, None, non_duplicate])
        recent_fetched: list[str] = []

        def _recent_fetch(city, *args, **kwargs):
            recent_fetched.append(city)
            return _recent_payload(past_rows=2, future_rows=0)

        client = _patch_main(
            monkeypatch,
            collection,
            lambda *a, **k: _payload(rows=2),
            recent_fetch=_recent_fetch,
            argv=RECENT_ARGV,
            cities=TWO_CITIES,
        )

        # The RuntimeError must be handled, not escape main().
        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1
        # Aborted after the failure: the second city's recent fill never ran.
        assert recent_fetched == ["Madrid"]
        assert len(collection.calls) == 3
        out = capsys.readouterr().out
        assert "[ERROR] write failed for Madrid (recent fill)" in out
        assert "::error title=Backfill write failure::" in out
        assert "Madrid (recent fill)" in out
        assert "Write failures : 1" in out
        # Both archive chunks were written before the recent fill failed, and
        # the summary still prints.
        assert "Total inserted : 4" in out
        assert "Backfill FAILED." in out
        assert client.closed is True

    def test_accounting_mismatch_exits_1(self, monkeypatch, capsys):
        collection = FakeCollection()
        _patch_main(
            monkeypatch,
            collection,
            lambda *a, **k: _payload(rows=2),
            recent_fetch=lambda *a, **k: _recent_payload(past_rows=2, future_rows=0),
            argv=RECENT_ARGV,
        )
        sizes: list[int] = []

        def _under_reporting_upsert(coll, docs):
            sizes.append(len(docs))
            # The archive batch adds up; the recent-fill batch does not.
            return (len(docs), 0) if len(sizes) == 1 else (0, 1)

        monkeypatch.setattr(bh, "upsert_documents", _under_reporting_upsert)

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 1
        assert sizes == [2, 2]
        out = capsys.readouterr().out
        assert "[ERROR] accounting mismatch for Madrid (recent fill)" in out
        assert "inserted(0) + skipped(1) != fetched(2)" in out
        assert "Write failures : 1" in out
        assert "Backfill FAILED." in out


# ---------------------------------------------------------------------------
# --chunk-days: rejected by argparse, never a ZeroDivisionError/infinite loop
# ---------------------------------------------------------------------------


class TestChunkDaysValidation:
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_non_positive_chunk_days_is_a_usage_error(self, monkeypatch, capsys, value):
        monkeypatch.setattr(sys, "argv", ["prog", "--chunk-days", value])

        with pytest.raises(SystemExit) as raised:
            bh.parse_args()

        assert raised.value.code == 2
        assert "must be >= 1" in capsys.readouterr().err

    def test_chunk_days_of_one_is_accepted(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--chunk-days", "1"])

        assert bh.parse_args().chunk_days == 1

    def test_main_exits_2_before_any_client_or_loop(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["prog", "--chunk-days", "0", "--mongo-url", "mongodb://fake"]
        )

        def _no_client(*args, **kwargs):
            raise AssertionError("a bad --chunk-days must fail before MongoDB is touched")

        monkeypatch.setattr(bh, "MongoClient", _no_client)

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 2


# ---------------------------------------------------------------------------
# --dry-run with an unreachable MongoDB
# ---------------------------------------------------------------------------


class TestDryRunOffline:
    def test_fallback_cities_are_the_shared_seed_tuple(self):
        """The offline fallback is the API's seed list, not a second registry.

        It used to be a private dict in this script, so an offline dry run
        planned a different station set than the app ingests.
        """
        assert len(bh.FALLBACK_CITIES) == 14
        assert bh.FALLBACK_CITIES["Malaga"] == (36.7213, -4.4214)  # unaccented on purpose
        assert "Pasadena" not in bh.FALLBACK_CITIES  # removed 2026-11-11
        assert set(bh.FALLBACK_CITIES) == {name for name, _, _ in bh.DEFAULT_CITIES}

    def test_unreachable_mongo_falls_back_to_the_seed_stations(self, monkeypatch, capsys):
        collection = FakeCollection()

        def _no_client(*args, **kwargs):
            raise ServerSelectionTimeoutError("no server available")

        def _no_fetch(*args, **kwargs):
            raise AssertionError("--dry-run must not call Open-Meteo")

        _patch_main(
            monkeypatch,
            collection,
            _no_fetch,
            recent_fetch=_no_fetch,
            argv=[*CHUNK_ARGV, "--dry-run"],
        )
        monkeypatch.setattr(bh, "MongoClient", _no_client)

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 0
        assert collection.calls == []
        out = capsys.readouterr().out
        assert "[WARN] MongoDB unreachable" in out
        assert "ServerSelectionTimeoutError" in out
        assert f"Cities         : {len(bh.FALLBACK_CITIES)}" in out
        assert "[DRY RUN]" in out

    def test_unreachable_mongo_still_honours_the_city_filter(self, monkeypatch, capsys):
        collection = FakeCollection()

        def _no_client(*args, **kwargs):
            raise ServerSelectionTimeoutError("no server available")

        def _no_fetch(*args, **kwargs):
            raise AssertionError("--dry-run must not call Open-Meteo")

        _patch_main(
            monkeypatch,
            collection,
            _no_fetch,
            recent_fetch=_no_fetch,
            argv=[*CHUNK_ARGV, "--dry-run", "--city", "Madrid"],
        )
        monkeypatch.setattr(bh, "MongoClient", _no_client)

        with pytest.raises(SystemExit) as raised:
            bh.main()

        assert raised.value.code == 0
        out = capsys.readouterr().out
        assert "[WARN] MongoDB unreachable" in out
        assert f"Cities         : 1 of {len(bh.FALLBACK_CITIES)}" in out


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


class TestFetchHistoricalChunk:
    def test_success_returns_payload_tagged_with_city(self):
        response = _Response(200, {"hourly": {"time": [BASE_TS]}})

        with patch.object(bh.requests, "get", return_value=response) as get:
            result = bh.fetch_historical_chunk(
                "Madrid", 40.4168, -3.7038, date(2026, 1, 1), date(2026, 1, 2)
            )

        assert result["city"] == "Madrid"
        assert result["hourly"]["time"] == [BASE_TS]
        assert get.call_args.args[0] == bh.OPEN_METEO_HISTORICAL_URL

    def test_non_200_returns_none_and_warns(self, capsys):
        with patch.object(bh.requests, "get", return_value=_Response(500, text="boom")):
            result = bh.fetch_historical_chunk(
                "Madrid", 40.4168, -3.7038, date(2026, 1, 1), date(2026, 1, 2)
            )

        assert result is None
        assert "HTTP 500" in capsys.readouterr().out

    def test_request_exception_returns_none(self, capsys):
        with patch.object(bh.requests, "get", side_effect=ConnectionError("down")):
            result = bh.fetch_historical_chunk(
                "Madrid", 40.4168, -3.7038, date(2026, 1, 1), date(2026, 1, 2)
            )

        assert result is None
        assert "[ERROR]" in capsys.readouterr().out


class TestFetchRecentChunk:
    def test_success_returns_payload_tagged_with_city(self):
        response = _Response(200, {"hourly": {"time": [BASE_TS]}})

        with patch.object(bh.requests, "get", return_value=response) as get:
            result = bh.fetch_recent_chunk("Madrid", 40.4168, -3.7038, 5)

        assert result["city"] == "Madrid"
        assert result["hourly"]["time"] == [BASE_TS]
        assert get.call_args.args[0] == bh.OPEN_METEO_FORECAST_URL

    def test_non_200_returns_none_and_warns(self, capsys):
        with patch.object(bh.requests, "get", return_value=_Response(503, text="unavailable")):
            result = bh.fetch_recent_chunk("Madrid", 40.4168, -3.7038, 5)

        assert result is None
        assert "HTTP 503" in capsys.readouterr().out

    def test_request_exception_returns_none(self):
        with patch.object(bh.requests, "get", side_effect=TimeoutError("slow")):
            result = bh.fetch_recent_chunk("Madrid", 40.4168, -3.7038, 5)

        assert result is None


# ---------------------------------------------------------------------------
# --city filtering / argument parsing
# ---------------------------------------------------------------------------


REGISTRY = {
    "Madrid": (40.4168, -3.7038),
    "Pasadena": (34.1478, -118.1445),
    "Almería": (36.8381, -2.4597),
    "Sevilla": (37.38283, -5.97317),
}


class TestCityFilter:
    def test_parse_default_is_empty(self):
        assert bh.parse_city_filter(None) == []
        assert bh.parse_city_filter([]) == []

    def test_parse_single_city(self):
        assert bh.parse_city_filter(["Pasadena"]) == ["Pasadena"]

    def test_parse_comma_separated(self):
        assert bh.parse_city_filter(["Pasadena, Madrid"]) == ["Pasadena", "Madrid"]

    def test_parse_repeated_flags(self):
        assert bh.parse_city_filter(["Pasadena", "Madrid"]) == ["Pasadena", "Madrid"]

    def test_parse_mixed_and_deduplicated(self):
        assert bh.parse_city_filter(["Pasadena,Madrid", "Madrid"]) == ["Pasadena", "Madrid"]
        assert bh.parse_city_filter(["Pasadena,"]) == ["Pasadena"]

    def test_select_without_filter_keeps_registry(self):
        selected = bh.select_cities(REGISTRY, [])
        assert selected == REGISTRY
        assert selected is not REGISTRY  # a copy, not the registry itself

    def test_select_single_city(self):
        assert bh.select_cities(REGISTRY, ["Pasadena"]) == {"Pasadena": (34.1478, -118.1445)}

    def test_select_multiple_cities(self):
        selected = bh.select_cities(REGISTRY, ["Pasadena", "Madrid"])
        assert list(selected) == ["Pasadena", "Madrid"]

    def test_select_is_accent_and_case_insensitive(self):
        assert list(bh.select_cities(REGISTRY, ["almeria"])) == ["Almería"]
        assert list(bh.select_cities(REGISTRY, ["PASADENA"])) == ["Pasadena"]

    def test_select_unknown_city_raises(self):
        with pytest.raises(KeyError, match="Pasadina"):
            bh.select_cities(REGISTRY, ["Pasadina"])

    def test_parse_args_accepts_repeatable_and_comma_city(self, monkeypatch):
        argv = ["prog", "--days", "35", "--recent", "5", "--city", "Pasadena,Madrid"]
        monkeypatch.setattr(sys, "argv", argv)

        args = bh.parse_args()

        assert args.city == ["Pasadena,Madrid"]
        assert args.days == 35
        assert args.recent == 5
        assert bh.parse_city_filter(args.city) == ["Pasadena", "Madrid"]

    def test_parse_args_city_defaults_to_none(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--days", "35"])

        args = bh.parse_args()

        assert args.city is None
        assert bh.parse_city_filter(args.city) == []

    def test_existing_flags_still_default(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog"])

        args = bh.parse_args()

        assert args.recent == 0
        assert args.chunk_days == 180
        assert args.start and args.end
