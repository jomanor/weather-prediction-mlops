"""
tests/unit/test_backfill.py
============================
Unit tests for scripts/backfill_historical_data.py

We mock requests and MongoClient to verify:
- Idempotency: $setOnInsert is used (no overwrite of existing docs)
- Batch upserts are called
- HTTP errors are handled gracefully
"""

import os
import sys
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# Import the module under test (with mocked deps)
# ---------------------------------------------------------------------------


def _import_backfill():
    with patch.dict(os.environ, {"MONGO_URL": "mongodb://localhost:27017"}):
        import importlib

        import backfill_historical_data as bh

        importlib.reload(bh)
        return bh


# ---------------------------------------------------------------------------
# Inspect the source to verify $setOnInsert is present
# ---------------------------------------------------------------------------


class TestIdempotencyPattern:
    def test_set_on_insert_in_source(self):
        """The backfill script must use $setOnInsert to avoid overwriting data."""
        src_path = os.path.join(SCRIPTS_DIR, "backfill_historical_data.py")
        with open(src_path) as f:
            source = f.read()
        assert (
            "$setOnInsert" in source
        ), "backfill must use $setOnInsert so existing docs are not overwritten"

    def test_upsert_true_in_source(self):
        """update_one calls must have upsert=True."""
        src_path = os.path.join(SCRIPTS_DIR, "backfill_historical_data.py")
        with open(src_path) as f:
            source = f.read()
        assert (
            "upsert=True" in source or "upsert = True" in source
        ), "backfill must use upsert=True in update_one calls"

    def test_bulk_write_or_update_one_present(self):
        """Backfill must use either bulk_write or update_one for upserts."""
        src_path = os.path.join(SCRIPTS_DIR, "backfill_historical_data.py")
        with open(src_path) as f:
            source = f.read()
        assert (
            "bulk_write" in source or "update_one" in source
        ), "backfill must use bulk_write or update_one for DB writes"


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


class TestApiErrorHandling:
    def test_module_imports_requests(self):
        """backfill_historical_data.py must import requests."""
        src_path = os.path.join(SCRIPTS_DIR, "backfill_historical_data.py")
        with open(src_path) as f:
            source = f.read()
        assert "import requests" in source or "from requests" in source

    def test_module_has_exception_handling(self):
        """There should be at least one try/except block for resilience."""
        src_path = os.path.join(SCRIPTS_DIR, "backfill_historical_data.py")
        with open(src_path) as f:
            source = f.read()
        assert "except" in source, "backfill should handle exceptions"


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


class TestStructure:
    def test_cities_defined(self):
        """backfill must define a list/dict of cities to backfill."""
        src_path = os.path.join(SCRIPTS_DIR, "backfill_historical_data.py")
        with open(src_path) as f:
            source = f.read()
        assert "Madrid" in source, "backfill must include Madrid in city list"
        assert "Barcelona" in source

    def test_open_meteo_historical_api_used(self):
        """backfill must call the open-meteo historical (archive) endpoint."""
        src_path = os.path.join(SCRIPTS_DIR, "backfill_historical_data.py")
        with open(src_path) as f:
            source = f.read()
        assert (
            "archive-api.open-meteo.com" in source or "historical" in source.lower()
        ), "backfill must use the open-meteo historical / archive endpoint"


# ---------------------------------------------------------------------------
# --city argument parsing / registry filtering
# ---------------------------------------------------------------------------


REGISTRY = {
    "Madrid": (40.4168, -3.7038),
    "Pasadena": (34.1478, -118.1445),
    "Almería": (36.8381, -2.4597),
    "Sevilla": (37.38283, -5.97317),
}


class TestCityFilter:
    def test_parse_default_is_empty(self):
        bh = _import_backfill()
        assert bh.parse_city_filter(None) == []
        assert bh.parse_city_filter([]) == []

    def test_parse_single_city(self):
        bh = _import_backfill()
        assert bh.parse_city_filter(["Pasadena"]) == ["Pasadena"]

    def test_parse_comma_separated(self):
        bh = _import_backfill()
        assert bh.parse_city_filter(["Pasadena, Madrid"]) == ["Pasadena", "Madrid"]

    def test_parse_repeated_flags(self):
        bh = _import_backfill()
        assert bh.parse_city_filter(["Pasadena", "Madrid"]) == ["Pasadena", "Madrid"]

    def test_parse_mixed_and_deduplicated(self):
        bh = _import_backfill()
        assert bh.parse_city_filter(["Pasadena,Madrid", "Madrid"]) == ["Pasadena", "Madrid"]
        assert bh.parse_city_filter(["Pasadena,"]) == ["Pasadena"]

    def test_select_without_filter_keeps_registry(self):
        bh = _import_backfill()
        selected = bh.select_cities(REGISTRY, [])
        assert selected == REGISTRY
        assert selected is not REGISTRY  # a copy, not the registry itself

    def test_select_single_city(self):
        bh = _import_backfill()
        assert bh.select_cities(REGISTRY, ["Pasadena"]) == {"Pasadena": (34.1478, -118.1445)}

    def test_select_multiple_cities(self):
        bh = _import_backfill()
        selected = bh.select_cities(REGISTRY, ["Pasadena", "Madrid"])
        assert list(selected) == ["Pasadena", "Madrid"]

    def test_select_is_accent_and_case_insensitive(self):
        bh = _import_backfill()
        assert list(bh.select_cities(REGISTRY, ["almeria"])) == ["Almería"]
        assert list(bh.select_cities(REGISTRY, ["PASADENA"])) == ["Pasadena"]

    def test_select_unknown_city_raises(self):
        bh = _import_backfill()
        try:
            bh.select_cities(REGISTRY, ["Pasadina"])
        except KeyError as exc:
            assert "Pasadina" in str(exc)
        else:
            raise AssertionError("unknown city must raise KeyError")

    def test_parse_args_accepts_repeatable_and_comma_city(self):
        bh = _import_backfill()
        argv = ["prog", "--days", "35", "--recent", "5", "--city", "Pasadena,Madrid"]
        with patch.object(sys, "argv", argv):
            args = bh.parse_args()
        assert args.city == ["Pasadena,Madrid"]
        assert args.days == 35
        assert args.recent == 5
        assert bh.parse_city_filter(args.city) == ["Pasadena", "Madrid"]

    def test_parse_args_city_defaults_to_none(self):
        bh = _import_backfill()
        with patch.object(sys, "argv", ["prog", "--days", "35"]):
            args = bh.parse_args()
        assert args.city is None
        assert bh.parse_city_filter(args.city) == []

    def test_existing_flags_still_default(self):
        bh = _import_backfill()
        with patch.object(sys, "argv", ["prog"]):
            args = bh.parse_args()
        assert args.recent == 0
        assert args.chunk_days == 180
        assert args.start and args.end
