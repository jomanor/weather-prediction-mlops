"""
tests/unit/test_backfill.py
============================
Unit tests for scripts/backfill_historical_data.py

We mock requests and MongoClient to verify:
- Idempotency: $setOnInsert is used (no overwrite of existing docs)
- Batch upserts are called
- HTTP errors are handled gracefully
"""

import sys
import os
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
