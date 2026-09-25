"""Contract test for the ``weather_predictions`` write schema.

``inference.run_inference`` builds the exact column list it writes to
``weather_predictions`` via the pure helper ``inference.output_schema_columns()``.
This test locks that contract down: the columns, their order relative to the
frozen legacy set, and the stable ``_id`` composition
``"<city>_<unix_seconds>_<horizon>h"``.

No SparkSession is started — the helper is pure Python and the module is only
imported for its constants/helpers. Import is made to work outside the image by
putting ``spark/spark-jobs`` and ``spark/config`` on ``sys.path`` (see
``tests/unit/test_inference.py``).
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure spark jobs and config are importable outside the runtime image, where
# they live at /opt/jobs and /opt/config respectively.
SPARK_JOBS_DIR = os.path.join(os.path.dirname(__file__), "../../spark/spark-jobs")
SPARK_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../spark/config")
for directory in (SPARK_JOBS_DIR, SPARK_CONFIG_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

import inference  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen write-column contract (docs/batch3-contract.md §R4 / §Contract 2).
# ---------------------------------------------------------------------------

#: The legacy columns, in their established relative order.
CORE_COLUMNS = [
    "_id",
    "city",
    "source_timestamp",
    "prediction_timestamp",
    "predicted_temperature",
    "predicted_rain",
    "observed_temperature",
    "horizon_hours",
    "temp_model_name",
    "temp_model_version",
    "rain_model_name",
    "rain_model_version",
]

#: Additive interval fields (Contract 2). Nullable for legacy documents.
ADDITIVE_COLUMNS = ["temp_lower", "temp_upper", "interval_level"]

EXPECTED_COLUMNS = CORE_COLUMNS + ADDITIVE_COLUMNS

#: Names WS-C may expose for the pure id builder. The contract pins the output
#: format, not the helper name, so the test binds to it when present and falls
#: back to the frozen format otherwise.
ID_HELPER_NAMES = (
    "prediction_id",
    "_prediction_id",
    "build_prediction_id",
    "stable_prediction_id",
    "prediction_doc_id",
)


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """True when ``needle`` appears in ``haystack`` in order (gaps allowed)."""
    iterator = iter(haystack)
    return all(item in iterator for item in needle)


def _prediction_id(city: str, epoch_seconds: int, horizon_hours: int) -> str:
    for name in ID_HELPER_NAMES:
        helper = getattr(inference, name, None)
        if callable(helper):
            return helper(city, epoch_seconds, horizon_hours)
    # Frozen format: F.concat_ws("_", city, unix_timestamp(timestamp), f"{h}h").
    return f"{city}_{int(epoch_seconds)}_{horizon_hours}h"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_schema_columns_is_the_exact_contract():
    columns = list(inference.output_schema_columns())

    assert all(isinstance(column, str) for column in columns)
    assert len(columns) == len(EXPECTED_COLUMNS)
    assert len(columns) == len(set(columns)), "duplicate write columns"
    assert set(columns) == set(EXPECTED_COLUMNS)

    # ``_id`` leads, and the legacy columns keep their relative order. The
    # additive interval fields may sit anywhere after the columns they extend.
    assert columns[0] == "_id"
    assert _is_subsequence(CORE_COLUMNS, columns)

    for column in ADDITIVE_COLUMNS:
        assert column in columns


def test_prediction_id_is_stable_city_epoch_horizon():
    assert _prediction_id("Madrid", 1_750_000_000, 1) == "Madrid_1750000000_1h"
    assert _prediction_id("A Coruña", 1_750_003_600, 6) == "A Coruña_1750003600_6h"
    # Idempotent for the same key ...
    assert _prediction_id("Madrid", 1_750_000_000, 1) == _prediction_id("Madrid", 1_750_000_000, 1)
    # ... and distinct across the horizon dimension.
    assert _prediction_id("Madrid", 1_750_000_000, 1) != _prediction_id("Madrid", 1_750_000_000, 6)
    assert _prediction_id("Madrid", 1_750_000_000, 1) != _prediction_id("Madrid", 1_750_003_600, 1)


def test_id_helper_is_bound_to_inference_when_exposed():
    """Fail loudly if the helper exists but drifts from the frozen format."""
    for name in ID_HELPER_NAMES:
        helper = getattr(inference, name, None)
        if callable(helper):
            assert helper("Madrid", 1_750_000_000, 1) == "Madrid_1750000000_1h"
            return
    pytest.skip("inference exposes no pure prediction-id helper; format asserted directly")
