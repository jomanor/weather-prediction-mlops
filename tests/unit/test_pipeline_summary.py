"""M4 drift warning from ``scripts/pipeline_summary.py``.

Covers the pure helpers only: the summary must never fail the pipeline, so a
legacy registry doc without ``diagnostics`` is skipped, non-finite PSI values
are ignored, and a model is flagged only when ``maxDriftPsi > 0.25``.
"""

from __future__ import annotations

import math
import sys

from tests.conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pipeline_summary as ps  # noqa: E402


def _model(**overrides):
    doc = {
        "model_type": "temp",
        "horizon_hours": 1,
        "model_name": "GBTRegressor",
        "diagnostics": {"drift_psi": {"temperature": 0.08, "humidity": 0.31}},
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# max_drift_psi
# ---------------------------------------------------------------------------


def test_max_drift_psi_is_the_max_finite_value():
    assert ps.max_drift_psi(_model()) == 0.31


def test_max_drift_psi_skips_missing_and_malformed_diagnostics():
    assert ps.max_drift_psi({}) is None
    assert ps.max_drift_psi({"diagnostics": None}) is None
    assert ps.max_drift_psi({"diagnostics": "nope"}) is None
    assert ps.max_drift_psi({"diagnostics": {}}) is None
    assert ps.max_drift_psi({"diagnostics": {"drift_psi": {}}}) is None
    assert ps.max_drift_psi({"diagnostics": {"drift_psi": None}}) is None
    assert ps.max_drift_psi(None) is None


def test_max_drift_psi_ignores_non_finite_and_boolean_values():
    doc = {
        "diagnostics": {
            "drift_psi": {
                "temperature": None,
                "humidity": float("nan"),
                "pressure": float("inf"),
                "wind_speed": True,  # bool is an int subclass; not a PSI
                "ok": 0.2,
            }
        }
    }
    assert ps.max_drift_psi(doc) == 0.2
    # A float('nan') would poison max(); assert it did not leak through.
    assert not math.isnan(ps.max_drift_psi(doc))


# ---------------------------------------------------------------------------
# drift_report
# ---------------------------------------------------------------------------


def test_drift_report_counts_scored_models_and_skips_legacy_docs():
    calm = _model(diagnostics={"drift_psi": {"temperature": 0.08}})
    lines, warning = ps.drift_report([calm, {"model_name": "legacy"}])
    assert "1 model(s) scored" in lines[0]
    assert warning is None


def test_drift_report_warns_above_threshold_and_lists_models():
    calm = _model(diagnostics={"drift_psi": {"temperature": 0.08}})
    drifting = _model(
        model_type="rain",
        horizon_hours=24,
        model_name="RandomForest",
        diagnostics={"drift_psi": {"humidity": 0.9}},
    )
    lines, warning = ps.drift_report([calm, drifting])
    assert "2 model(s) scored" in lines[0]
    assert "1 above PSI 0.25" in lines[0]
    assert warning is not None
    assert "rain 24h RandomForest" in warning
    assert "maxDriftPsi=0.900" in warning
    assert any("drift warning" in line for line in lines)


def test_drift_report_threshold_is_strictly_greater_than():
    at_threshold = _model(diagnostics={"drift_psi": {"temperature": 0.25}})
    just_over = _model(diagnostics={"drift_psi": {"temperature": 0.2501}})
    assert ps.drift_report([at_threshold])[1] is None
    assert ps.drift_report([just_over])[1] is not None


def test_drift_report_is_empty_but_complete_without_diagnostics():
    lines, warning = ps.drift_report([])
    assert lines == ["- `model_registry` drift: 0 model(s) scored, 0 above PSI 0.25"]
    assert warning is None


def test_drift_threshold_is_the_contract_value():
    """The warn threshold is the contract's 0.25."""
    assert ps.DRIFT_PSI_WARN_THRESHOLD == 0.25
