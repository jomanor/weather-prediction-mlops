"""M4 drift summary from ``scripts/pipeline_summary.py``.

Covers the pure helpers only: the summary must never fail the pipeline, so a
legacy registry doc without ``diagnostics`` is skipped, non-finite PSI values
are ignored, and the top drifting features are listed per model. The actionable
``::warning`` fires only when a model's test ``skill_score`` is at or below the
persistence baseline (``SKILL_SCORE_WARN_THRESHOLD = 0.0``) — not on PSI, whose
equal-width ``pressure`` value saturates at ~0.51 for every model.
"""

from __future__ import annotations

import math
import sys

from tests.conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pipeline_summary as ps  # noqa: E402


def _model(**overrides):
    doc = {
        "model_type": "GradientBoostedTrees",
        "target": "temperature",
        "horizon_hours": 1,
        "model_name": "temp_prediction_1h_GradientBoostedTrees",
        "metrics": {"skill_score": 0.4},
        "diagnostics": {
            "drift_psi": {
                "pressure": 0.51,
                "temperature": 0.26,
                "specific_humidity": 0.18,
                "humidity": 0.07,
                "wind_speed": 0.06,
                "precipitation": 0.02,
            }
        },
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# _finite_drift / max_drift_psi
# ---------------------------------------------------------------------------


def test_max_drift_psi_is_the_max_finite_value():
    assert ps.max_drift_psi(_model()) == 0.51


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
# top_drift_features
# ---------------------------------------------------------------------------


def test_top_drift_features_orders_descending_and_truncates():
    doc = _model(diagnostics={"drift_psi": {"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.9, "e": 0.7}})
    assert ps.top_drift_features(doc, n=3) == [("d", 0.9), ("e", 0.7), ("b", 0.5)]


def test_top_drift_features_skips_non_finite_and_boolean():
    doc = _model(
        diagnostics={"drift_psi": {"a": float("nan"), "b": True, "c": 0.2, "d": float("inf")}}
    )
    assert ps.top_drift_features(doc) == [("c", 0.2)]


def test_top_drift_features_empty_for_legacy_or_missing_diagnostics():
    assert ps.top_drift_features({}) == []
    assert ps.top_drift_features({"diagnostics": {"drift_psi": {}}}) == []
    assert ps.top_drift_features(_model(diagnostics={"drift_psi": {"p": None}})) == []


# ---------------------------------------------------------------------------
# skill_score
# ---------------------------------------------------------------------------


def test_skill_score_returns_finite_value():
    assert ps.skill_score(_model()) == 0.4


def test_skill_score_none_when_absent_malformed_or_non_finite():
    assert ps.skill_score({}) is None
    assert ps.skill_score({"metrics": None}) is None
    assert ps.skill_score({"metrics": {}}) is None
    assert ps.skill_score(_model(metrics={"skill_score": None})) is None
    assert ps.skill_score(_model(metrics={"skill_score": float("nan")})) is None
    assert ps.skill_score(_model(metrics={"skill_score": True})) is None


# ---------------------------------------------------------------------------
# drift_report
# ---------------------------------------------------------------------------


def test_drift_report_lists_top_features_and_no_warning_when_healthy():
    lines, warning = ps.drift_report([_model()])
    assert lines[0] == "- `model_registry` drift: 1 model(s) scored"
    assert any("temperature 1h (GradientBoostedTrees)" in line for line in lines)
    assert any("pressure=0.510" in line for line in lines)
    assert any("specific_humidity=0.180" in line for line in lines)
    assert warning is None


def test_drift_report_warns_when_skill_at_or_below_persistence_baseline():
    degrading = _model(target="temperature", horizon_hours=24, metrics={"skill_score": -0.1056})
    lines, warning = ps.drift_report([_model(), degrading])
    assert "2 model(s) scored" in lines[0]
    assert warning is not None
    assert "temperature 24h (GradientBoostedTrees)" in warning
    assert "skill_score=-0.106" in warning
    assert any("skill warning" in line for line in lines)


def test_drift_report_warns_at_exactly_the_threshold():
    # skill_score == 0.0 means "no better than persistence" -> still actionable.
    at_baseline = _model(metrics={"skill_score": 0.0})
    _, warning = ps.drift_report([at_baseline])
    assert warning is not None


def test_drift_report_does_not_warn_when_skill_is_positive():
    healthy = _model(metrics={"skill_score": 0.0001})
    _, warning = ps.drift_report([healthy])
    assert warning is None


def test_drift_report_skips_legacy_docs_and_reports_zero_scored():
    lines, warning = ps.drift_report([{"model_name": "legacy"}])
    assert lines == ["- `model_registry` drift: 0 model(s) scored"]
    assert warning is None


def test_drift_report_is_empty_but_complete():
    lines, warning = ps.drift_report([])
    assert lines == ["- `model_registry` drift: 0 model(s) scored"]
    assert warning is None


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_threshold_constants():
    """The warn trigger is skill at/below the persistence baseline; PSI is
    informational only (top-features count), never a warning."""
    assert ps.SKILL_SCORE_WARN_THRESHOLD == 0.0
    assert ps.TOP_DRIFT_FEATURES == 3
    assert not hasattr(ps, "DRIFT_PSI_WARN_THRESHOLD")
