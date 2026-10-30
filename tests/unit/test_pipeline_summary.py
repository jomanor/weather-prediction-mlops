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
from datetime import datetime, timezone

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


# ---------------------------------------------------------------------------
# Prediction coverage (2026-10-30: 1 of 15 cities scored, silently)
# ---------------------------------------------------------------------------


def test_coverage_warning_threshold_is_ninety_percent():
    assert ps.PREDICTION_COVERAGE_WARN_RATIO == 0.9


def test_coverage_warning_fires_when_one_of_fifteen_cities_is_scored():
    body = ps.coverage_warning(1, 15)
    assert body is not None
    assert "prediction coverage 1/15 cities (7%)" in body


def test_coverage_warning_is_silent_at_and_above_ninety_percent():
    assert ps.coverage_warning(15, 15) is None
    assert ps.coverage_warning(9, 10) is None  # exactly 90 %
    assert ps.coverage_warning(14, 15) is None


def test_coverage_warning_fires_just_below_ninety_percent():
    assert ps.coverage_warning(8, 10) is not None
    assert ps.coverage_warning(13, 15) is not None


def test_coverage_warning_is_silent_when_no_cities_are_known():
    # An empty/fresh weather_features must not produce a false alarm.
    assert ps.coverage_warning(0, 0) is None


def _coverage_db(rows, cities):
    """mongomock db with ``rows`` predictions and ``cities`` featured stations."""
    import mongomock

    db = mongomock.MongoClient().weather_db
    if rows:
        db.weather_predictions.insert_many(rows)
    for city in cities:
        db.weather_features.insert_one(
            {"city": city, "timestamp": datetime(2026, 10, 30, 9, tzinfo=timezone.utc)}
        )
    return db


class TestPredictionCoverageSection:
    def test_rows_and_cities_are_reported_per_horizon(self):
        rows = [
            {"city": "Madrid", "horizon_hours": 1, "source_timestamp": datetime(2026, 10, 30, 9)},
            {"city": "Valencia", "horizon_hours": 1, "source_timestamp": datetime(2026, 10, 30, 9)},
            {"city": "Madrid", "horizon_hours": 3, "source_timestamp": datetime(2026, 10, 29, 15)},
        ]
        db = _coverage_db(rows, ["Madrid", "Valencia"])

        stats = ps.prediction_horizon_stats(db)
        assert sorted(stats) == [1, 3]
        assert stats[1]["rows"] == 2 and stats[1]["cities"] == 2
        assert stats[1]["cities_at_newest"] == 2
        assert stats[3]["rows"] == 1 and stats[3]["cities"] == 1

        lines = ps.prediction_coverage_lines(stats, ps.available_cities(db))
        joined = "\n".join(lines)
        assert "1h 2 rows / 2 cities" in joined
        assert "3h 1 rows / 1 cities" in joined
        assert "oldest source_timestamp 2026-10-29T15:00:00Z" in joined
        assert "coverage: 1/2 cities (50%)" in joined

    def test_newest_batch_coverage_is_what_collapses(self):
        """History keeps 15 cities while the newest batch holds one: a lifetime
        city count would hide the incident, ``cities_at_newest`` does not."""
        rows = [
            {"city": f"City{index}", "horizon_hours": 1, "source_timestamp": datetime(2026, 10, 29)}
            for index in range(15)
        ] + [
            {"city": "Pasadena", "horizon_hours": 1, "source_timestamp": datetime(2026, 9, 25)},
        ]
        db = _coverage_db(rows, [f"City{index}" for index in range(15)])

        stats = ps.prediction_horizon_stats(db)
        assert stats[1]["cities"] == 16  # lifetime rows still name 16 cities
        assert stats[1]["cities_at_newest"] == 15

    def test_empty_predictions_report_no_rows_without_warning(self):
        db = _coverage_db([], ["Madrid"])
        stats = ps.prediction_horizon_stats(db)
        assert stats == {}
        assert ps.prediction_coverage_lines(stats, 1) == [
            "- `weather_predictions` coverage: no prediction rows"
        ]

    def test_aggregation_failure_is_swallowed(self):
        class _Boom:
            def __getitem__(self, _name):
                raise RuntimeError("atlas unavailable")

        assert ps.prediction_horizon_stats(_Boom()) == {}
        assert ps.available_cities(_Boom()) == 0

    def test_summary_emits_the_coverage_warning_only_below_ninety_percent(self):
        """Full coverage: no warning. One of fifteen cities: ``::warning``."""
        cities = [f"City{index}" for index in range(15)]
        full = ps.summary_and_warnings(
            _coverage_db(
                [
                    {
                        "city": city,
                        "horizon_hours": 1,
                        "source_timestamp": datetime(2026, 10, 30, 9),
                    }
                    for city in cities
                ],
                cities,
            )
        )
        full_lines, full_warnings = full
        assert not any(title == "Prediction coverage" for title, _ in full_warnings)
        assert any("coverage: 15/15 cities (100%)" in line for line in full_lines)

        partial = ps.summary_and_warnings(
            _coverage_db(
                [
                    {
                        "city": "Pasadena",
                        "horizon_hours": 1,
                        "source_timestamp": datetime(2026, 9, 25),
                    }
                ],
                cities,
            )
        )
        partial_lines, partial_warnings = partial
        titles = [title for title, _ in partial_warnings]
        assert "Prediction coverage" in titles
        body = dict(partial_warnings)["Prediction coverage"]
        assert "1/15 cities (7%)" in body
        # Existing sections are intact.
        assert any(line.startswith("- `weather_features`:") for line in partial_lines)
        assert any(line.startswith("- `model_registry`") for line in partial_lines)
        assert any(line.startswith("- `weather_predictions`") for line in partial_lines)

    def test_summary_warning_starts_below_ninety_percent_not_below_fifty(self):
        """13 of 15 cities (87 %) must warn: pins the 90 % boundary end to end."""
        cities = [f"City{index}" for index in range(15)]
        _, warnings = ps.summary_and_warnings(
            _coverage_db(
                [
                    {
                        "city": city,
                        "horizon_hours": 1,
                        "source_timestamp": datetime(2026, 10, 30, 9),
                    }
                    for city in cities[:13]
                ],
                cities,
            )
        )
        assert "Prediction coverage" in [title for title, _ in warnings]

    def test_main_prints_the_annotation_on_stderr_only(self, capsys, monkeypatch):
        import mongomock

        cities = [f"City{index}" for index in range(15)]
        db = _coverage_db(
            [{"city": "Pasadena", "horizon_hours": 1, "source_timestamp": datetime(2026, 9, 25)}],
            cities,
        )
        # ``main`` owns its client: hand it one whose weather_db is pre-seeded.
        monkeypatch.setenv("MONGO_URI", "mongodb://unit-test")
        client = mongomock.MongoClient()
        client.weather_db.weather_predictions.insert_many(db.weather_predictions.find({}))
        client.weather_db.weather_features.insert_many(db.weather_features.find({}))
        monkeypatch.setattr(ps, "MongoClient", lambda *a, **k: client)

        assert ps.main() == 0
        captured = capsys.readouterr()
        assert "### Pipeline data snapshot" in captured.out
        assert "::warning" not in captured.out, "annotations must not pollute the step summary"
        assert (
            "::warning title=Prediction coverage::prediction coverage 1/15 cities" in captured.err
        )
