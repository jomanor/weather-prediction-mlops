"""Registry/prediction document mapping for the Batch 3 additive fields.

These fields are written by the Spark jobs; the API only passes them through.
The tests pin the two failure modes: inventing a value for a key the document
does not carry, and dropping the value when it is a stringly-typed number.
"""

from datetime import datetime, timedelta, timezone

from app.repositories.model_repo import ModelRepository, model_from_doc
from app.repositories.prediction_repo import PredictionRepository, prediction_from_doc
from tests.fakes import FakeMongoCollection, FakeMongoDb

UTC = timezone.utc
SEEN_AT = datetime(2026, 9, 23, 13, tzinfo=UTC)


def _doc(**overrides) -> dict:
    doc = {"city": "Madrid", "source_timestamp": SEEN_AT}
    doc.update(overrides)
    return doc


def test_prediction_maps_interval_fields_and_coerces_numbers():
    prediction = prediction_from_doc(
        _doc(
            predicted_temperature=23.6,
            temp_lower="21.7",
            temp_upper=25.7,
            interval_level="0.8",
        )
    )

    assert prediction is not None
    assert (prediction.temp_lower, prediction.temp_upper) == (21.7, 25.7)
    assert prediction.interval_level == 0.8


def test_prediction_legacy_doc_keeps_interval_fields_null():
    prediction = prediction_from_doc(_doc(predicted_temperature=23.6))

    assert prediction is not None
    assert prediction.temp_lower is None
    assert prediction.temp_upper is None
    assert prediction.interval_level is None


def test_prediction_without_source_timestamp_is_dropped():
    assert prediction_from_doc({"city": "Madrid"}) is None


def test_model_exposes_honest_metrics_and_registry_blocks():
    model = model_from_doc(
        {
            "model_name": "temp_prediction_1h_GradientBoostedTrees",
            "timestamp": SEEN_AT,
            "metrics": {
                "rmse": 1.44,
                "mae": 1.12,
                "r2": 0.91,
                "persistence_rmse": 3.05,
                "climatology_rmse": 3.41,
                "skill_score": "0.53",
                "brier": None,
                "persistence_brier": 0.14,
                "prevalence": 0.18,
                "coverage": 0.79,
            },
            "split": {"kind": "temporal", "test_start": "2026-09-19T00:00:00Z"},
            "interval": {"level": 0.8, "lower_offset": -1.9, "upper_offset": 2.1},
            "commit": "abc1234",
        }
    )

    assert model is not None and model.metrics is not None
    assert model.metrics.persistence_rmse == 3.05
    assert model.metrics.climatology_rmse == 3.41
    assert model.metrics.skill_score == 0.53
    assert model.metrics.brier is None
    assert model.metrics.persistence_brier == 0.14
    assert model.metrics.prevalence == 0.18
    assert model.metrics.coverage == 0.79
    assert model.split == {"kind": "temporal", "test_start": "2026-09-19T00:00:00Z"}
    assert model.interval == {"level": 0.8, "lower_offset": -1.9, "upper_offset": 2.1}
    assert model.commit == "abc1234"


def test_model_missing_blocks_stay_none_never_invented():
    model = model_from_doc({"model_name": "rain_prediction_1h_RandomForest"})

    assert model is not None
    assert model.metrics is None
    assert model.split is None
    assert model.interval is None
    assert model.commit is None


def test_model_non_dict_blocks_and_non_string_commit_are_dropped():
    model = model_from_doc(
        {
            "model_name": "rain_prediction_1h_RandomForest",
            "split": "temporal",
            "interval": 0.8,
            "commit": 42,
        }
    )

    assert model is not None
    assert model.split is None
    assert model.interval is None
    assert model.commit is None


async def test_list_models_sorts_by_model_name_then_timestamp():
    registry = FakeMongoCollection(
        [
            {"model_name": "b_model", "timestamp": SEEN_AT},
            {"model_name": "a_model", "timestamp": SEEN_AT},
        ]
    )
    db = FakeMongoDb(model_registry=registry)

    models = await ModelRepository(db).list_models()

    assert registry.cursors[0].sort_spec == [("model_name", 1), ("timestamp", -1)]
    assert [model.name for model in models] == ["a_model", "b_model"]


# ---------------------------------------------------------------------------
# Batch 4, Contract 3: additive diagnostics mapping
# ---------------------------------------------------------------------------


def test_model_maps_diagnostics_slices_and_drift():
    model = model_from_doc(
        {
            "model_name": "temp_prediction_1h_GradientBoostedTrees",
            "diagnostics": {
                "by_city": [
                    {
                        "label": "Madrid",
                        "n": 24,
                        "rmse": "1.2",
                        "mae": 1.0,
                        "bias": -0.1,
                        "brier": None,
                    }
                ],
                "by_hour_of_day": [
                    {"label": "0", "n": 10, "rmse": 1.3, "mae": 1.1, "bias": 0.0, "brier": None}
                ],
                "by_rain_bucket": [{"label": "dry", "n": 40, "brier": "0.12"}],
                "drift_psi": {"temperature": "0.08", "humidity": None},
            },
        }
    )

    assert model is not None and model.diagnostics is not None
    diagnostics = model.diagnostics
    assert diagnostics.by_city[0].label == "Madrid"
    assert diagnostics.by_city[0].n == 24
    assert diagnostics.by_city[0].rmse == 1.2  # stringly-typed number coerced
    assert diagnostics.by_city[0].brier is None
    assert diagnostics.by_hour_of_day[0].label == "0"
    assert diagnostics.by_rain_bucket[0].brier == 0.12
    assert diagnostics.by_rain_bucket[0].rmse is None
    assert diagnostics.drift_psi == {"temperature": 0.08, "humidity": None}


def test_model_without_diagnostics_is_none_never_invented():
    model = model_from_doc({"model_name": "rain_prediction_1h_RandomForest"})

    assert model is not None
    assert model.diagnostics is None


def test_model_non_dict_diagnostics_is_dropped():
    model = model_from_doc({"model_name": "rain_prediction_1h_RandomForest", "diagnostics": []})

    assert model is not None
    assert model.diagnostics is None


def test_model_diagnostics_skips_malformed_slices():
    model = model_from_doc(
        {
            "model_name": "rain_prediction_1h_RandomForest",
            "diagnostics": {"by_city": ["nope", {"n": 5}], "drift_psi": "x"},
        }
    )

    assert model is not None and model.diagnostics is not None
    assert model.diagnostics.by_city == []
    assert model.diagnostics.by_hour_of_day == []
    assert model.diagnostics.by_rain_bucket == []
    assert model.diagnostics.drift_psi == {}


# ---------------------------------------------------------------------------
# Batch 4, Contract 2: latest_per_city groups by (city, horizon_hours)
# ---------------------------------------------------------------------------

_PREDICTION_DOCS = [
    {
        "city": "Madrid",
        "source_timestamp": SEEN_AT,
        "horizon_hours": 1,
        "prediction_timestamp": SEEN_AT,
        "predicted_temperature": 21.5,
    },
    {
        "city": "Madrid",
        "source_timestamp": SEEN_AT - timedelta(hours=2),
        "horizon_hours": 1,
        "prediction_timestamp": SEEN_AT - timedelta(hours=2),
        "predicted_temperature": 20.0,
    },
    {
        "city": "Madrid",
        "source_timestamp": SEEN_AT,
        "horizon_hours": 3,
        "prediction_timestamp": SEEN_AT,
        "predicted_temperature": 22.0,
    },
]


async def test_latest_per_city_groups_per_horizon_with_index_backed_sort():
    predictions = FakeMongoCollection(_PREDICTION_DOCS)
    repo = PredictionRepository(FakeMongoDb(weather_predictions=predictions))

    result = await repo.latest_per_city()

    call = predictions.aggregate_calls[0]
    assert call["allowDiskUse"] is True
    # Inner sort must match the existing
    # ``{city: 1, horizon_hours: 1, prediction_timestamp: -1}`` index; the outer
    # sort is the response order.
    assert call["pipeline"][0]["$sort"] == {
        "city": 1,
        "horizon_hours": 1,
        "prediction_timestamp": -1,
    }
    assert call["pipeline"][-1]["$sort"] == {"city": 1, "horizon_hours": 1}
    # One row per (city, horizon): the older h1 doc is grouped away.
    assert [(p.horizon_hours, p.predicted_temperature) for p in result] == [(1, 21.5), (3, 22.0)]


async def test_latest_per_city_filters_horizon():
    predictions = FakeMongoCollection(_PREDICTION_DOCS)
    repo = PredictionRepository(FakeMongoDb(weather_predictions=predictions))

    result = await repo.latest_per_city(horizon=3)

    pipeline = predictions.aggregate_calls[0]["pipeline"]
    assert pipeline[0] == {"$match": {"horizon_hours": 3}}
    assert [p.horizon_hours for p in result] == [3]


async def test_for_city_filters_by_horizon():
    predictions = FakeMongoCollection(_PREDICTION_DOCS)
    repo = PredictionRepository(FakeMongoDb(weather_predictions=predictions))

    result = await repo.for_city("Madrid", horizon=3)

    assert predictions.find_calls[0][0] == {"city": "Madrid", "horizon_hours": 3}
    assert [p.horizon_hours for p in result] == [3]
