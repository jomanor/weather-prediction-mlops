"""Registry/prediction document mapping for the Batch 3 additive fields.

These fields are written by the Spark jobs; the API only passes them through.
The tests pin the two failure modes: inventing a value for a key the document
does not carry, and dropping the value when it is a stringly-typed number.
"""

from datetime import datetime, timezone

from app.repositories.model_repo import ModelRepository, model_from_doc
from app.repositories.prediction_repo import prediction_from_doc
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
