"""``scripts/promote_model.py`` — the human M5 promotion gate.

Covers selector resolution, promote/rollback slot semantics, idempotency (a
re-run must not re-stamp ``promoted_at`` or re-stage anything), the dry-run mode,
the refuse-to-act cases, and the CLI exit codes. Everything runs against the
in-memory registry fake: no Mongo, no Spark.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest
from tests.conftest import REPO_ROOT
from tests.unit.fake_registry import FakeDb, FakeRegistryCollection, dt, make_entry

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "spark" / "spark-jobs"))

import model_selection as ms  # noqa: E402
import promote_model as pm  # noqa: E402

NOW = datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc)


def _db(*docs) -> FakeDb:
    return FakeDb({ms.REGISTRY_COLLECTION: FakeRegistryCollection(list(docs))})


def _registry(db: FakeDb) -> FakeRegistryCollection:
    return db[ms.REGISTRY_COLLECTION]


def _stages(db: FakeDb) -> dict[str, str]:
    return {doc["_id"]: doc["stage"] for doc in _registry(db).docs}


# ---------------------------------------------------------------------------
# Selector resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "horizon", "expected"),
    [
        ("temp_prediction_", 1, "temp_prediction_1h_"),
        ("temp_prediction", 12, "temp_prediction_12h_"),
        ("temp_prediction_1h_", 3, "temp_prediction_1h_"),  # name wins over the flag
        ("temp_prediction_1h_GradientBoostedTrees", None, "temp_prediction_1h_GradientBoostedTrees"),
        ("temp_prediction_", None, "temp_prediction_"),
        ("rain_prediction_1h_G", 24, "rain_prediction_1h_G"),  # horizon already in name
    ],
)
def test_resolve_prefix(name, horizon, expected):
    assert pm.resolve_prefix(name, horizon) == expected


@pytest.mark.parametrize(
    ("env", "expected"),
    [({"GITHUB_ACTOR": "alice"}, "alice"), ({}, None)],
)
def test_resolve_actor_prefers_github_actor(monkeypatch, env, expected):
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(pm.getpass, "getuser", lambda: "localuser")
    assert pm.resolve_actor() == (expected or "localuser")


def test_resolve_actor_unknown_when_nothing_available(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTOR", raising=False)

    def boom():
        raise OSError("no passwd entry")

    monkeypatch.setattr(pm.getpass, "getuser", boom)
    assert pm.resolve_actor() == "unknown"


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------


def test_promote_newest_staging_supersedes_slot_peer():
    db = _db(
        make_entry(
            "temp_prediction_1h_Old", doc_id="01", stage="production", timestamp=dt(1), version="v1"
        ),
        make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(3)),
        make_entry("temp_prediction_1h_Older", doc_id="03", stage="staging", timestamp=dt(2)),
        # Different slot: must stay untouched.
        make_entry("temp_prediction_3h_GBT", doc_id="04", stage="production", horizon=3),
        make_entry("rain_prediction_1h_GBT", doc_id="05", stage="production", target="rain"),
        make_entry("temp_prediction_1h_Done", doc_id="06", stage="superseded", timestamp=dt(9)),
    )

    result = pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW)

    assert _stages(db) == {
        "01": "superseded",
        "02": "production",
        "03": "staging",
        "04": "production",
        "05": "production",
        "06": "superseded",
    }
    promoted = _registry(db).find_one({"_id": "02"})
    assert promoted["promoted_at"] == NOW
    assert promoted["promoted_by"] == "alice"
    superseded = _registry(db).find_one({"_id": "01"})
    assert superseded["superseded_at"] == NOW
    assert superseded["superseded_by"] == "alice"

    assert result["changed"] is True
    assert [row["stage_after"] for row in result["rows"]] == ["production", "superseded"]


def test_promote_is_idempotent_when_reapplied_to_the_same_doc():
    db = _db(
        make_entry("temp_prediction_1h_Old", doc_id="01", stage="production", timestamp=dt(1)),
        make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(3)),
    )

    pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW)
    first = _registry(db).find_one({"_id": "02"})
    assert first["promoted_at"] == NOW

    later = datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc)
    result = pm.promote(db, prefix="temp_prediction_1h_", doc_id="02", actor="bob", now=later)

    second = _registry(db).find_one({"_id": "02"})
    assert second["promoted_at"] == NOW  # original stamp preserved
    assert second["promoted_by"] == "alice"
    assert _stages(db) == {"01": "superseded", "02": "production"}
    assert result["changed"] is False


def test_promote_by_id_targets_one_doc_and_ignores_staging_order():
    db = _db(
        make_entry("temp_prediction_1h_Prod", doc_id="01", stage="production", timestamp=dt(1)),
        make_entry("temp_prediction_1h_Newer", doc_id="02", stage="staging", timestamp=dt(5)),
        make_entry("temp_prediction_1h_Chosen", doc_id="03", stage="staging", timestamp=dt(2)),
    )

    pm.promote(db, prefix=None, doc_id="03", actor="alice", now=NOW)

    assert _stages(db) == {
        "01": "superseded",
        "02": "staging",
        "03": "production",
    }


def test_promote_does_not_touch_other_collections():
    db = _db(make_entry("temp_prediction_1h_New", doc_id="02", stage="staging"))
    db["weather_predictions"]  # accessor exists but must stay empty
    pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW)
    assert db["weather_predictions"].docs == []


def test_promote_dry_run_writes_nothing():
    db = _db(
        make_entry("temp_prediction_1h_Old", doc_id="01", stage="production", timestamp=dt(1)),
        make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(3)),
    )

    result = pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW, dry_run=True)

    assert _stages(db) == {"01": "production", "02": "staging"}
    assert result["dry_run"] is True
    assert result["rows"][0]["stage_after"] == "production"
    assert result["rows"][0]["model_name"] == "temp_prediction_1h_New"


def test_prefix_rerun_with_older_staging_left_is_a_noop():
    # QA repro: promote v3, then v1/v2 remain staging; a retried command used to
    # promote v2 and supersede v3. Now the incumbent wins on timestamp.
    db = _db(
        make_entry("temp_prediction_1h_V1", doc_id="01", stage="staging", timestamp=dt(1)),
        make_entry("temp_prediction_1h_V2", doc_id="02", stage="staging", timestamp=dt(2)),
        make_entry("temp_prediction_1h_V3", doc_id="03", stage="production", timestamp=dt(3)),
    )

    result = pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW)

    assert result["changed"] is False
    assert result["noop"] == "already production"
    assert _stages(db) == {"01": "staging", "02": "staging", "03": "production"}
    assert _registry(db).updates == []  # no writes at all


def test_prefix_with_only_production_is_a_clean_noop():
    db = _db(
        make_entry("temp_prediction_1h_V3", doc_id="03", stage="production", timestamp=dt(3))
    )
    result = pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW)
    assert result["changed"] is False
    assert result["target"]["_id"] == "03"
    assert _registry(db).updates == []


def test_prefix_promotes_a_newer_staging_candidate():
    db = _db(
        make_entry("temp_prediction_1h_V3", doc_id="03", stage="production", timestamp=dt(3)),
        make_entry("temp_prediction_1h_V4", doc_id="04", stage="staging", timestamp=dt(4)),
    )
    result = pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW)
    assert result["changed"] is True
    assert _stages(db) == {"03": "superseded", "04": "production"}


def test_supersede_is_written_before_the_promotion():
    # A crash between the two writes must leave zero production (staging
    # fallback), never two production documents.
    db = _db(
        make_entry("temp_prediction_1h_Old", doc_id="01", stage="production", timestamp=dt(1)),
        make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(3)),
    )
    pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW)

    updates = _registry(db).updates
    assert [u[0]["_id"] for u in updates] == ["01", "02"]
    assert updates[0][1]["$set"]["stage"] == "superseded"
    assert updates[1][1]["$set"]["stage"] == "production"


def test_promote_refuses_a_missing_gridfs_artifact():
    class _FakeFS:
        def __init__(self, ids=()):
            self.ids = set(ids)

        def exists(self, file_id):
            return file_id in self.ids

    db = _db(
        make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(3))
    )
    with pytest.raises(pm.PromotionError, match="does not exist"):
        pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW, fs=_FakeFS())
    assert _stages(db) == {"02": "staging"}
    assert _registry(db).updates == []

    # With the artifact present the same promote succeeds.
    pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW, fs=_FakeFS({"gridfs-id"}))
    assert _stages(db) == {"02": "production"}


def test_horizon_less_target_only_supersedes_horizon_less_peers():
    target = make_entry("temp_prediction_Legacy", doc_id="03", stage="staging")
    target.pop("horizon_hours")
    peer = make_entry("temp_prediction_LegacyOld", doc_id="02", stage="production")
    peer.pop("horizon_hours")
    db = _db(
        target,
        peer,
        make_entry("temp_prediction_1h_Prod", doc_id="01", stage="production", horizon=1),
    )

    pm.promote(db, prefix=None, doc_id="03", actor="alice", now=NOW)

    assert _stages(db) == {"01": "production", "02": "superseded", "03": "production"}


def test_promote_without_match_raises():
    db = _db(make_entry("rain_prediction_1h_GBT", doc_id="01", stage="staging", target="rain"))
    with pytest.raises(pm.PromotionError, match="No staging model matches"):
        pm.promote(db, prefix="temp_prediction_1h_")


def test_promote_refuses_target_without_gridfs_file_id():
    db = _db(
        make_entry(
            "temp_prediction_1h_New", doc_id="02", stage="staging", gridfs_file_id=None
        )
    )
    with pytest.raises(pm.PromotionError, match="gridfs_file_id"):
        pm.promote(db, prefix="temp_prediction_1h_", actor="alice", now=NOW)
    assert _stages(db) == {"02": "staging"}


def test_promote_unknown_id_raises():
    db = _db(make_entry("temp_prediction_1h_New", doc_id="02", stage="staging"))
    with pytest.raises(pm.PromotionError, match="No model_registry document"):
        pm.promote(db, doc_id="ff")


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_restores_most_recent_superseded_and_supersedes_current():
    db = _db(
        make_entry(
            "temp_prediction_1h_Oldest",
            doc_id="01",
            stage="superseded",
            timestamp=dt(1),
            superseded_at=dt(2),
        ),
        make_entry(
            "temp_prediction_1h_Prev",
            doc_id="02",
            stage="superseded",
            timestamp=dt(3),
            superseded_at=dt(8),
            superseded_by="alice",
        ),
        make_entry("temp_prediction_1h_Current", doc_id="03", stage="production", timestamp=dt(9)),
        make_entry("temp_prediction_3h_GBT", doc_id="04", stage="production", horizon=3),
    )

    result = pm.rollback(db, prefix="temp_prediction_1h_", actor="bob", now=NOW)

    assert _stages(db) == {
        "01": "superseded",
        "02": "production",  # most recently superseded (dt(8))
        "03": "superseded",
        "04": "production",  # other slot untouched
    }
    restored = _registry(db).find_one({"_id": "02"})
    assert restored["promoted_at"] == NOW
    assert restored["promoted_by"] == "bob"
    # The stale supersede audit is cleared: a production doc is not superseded.
    assert "superseded_at" not in restored
    assert "superseded_by" not in restored
    assert _registry(db).find_one({"_id": "03"})["superseded_by"] == "bob"
    assert result["action"] == "rollback"


def test_rollback_without_candidate_raises():
    db = _db(make_entry("temp_prediction_1h_Current", doc_id="03", stage="production"))
    with pytest.raises(pm.PromotionError, match="No superseded model matches"):
        pm.rollback(db, prefix="temp_prediction_1h_")


def test_rollback_refuses_unloadable_candidate():
    db = _db(
        make_entry(
            "temp_prediction_1h_Gone",
            doc_id="02",
            stage="superseded",
            superseded_at=dt(8),
            gridfs_file_id=None,
        ),
        make_entry("temp_prediction_1h_Current", doc_id="03", stage="production"),
    )
    with pytest.raises(pm.PromotionError, match="gridfs_file_id"):
        pm.rollback(db, prefix="temp_prediction_1h_")
    assert _stages(db) == {"02": "superseded", "03": "production"}


def test_rollback_does_not_oscillate_on_a_rerun():
    db = _db(
        make_entry(
            "temp_prediction_1h_Prev",
            doc_id="02",
            stage="superseded",
            superseded_at=dt(8),
            timestamp=dt(3),
        ),
        make_entry("temp_prediction_1h_Current", doc_id="03", stage="production", timestamp=dt(9)),
    )
    pm.rollback(db, prefix="temp_prediction_1h_", actor="bob", now=NOW)
    state = _stages(db)
    assert state == {"02": "production", "03": "superseded"}

    # The doc this rollback just superseded is NOT a rollback candidate, so a
    # re-run (or a retried CI job) cannot flip production back to it.
    with pytest.raises(pm.PromotionError):
        pm.rollback(db, prefix="temp_prediction_1h_", actor="bob", now=NOW)
    assert _stages(db) == state


def test_repeated_rollbacks_walk_back_through_promote_history():
    db = _db(
        make_entry(
            "temp_prediction_1h_Older",
            doc_id="01",
            stage="superseded",
            superseded_at=dt(2),
            timestamp=dt(1),
        ),
        make_entry(
            "temp_prediction_1h_Prev",
            doc_id="02",
            stage="superseded",
            superseded_at=dt(8),
            timestamp=dt(3),
        ),
        make_entry("temp_prediction_1h_Current", doc_id="03", stage="production", timestamp=dt(9)),
    )
    pm.rollback(db, prefix="temp_prediction_1h_", now=NOW)
    assert _stages(db)["02"] == "production"

    pm.rollback(db, prefix="temp_prediction_1h_", now=NOW)
    assert _stages(db) == {"01": "production", "02": "superseded", "03": "superseded"}


def test_rollback_dry_run_writes_nothing():
    db = _db(
        make_entry("temp_prediction_1h_Prev", doc_id="02", stage="superseded", superseded_at=dt(8)),
        make_entry("temp_prediction_1h_Current", doc_id="03", stage="production"),
    )
    result = pm.rollback(db, prefix="temp_prediction_1h_", now=NOW, dry_run=True)
    assert _stages(db) == {"02": "superseded", "03": "production"}
    assert result["dry_run"] is True


# ---------------------------------------------------------------------------
# Table formatting / CLI
# ---------------------------------------------------------------------------


def test_format_table_has_a_row_per_model_with_before_and_after():
    table = pm.format_table(
        [
            {"model_name": "temp_prediction_1h_New", "version": "v2",
             "stage_before": "staging", "stage_after": "production"},
            {"model_name": "temp_prediction_1h_Old", "version": "v1",
             "stage_before": "production", "stage_after": "superseded"},
        ]
    )
    lines = table.splitlines()
    assert lines[0].split() == ["model_name", "version", "stage_before", "stage_after"]
    assert "staging" in lines[2] and "production" in lines[2]
    assert "superseded" in lines[3]


def test_format_table_empty():
    assert pm.format_table([]) == "(no changes)"


def _run_cli(monkeypatch, db, argv):
    monkeypatch.setenv("MONGO_URI", "mongodb://example.invalid/weather_db")
    monkeypatch.setenv("GITHUB_ACTOR", "ci-actor")
    monkeypatch.setattr(pm, "MongoClient", lambda uri: _FakeClient(db))
    # The fake database is not a pymongo ``Database``, so the real GridFS
    # constructor refuses it; the artifact check is exercised directly in
    # ``test_promote_refuses_a_missing_gridfs_artifact`` instead.
    monkeypatch.setattr(pm, "GridFS", lambda _db: None)
    return pm.main(argv)


class _FakeClient:
    def __init__(self, db):
        self._db = db
        self.closed = False

    def __getitem__(self, _name):
        return self._db

    def close(self):
        self.closed = True


def test_cli_promote_prints_table_and_exits_zero(monkeypatch, capsys):
    db = _db(make_entry("temp_prediction_1h_New", doc_id="02", stage="staging", timestamp=dt(3)))
    code = _run_cli(monkeypatch, db, ["--model-name", "temp_prediction_", "--horizon", "1"])

    assert code == 0
    out = capsys.readouterr().out
    assert "promote" in out
    assert "ci-actor" in out
    assert "production" in out
    assert _stages(db) == {"02": "production"}
    assert "mongodb://" not in out  # the URI is never printed


def test_cli_rollback_exits_zero(monkeypatch):
    db = _db(
        make_entry("temp_prediction_1h_Prev", doc_id="02", stage="superseded", superseded_at=dt(8))
    )
    code = _run_cli(monkeypatch, db, ["--model-name", "temp_prediction_1h_", "--rollback"])
    assert code == 0
    assert _stages(db) == {"02": "production"}


def test_cli_prefix_rerun_exits_zero_with_explicit_message(monkeypatch, capsys):
    db = _db(
        make_entry("temp_prediction_1h_V2", doc_id="02", stage="staging", timestamp=dt(2)),
        make_entry("temp_prediction_1h_V3", doc_id="03", stage="production", timestamp=dt(3)),
    )
    code = _run_cli(monkeypatch, db, ["--model-name", "temp_prediction_1h_"])
    assert code == 0
    out = capsys.readouterr().out
    assert "already production, nothing to do" in out
    assert _stages(db) == {"02": "staging", "03": "production"}


def test_cli_only_production_exits_zero(monkeypatch, capsys):
    db = _db(
        make_entry("temp_prediction_1h_V3", doc_id="03", stage="production", timestamp=dt(3))
    )
    code = _run_cli(monkeypatch, db, ["--model-name", "temp_prediction_1h_"])
    assert code == 0
    assert "already production, nothing to do" in capsys.readouterr().out


def test_cli_nothing_matching_exits_nonzero(monkeypatch, capsys):
    db = _db()
    code = _run_cli(monkeypatch, db, ["--model-name", "temp_prediction_1h_"])
    assert code == 1
    assert "No staging model matches" in capsys.readouterr().err


def test_cli_without_selector_exits_2(monkeypatch):
    db = _db()
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, db, [])
    assert excinfo.value.code == 2


def test_cli_without_mongo_uri_exits_nonzero(monkeypatch, capsys):
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("MONGO_URL", raising=False)
    db = _db()
    code = pm.main(["--model-name", "temp_prediction_1h_"])
    assert code == 1
    assert "MONGO_URI is not set" in capsys.readouterr().err


def test_cli_invalid_id_exits_nonzero(monkeypatch, capsys):
    db = _db()
    code = _run_cli(monkeypatch, db, ["--id", "not-an-objectid"])
    assert code == 1
    assert "not a valid ObjectId" in capsys.readouterr().err


def test_script_never_mentions_deletes():
    # Promotion is stage-only: no delete_many/drop anywhere in the script.
    source = (REPO_ROOT / "scripts" / "promote_model.py").read_text()
    assert "delete_many" not in source
    assert "drop(" not in source


def test_script_does_not_print_the_uri():
    source = (REPO_ROOT / "scripts" / "promote_model.py").read_text()
    assert "print(uri" not in source
    assert "print(f\"{uri" not in source


# ---------------------------------------------------------------------------
# Slot semantics: one production document per (family, horizon)
# ---------------------------------------------------------------------------


def test_promote_by_full_name_supersedes_a_different_algorithm_in_the_slot():
    # Selecting one exact model must still leave exactly one production doc for
    # temp_prediction_1h_*, so a RandomForest promote supersedes a GBT.
    db = _db(
        make_entry(
            "temp_prediction_1h_GradientBoostedTrees",
            doc_id="01",
            stage="production",
            timestamp=dt(1),
        ),
        make_entry(
            "temp_prediction_1h_RandomForest",
            doc_id="02",
            stage="staging",
            timestamp=dt(3),
        ),
        make_entry(
            "rain_prediction_1h_GradientBoostedTrees",
            doc_id="03",
            stage="production",
            target="rain",
            timestamp=dt(1),
        ),
    )

    pm.promote(
        db, prefix="temp_prediction_1h_RandomForest", actor="alice", now=NOW
    )

    assert _stages(db) == {"01": "superseded", "02": "production", "03": "production"}


def test_slot_prefix_derives_the_family_and_falls_back():
    assert pm.slot_prefix({"target": "temperature", "horizon_hours": 12}) == (
        "temp_prediction_12h_"
    )
    assert pm.slot_prefix({"target": "rain", "horizon_hours": 1}) == "rain_prediction_1h_"
    # Legacy doc without target/horizon_hours -> the selector prefix is used.
    assert pm.slot_prefix({}, fallback="temp_prediction_") == "temp_prediction_"
    assert pm.slot_prefix({}, fallback=None) is None


def test_rollback_by_id_restores_only_its_own_slot():
    db = _db(
        make_entry(
            "temp_prediction_1h_Prev",
            doc_id="02",
            stage="superseded",
            superseded_at=dt(8),
        ),
        make_entry("temp_prediction_1h_Current", doc_id="03", stage="production", timestamp=dt(9)),
        make_entry(
            "temp_prediction_3h_GBT", doc_id="04", stage="production", horizon=3, timestamp=dt(9)
        ),
    )

    pm.rollback(db, prefix=None, doc_id="02", actor="bob", now=NOW)

    assert _stages(db) == {
        "02": "production",
        "03": "superseded",
        "04": "production",  # other horizon untouched
    }
