"""Human promotion gate for the model registry (M5).

Every training run writes ``stage="staging"``; nothing is scored as production
until a human promotes it. This script is that gate:

    # promote the newest staging temperature model for horizon 1
    python scripts/promote_model.py --model-name temp_prediction_ --horizon 1

    # promote one exact document, without writing (review first)
    python scripts/promote_model.py --id 6512ab... --dry-run

    # undo the last promote
    python scripts/promote_model.py --model-name temp_prediction_1h_ --rollback

Semantics (frozen in Batch 5):

- A promote sets ``stage="production"``, ``promoted_at`` (UTC) and
  ``promoted_by``. Any *other* production document in the same slot (same
  ``model_name`` prefix + ``horizon_hours``) becomes ``stage="superseded"`` with
  ``superseded_at``/``superseded_by``. Nothing is ever deleted and
  ``weather_predictions`` is never touched.
- A rollback restores the most recently superseded document in the selector back
  to production (clearing ``superseded_at``/``superseded_by``) and marks the
  current production document in that slot superseded.
- Re-running the same command is a no-op: an already-production target keeps its
  original ``promoted_at``, and superseded documents are never re-staged.  When
  selecting by prefix, an incumbent whose ``timestamp`` is >= the newest staging
  candidate (including "only production exists") returns ``changed=False`` and
  exits 0 with an explicit "already production, nothing to do" message.
- Exit code is non-zero with a clear message only when *neither* a staging
  candidate nor a production incumbent matches, when the target has no
  ``gridfs_file_id`` / its GridFS artifact is missing (an unloadable model must
  never become production) and when ``--rollback`` finds no superseded candidate.
- Slot peers are demoted *before* the target is promoted, so a crash between the
  two writes leaves zero production (inference falls back to staging) rather
  than two production documents.

The selector prefix comes from :mod:`model_selection` — the same module
``inference.py`` uses — so "what production means" has exactly one definition.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from datetime import datetime, timezone

# ``model_selection`` lives with the Spark jobs; it is pure stdlib, so importing
# it here cannot drag pyspark/pymongo into the script.
SPARK_JOBS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "spark", "spark-jobs"
)
if SPARK_JOBS_DIR not in sys.path:
    sys.path.insert(0, SPARK_JOBS_DIR)

import model_selection as ms  # noqa: E402
from gridfs import GridFS  # noqa: E402
from pymongo import MongoClient  # noqa: E402

DEFAULT_DB = "weather_db"

#: A horizon already present in the prefix ("temp_prediction_1h_", "..._12h_GBT").
_HORIZON_IN_NAME = re.compile(r"\d+h(?:_|$)")

#: Rollback candidate ordering: newest supersede first, then newest training run.
_ROLLBACK_SORT = [("superseded_at", -1), ("timestamp", -1)]


class PromotionError(RuntimeError):
    """A refuse-to-act condition; ``main`` prints it and exits non-zero."""


# ---------------------------------------------------------------------------
# Actor / selector / clock helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp for ``promoted_at``/``superseded_at``."""
    return datetime.now(timezone.utc)


def resolve_actor() -> str:
    """Who is promoting: ``$GITHUB_ACTOR``, else the local user, else ``"unknown"``.

    The CI actor is preferred over ``getpass.getuser()`` because on GitHub
    Actions the OS user is the uninformative ``runner`` while the actor is the
    auditable identity.
    """
    actor = os.getenv("GITHUB_ACTOR")
    if actor:
        return actor
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - a missing passwd entry must not fail a promote
        return "unknown"


def resolve_prefix(model_name: str, horizon: int | None = None) -> str:
    """Turn ``--model-name`` (+ optional ``--horizon``) into a match prefix.

    ``--horizon`` only narrows when the prefix does not already name a horizon:

    - ``temp_prediction_`` + ``--horizon 1`` -> ``temp_prediction_1h_``
    - ``temp_prediction_1h_`` + ``--horizon 3`` -> ``temp_prediction_1h_`` (the
      horizon in the name wins; the flag cannot contradict it)
    - ``temp_prediction_1h_GradientBoostedTrees`` -> used verbatim (a full name
      is also a prefix, and matches exactly one document)
    """
    if horizon is None or _HORIZON_IN_NAME.search(model_name):
        return model_name
    separator = "" if model_name.endswith("_") else "_"
    return f"{model_name}{separator}{horizon}h_"


def _promotion_fields(actor: str, now: datetime) -> dict:
    return {"stage": ms.STAGE_PRODUCTION, "promoted_at": now, "promoted_by": actor}


def _supersede_fields(actor: str, now: datetime, via: str = "promote") -> dict:
    """Stage-change fields for a superseded document.

    ``superseded_via`` records *why* a document left production. A rollback only
    ever restores documents a promote displaced, so re-running a rollback cannot
    oscillate production between two documents (a retried CI job must not flip
    the stage back).
    """
    return {
        "stage": ms.STAGE_SUPERSEDED,
        "superseded_at": now,
        "superseded_by": actor,
        "superseded_via": via,
    }


# ---------------------------------------------------------------------------
# Slot queries
# ---------------------------------------------------------------------------


#: Registry ``target`` -> the family segment of ``model_name``.
_FAMILY_BY_TARGET = {"temperature": "temp", "rain": "rain"}


def slot_prefix(doc: dict, fallback: str | None = None) -> str | None:
    """The family prefix a document occupies, e.g. ``temp_prediction_1h_``.

    A promote must leave exactly ONE production document per slot, so peers are
    matched on the slot the target occupies — not on how the operator spelled the
    selector. Promoting by full name (``temp_prediction_1h_RandomForest``) or by
    ``--id`` therefore still supersedes a *different algorithm* that is currently
    in production for the same horizon. Falls back to the selector prefix when a
    legacy document carries no ``target``/``horizon_hours``.
    """
    kind = _FAMILY_BY_TARGET.get(doc.get("target"))
    horizon = doc.get("horizon_hours")
    if kind is None or horizon is None:
        return fallback
    return f"{kind}_prediction_{horizon}h_"


def _production_in_slot(
    coll,
    target: dict,
    prefix: str | None,
) -> list[dict]:
    """Production documents that share *target*'s slot and must be superseded.

    The slot is ``(family model_name prefix, horizon_hours, target)`` — exactly
    the set a single ``temp_prediction_1h_*`` model can serve. Ordering is
    ``[("timestamp", -1)]`` so the printed table is deterministic.
    """
    horizon = target.get("horizon_hours")
    query: dict = {
        "_id": {"$ne": target["_id"]},
        "stage": ms.STAGE_PRODUCTION,
    }
    if horizon is None:
        # A legacy target with no horizon must not match peers that *do* carry
        # one: ``horizon_hours: None`` in Mongo also matches nothing useful, and
        # it silently let a horizon-less promote demote other horizons.
        query["horizon_hours"] = {"$exists": False}
    else:
        query["horizon_hours"] = horizon
    slot = slot_prefix(target, prefix)
    if slot:
        query["model_name"] = {"$regex": f"^{re.escape(slot)}"}
    if target.get("target"):
        query["target"] = target["target"]
    return list(coll.find(query, sort=list(ms.SORT)))


def _rollback_candidates(prefix: str) -> dict:
    """Superseded documents a rollback may restore.

    Excludes docs a *rollback* superseded (``superseded_via="rollback"``), so
    repeated rollbacks walk back through promote history instead of toggling the
    last two models forever. Documents written before this field existed have no
    ``superseded_via`` and stay eligible.
    """
    query = ms.prefix_query(prefix, ms.STAGE_SUPERSEDED)
    query["superseded_via"] = {"$ne": "rollback"}
    return query


def _require_loadable(doc: dict, fs=None) -> None:
    """Refuse a document whose artifact cannot be loaded from GridFS.

    A set ``gridfs_file_id`` is not enough: a failed ``fs.delete`` during prune
    (or a manual cleanup) leaves a dead id behind, and promoting it makes the
    next hourly inference abort on ``NoFile``.  When a GridFS handle is supplied
    (``main`` always does) the artifact is checked with ``fs.exists``.
    """
    file_id = doc.get("gridfs_file_id")
    if not file_id:
        raise PromotionError(
            f"model_registry doc {doc.get('_id')} ('{doc.get('model_name')}') has no "
            "gridfs_file_id; refusing to put an unloadable model in production."
        )
    if fs is not None and not fs.exists(file_id):
        raise PromotionError(
            f"model_registry doc {doc.get('_id')} ('{doc.get('model_name')}') points at "
            f"GridFS file {file_id} that does not exist; refusing to promote a dead artifact."
        )


# ---------------------------------------------------------------------------
# Promote / rollback
# ---------------------------------------------------------------------------


def _record(doc: dict, before: str | None, after: str) -> dict:
    return {
        "model_name": doc.get("model_name"),
        "version": doc.get("version"),
        "stage_before": before,
        "stage_after": after,
    }


def promote(
    db,
    *,
    prefix: str | None = None,
    doc_id=None,
    actor: str | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    fs=None,
) -> dict:
    """Promote one registry document to production and supersede its slot peers.

    Returns ``{"rows": [...], "target": doc, "changed": bool, "dry_run": bool}``;
    raises :class:`PromotionError` when nothing matches or the target is
    unloadable.
    """
    coll = ms.registry_collection(db)
    actor = actor or resolve_actor()
    now = now or utcnow()

    if doc_id is not None:
        target = coll.find_one({"_id": doc_id})
        if target is None:
            raise PromotionError(f"No model_registry document with _id={doc_id}.")
        _require_loadable(target, fs)
    else:
        if not prefix:
            raise PromotionError("A --model-name prefix or an --id is required.")
        # Prefix selection is newest-first.  If the incumbent production model is
        # already at least as new as the newest staging candidate, a retried
        # command must be a clean no-op instead of reshuffling stages (QA repro:
        # promote v3, retry, v2 gets promoted and v3 superseded).
        incumbent = ms.production_entry(db, prefix)
        candidate = ms.newest_matching(db, prefix, ms.STAGE_STAGING)
        if candidate is None and incumbent is None:
            raise PromotionError(
                f"No staging model matches prefix '{prefix}'. "
                "Train one first (spark/spark-jobs/ml_training.py) or pass --id."
            )
        incumbent_ts = incumbent.get("timestamp") if incumbent is not None else None
        candidate_ts = candidate.get("timestamp") if candidate is not None else None
        if candidate is None or (
            incumbent_ts is not None and candidate_ts is not None and incumbent_ts >= candidate_ts
        ):
            # Nothing newer to promote.  ``incumbent`` is production already;
            # ``candidate`` is only reached here when it is older than it.
            target = incumbent if incumbent is not None else candidate
            return {
                "rows": [_record(target, target.get("stage"), target.get("stage"))],
                "target": target,
                "changed": False,
                "dry_run": dry_run,
                "action": "promote",
                "actor": actor,
                "noop": "already production",
            }
        target = candidate
        _require_loadable(target, fs)

    peers = _production_in_slot(coll, target, prefix)
    rows = []
    if target.get("stage") != ms.STAGE_PRODUCTION:
        rows.append(_record(target, target.get("stage"), ms.STAGE_PRODUCTION))
    else:
        # Idempotent re-run: keep the original promoted_at/promoted_by.
        rows.append(_record(target, ms.STAGE_PRODUCTION, ms.STAGE_PRODUCTION))
    rows.extend(_record(peer, peer.get("stage"), ms.STAGE_SUPERSEDED) for peer in peers)

    changed = target.get("stage") != ms.STAGE_PRODUCTION or bool(peers)
    if not dry_run:
        # Demote peers BEFORE promoting the target: a crash in between leaves
        # zero production (staging fallback) instead of two production docs.
        for peer in peers:
            coll.update_one({"_id": peer["_id"]}, {"$set": _supersede_fields(actor, now)})
        if target.get("stage") != ms.STAGE_PRODUCTION:
            coll.update_one({"_id": target["_id"]}, {"$set": _promotion_fields(actor, now)})

    return {
        "rows": rows,
        "target": target,
        "changed": changed,
        "dry_run": dry_run,
        "action": "promote",
        "actor": actor,
    }


def rollback(
    db,
    *,
    prefix: str | None = None,
    doc_id=None,
    actor: str | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    fs=None,
) -> dict:
    """Restore the most recently superseded document in the slot to production.

    Raises :class:`PromotionError` when there is no candidate to restore or the
    candidate is unloadable.
    """
    coll = ms.registry_collection(db)
    actor = actor or resolve_actor()
    now = now or utcnow()

    if doc_id is not None:
        target = coll.find_one({"_id": doc_id})
        if target is None:
            raise PromotionError(f"No model_registry document with _id={doc_id}.")
        if target.get("stage") != ms.STAGE_SUPERSEDED:
            raise PromotionError(
                f"model_registry doc {doc_id} is stage='{target.get('stage')}', not "
                "'superseded'; nothing to roll back to."
            )
    else:
        if not prefix:
            raise PromotionError("A --model-name prefix or an --id is required.")
        target = coll.find_one(
            _rollback_candidates(prefix),
            sort=list(_ROLLBACK_SORT),
        )
        if target is None:
            raise PromotionError(
                f"No superseded model matches prefix '{prefix}'; nothing to roll back to."
            )

    _require_loadable(target, fs)

    peers = _production_in_slot(coll, target, prefix)
    rows = [_record(target, target.get("stage"), ms.STAGE_PRODUCTION)]
    rows.extend(_record(peer, peer.get("stage"), ms.STAGE_SUPERSEDED) for peer in peers)

    if not dry_run:
        # Same ordering rule as promote: never leave two production docs behind a
        # crash partway through the stage flip.
        for peer in peers:
            coll.update_one(
                {"_id": peer["_id"]},
                {"$set": _supersede_fields(actor, now, via="rollback")},
            )
        coll.update_one(
            {"_id": target["_id"]},
            {
                "$set": _promotion_fields(actor, now),
                # A restored model is not superseded any more; dropping the stale
                # audit fields keeps "one stage per doc" honest.
                "$unset": {"superseded_at": "", "superseded_by": ""},
            },
        )

    return {
        "rows": rows,
        "target": target,
        "changed": True,
        "dry_run": dry_run,
        "action": "rollback",
        "actor": actor,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_table(rows: list[dict]) -> str:
    """Compact one-line-per-model before/after table."""
    if not rows:
        return "(no changes)"
    headers = ("model_name", "version", "stage_before", "stage_after")
    cells = [
        (
            str(row.get("model_name") or "unknown"),
            str(row.get("version") or "unknown"),
            str(row.get("stage_before") or "none"),
            str(row.get("stage_after") or "none"),
        )
        for row in rows
    ]
    widths = [
        max(len(header), *(len(cell[i]) for cell in cells)) for i, header in enumerate(headers)
    ]
    lines = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in cells)
    return "\n".join(lines)


def _parse_object_id(raw: str | None):
    """CLI ``--id`` -> ``ObjectId`` (``None`` when absent); clear error otherwise."""
    if not raw:
        return None
    from bson import ObjectId

    try:
        return ObjectId(raw)
    except Exception as exc:  # noqa: BLE001 - invalid ids are user input, not faults
        raise PromotionError(f"--id '{raw}' is not a valid ObjectId: {exc}")


def _mongo_uri(cli_value: str | None) -> str:
    uri = cli_value or os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
    if not uri:
        raise PromotionError("MONGO_URI is not set (or pass --mongo-uri).")
    return uri


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote (or roll back) a model_registry document in the M5 lifecycle."
    )
    parser.add_argument(
        "--model-name",
        help="Model name prefix or full name, e.g. temp_prediction_1h_. "
        "Required unless --id is given.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="Narrow to {prefix}{horizon}h_ when the prefix names no horizon.",
    )
    parser.add_argument("--id", dest="doc_id", help="Target one model_registry ObjectId.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, write nothing.")
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Restore the most recently superseded model in the selector.",
    )
    parser.add_argument("--mongo-uri", dest="mongo_uri", help="Defaults to $MONGO_URI/$MONGO_URL.")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Mongo database (default {DEFAULT_DB}).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.model_name and not args.doc_id:
        parser.error("--model-name is required unless --id is given.")

    try:
        doc_id = _parse_object_id(args.doc_id)
        uri = _mongo_uri(args.mongo_uri)
        action = rollback if args.rollback else promote

        client = MongoClient(uri)
        try:
            db = client[args.db]
            # With only --id the slot is derived from the document itself (no
            # name filter), which is why the prefix stays None here.
            prefix = resolve_prefix(args.model_name, args.horizon) if args.model_name else None
            result = action(
                db,
                prefix=prefix,
                doc_id=doc_id,
                dry_run=args.dry_run,
                fs=GridFS(db),
            )
        finally:
            client.close()

    except PromotionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    verb = result["action"] + (" (dry run)" if result["dry_run"] else "")
    if result.get("noop"):
        target = result.get("target") or {}
        print(
            f"{verb}: '{target.get('model_name')}' (version {target.get('version')}) "
            f"is already production, nothing to do by {result['actor']}"
        )
    else:
        print(
            f"{verb} by {result['actor']}" + ("" if result["changed"] else " — nothing to change")
        )
    print(format_table(result["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
