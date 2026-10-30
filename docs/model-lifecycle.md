# Model lifecycle — canonical store, staging, promotion, shadow

Read this after `docs/handoff.md`. It declares **where a model lives**, how it
moves from `staging` to `production`, and what the CI MLflow store is *not*.
Nothing here duplicates the roadmap; the roadmap holds the plan and status.

## The canonical store

Two collections in `weather_db` are the system of record. Everything else —
MLflow, a local zip, a `$GITHUB_STEP_SUMMARY` line — is derived or ephemeral.

| Piece | Where | Written by | Read by | Notes |
| --- | --- | --- | --- | --- |
| Artifact (zipped Spark `PipelineModel`) | GridFS: `fs.files` / `fs.chunks`, addressed by `gridfs_file_id` | `spark/spark-jobs/ml_training.py:save_model` | `spark/spark-jobs/inference.py:load_latest_model` | Filename `{temp,rain}_prediction_{h}h_{Algorithm}_{YYYYmmdd_HHMMSS}.zip`. The `_id` is pre-generated so an upload retry cannot duplicate it. |
| Metadata (lineage + metrics) | `model_registry` (one doc per trained model) | `ml_training.save_model` | `inference.py`, `scripts/pipeline_summary.py`, backend `/api/models` | `{model_name: 1, timestamp: -1}` index. **The only** source of interval and lineage. |

`model_registry` fields a consumer may rely on:

| Field | Meaning |
| --- | --- |
| `model_name` | `temp_prediction_{h}h_{Algorithm}` / `rain_prediction_{h}h_{Algorithm}`. Lookups are a prefix regex ordered by `timestamp` desc. |
| `gridfs_file_id` | The GridFS artifact. Resolving this is what makes the doc loadable. |
| `model_type`, `target`, `horizon_hours` | Algorithm name, `temperature` / `rain`, 1/3/6/12/24. |
| `version` | `YYYYmmdd_HHMMSS` of the run. |
| `stage` | `staging` (written by training) → `production` (human promote) → `superseded` (previous production model). |
| `metrics` | Honest test-split metrics: `rmse`/`mae`/`r2`/`persistence_rmse`/`climatology_rmse`/`skill_score`/`coverage`/`n_test` for temperature, `auc_roc`/`auc_pr`/`brier`/`persistence_brier`/`prevalence`/`skill_score`/`n_test` for rain. |
| `interval` | Prediction interval block: `level` (0.8), `lower_offset`, `upper_offset`. Temperature models only. Sums to the published interval columns; a missing `interval` means "no honest interval". |
| `diagnostics` | `by_city`, `by_hour_of_day`, `by_rain_bucket`, `drift_psi` (train→test proxy). Rendered as the ModelsPage drift badge. |
| `split`, `data_snapshot`, `params`, `feature_importance`, `commit`, `schema_version` | Reproducibility: temporal split boundaries, training window/rows/cities/feature list, tuned grid params, top features, `GITHUB_SHA`. |

There is no `model_registry` TTL: rows are deleted only by
`scripts/prune_models.py`, together with their GridFS file.

## Why the CI MLflow file store is not a registry

`ml-pipeline.yml` used to export `MLFLOW_TRACKING_URI=file://$GITHUB_WORKSPACE/mlruns`
so `mlflow.start_run()` would not hang against a non-existent server. That
directory lives on the runner and is destroyed with it, so:

- every run started from an **empty** store — experiment history, metrics and
  `registered_model_name` entries vanished at job end;
- "the model is in MLflow" was never a durable statement, yet `inference.py`
  briefly resolved *metadata* from it and published rows with a fake
  `MLflow-latest` version and no `interval` (Batch 3 trap, do not regress);
- a file store has no `models:/…/Production` stage service, so nothing could be
  promoted or rolled back through it.

Consequences in this repo:

- `ml-pipeline.yml` **no longer sets `MLFLOW_TRACKING_URI`** and **no longer
  installs `mlflow`**. GridFS + `model_registry` are the only store CI uses.
- `ml_training._log_to_mlflow` is **opt-in and lazy**: it returns immediately
  (logging at info) when `MLFLOW_TRACKING_URI` is unset, imports `mlflow` inside
  the function (the module imports and trains with `mlflow` uninstalled), and
  turns any MLflow failure into a **warning** — the GridFS upload and registry
  insert have already succeeded at that point, so MLflow can never fail a run.
- `docker-compose.yml` still runs a local `mlflow` service for experimentation
  and still sets `MLFLOW_TRACKING_URI=http://mlflow:5000` on the scheduler. That
  is a *local* opt-in; it is not part of the deployed pipeline.
- With the variable unset, `inference.py` always loads the artifact from GridFS.
  If someone sets it, MLflow may serve the *artifact*, but metadata still comes
  from the newest `model_registry` doc, never from MLflow.

## Staging → production

```
train (nightly 35 2 * * *)          human                       any later promote
  save_model ------------------>  promote_model.py  ---------->  superseded
  stage="staging"                 stage="production"
  metrics/interval/diagnostics    (per model_name prefix)
```

Rules:

1. Training always writes `stage="staging"`; it never promotes. A bad night
   therefore cannot reach the API by itself.
2. Exactly **one** `production` document per `model_name` prefix
   (`temp_prediction_{h}h` / `rain_prediction_{h}h`, algorithm-agnostic). A
   promote demotes the incumbent to `superseded` in the same operation, so the
   pointer never dangles.
3. Promotion is a human decision backed by the registry, not a metric gate: read
   `metrics.skill_score` (must beat persistence; `<= 0` is the degradation
   warning in `scripts/pipeline_summary.py`), `metrics.n_test`,
   `diagnostics.drift_psi` and `interval.coverage` before promoting.
4. Rollback is a promote of an older `superseded` document — stages are fields,
   history is never destroyed.
5. **No-op rule (prefix selection):** when no `--id` is given, the script loads
   the incumbent production document in the slot first. If it exists and its
   `timestamp` is `>=` the newest staging candidate's, or only production exists,
   it prints `already production, nothing to do` and exits **0** without writing.
   A retried promote therefore cannot demote a newer production model back to an
   older staging one. Only "neither production nor staging matches" is an error.
   `--id` deliberately pins the given document (even an older one), and
   `--rollback` is unchanged.
6. A target is refused (non-zero, no writes) when its `gridfs_file_id` is
   missing **or the GridFS artifact does not exist** (`fs.exists`): a dead id
   from a failed delete must never become production, because the next hourly
   inference would abort on `NoFile`.
7. Slot peers are demoted **before** the target is promoted, so a crash between
   the two writes leaves zero production (inference falls back to staging) rather
   than two production documents.

`scripts/promote_model.py` is a single command (there is no `promote`/`rollback`
subcommand and no `--version` flag); `--rollback` selects the rollback action:

```bash
# Promote the newest staging model for horizon 1 (prefix + --horizon).
python scripts/promote_model.py --model-name temp_prediction_ --horizon 1

# Review the plan first: --dry-run prints from_stage -> to_stage, writes nothing.
python scripts/promote_model.py --model-name temp_prediction_ --horizon 1 --dry-run

# Promote one exact registry document (may deliberately be older).
python scripts/promote_model.py --id 6512ab... --dry-run
python scripts/promote_model.py --id 6512ab...

# Rolling back the last promote (newest superseded doc in the selector).
python scripts/promote_model.py --model-name temp_prediction_1h_ --rollback --dry-run
python scripts/promote_model.py --model-name temp_prediction_1h_ --rollback

# Repeat per family: temp_prediction_{1,3,6,12,24}h_, rain_prediction_{1,3,6,12,24}h_.
```

Contract every command must keep: `--dry-run` prints the intended
`from_stage -> to_stage` transitions and writes nothing; a real run is an
idempotent transition that leaves exactly one `production` doc per prefix;
without `--id` the newest eligible document is selected and a re-run with
nothing newer is a clean no-op; `MONGO_URI` is the only required environment
variable (or `--mongo-uri`).

## Shadow A/B

A shadow model is a second document scored **alongside** production, never
served by the API:

| Piece | Where |
| --- | --- |
| Shadow comparison | `model_shadow` — exactly **one document per (production, shadow) registry pair** (`model_registry` `_id`s/`version`s), updated in place, not appended per hour |
| Candidate being shadowed | the newest `staging` `model_registry` doc **strictly newer** than production (`shadow_entry` filters `timestamp > production.timestamp`); an older staging doc is never a challenger |
| Served predictions | `weather_predictions` — production only. A shadow's numbers never reach the API read path. |

Semantics:

- The metric is **disagreement, not accuracy**: for temperature it is the mean
  and max absolute delta in °C; for rain it is the mean probability delta plus a
  class-flip rate at the **0.5 probability threshold** (`>= 0.5` = rain). It is
  measured on the same cached `(city, source_timestamp)` rows, and rows where
  either model is null are excluded. Accuracy still comes from the shadow
  model's own `model_registry.metrics` on the held-out temporal test split — a
  shadow run cannot measure skill against the future, only divergence from the
  incumbent.
- Exactly **one document per (production `_id`, shadow `_id`) pair**, keyed by
  `_id = "{production_id}_{shadow_id}"` and `replace_one(..., upsert=True)` each
  run, so an hourly re-run updates in place. A *new* pair (a new promote or a
  newly trained challenger) mints a **new** document; old pair docs linger until
  `prune_models.py` removes them — either because `updated_at` is older than
  `KEEP_SHADOW_DAYS`, or immediately because one of the referenced model ids no
  longer exists in `model_registry`.
- Disagreement is a *review signal*: high disagreement with better test metrics
  is the interesting case (the incumbent may have drifted); disagreement alone
  never promotes anything.

## Hourly fallback rule

`inference.py` runs hourly (`5 * * * *`) and needs a model for every horizon.
Resolution order per model family:

1. the `stage="production"` document, if one exists;
2. otherwise the **newest `staging`** document, with a **warning** logged
   (`no production model for <prefix>; falling back to the newest staging
   model`) so a never-promoted pipeline still scores and the missing promotion
   is visible in the job log;
3. otherwise the horizon is skipped with a warning and the remaining horizons
   are still written in the same upsert.

A staging model therefore does serve traffic — that is why training never
promotes implicitly and why the warning matters.

## Storage and retention

- Each model is one zipped `PipelineModel`: **~0.5–2 MB**. The nightly retrain
  writes 10 models (5 horizons × 2 targets), i.e. roughly **1–2 MB/day**.
- `scripts/prune_models.py` is **stage-aware**: per algorithm-agnostic family it
  always keeps every artifact referenced by a `production` or `superseded`
  registry doc (production is served, superseded is rollback history), and
  applies `KEEP_MODELS` (default **2**) only to the remaining staging/unstaged
  artifacts, newest-first. It deletes an artifact's GridFS file and **only that
  artifact's** `model_registry` row. It runs after every train stage, including a
  failed or partial one, so a bad night cannot accumulate artifacts.
- It also prunes `model_shadow`: a doc whose `(production, shadow)` pair
  references a model id no longer in `model_registry` is deleted immediately,
  and a pair untouched for `KEEP_SHADOW_DAYS` (default **30**) is aged out.
- The `model_registry` index is `{model_name: 1, timestamp: -1}` — there is no
  TTL; deletes come from `prune_models.py` only. Because production and
  superseded are retained unconditionally, a rollback target survives a run
  regardless of `KEEP_MODELS`.
- GridFS counts against the Atlas M0 512 MB budget (see the DB row in
  `docs/handoff.md`); artifacts are the easiest axis to trim if headroom shrinks.
- `weather_features` retention (`WEATHER_FEATURES_RETENTION_DAYS`, 180 d) is
  unrelated to model retention but bounded by the same tier.

## Environment variables

| Variable | Required by | Default | Meaning |
| --- | --- | --- | --- |
| `MONGO_URI` | training, inference, feature job, `promote_model.py`, `prune_models.py` | — (**required**) | Atlas connection string. Never commit or print it. `MONGO_URL` is accepted as a fallback by the Spark jobs only. |
| `MONGO_DB` | backend only | `weather_db` | Database name the API reads. The Spark jobs hard-code `weather_db`. |
| `MLFLOW_TRACKING_URI` | nothing (optional) | unset = MLflow off | Opt-in MLflow logging. `file://…` is ephemeral and is **not** a registry. Unset in CI on purpose. |
| `KEEP_MODELS` | `scripts/prune_models.py` | `2` | Newest staging/unstaged artifacts kept per family; production and superseded artifacts are always kept. |
| `KEEP_SHADOW_DAYS` | `scripts/prune_models.py` | `30` | Age after which an unrefreshed `model_shadow` pair doc is deleted (orphaned pairs are deleted sooner). |
| `SPARK_TMP_DIR` | training, inference | `/opt/spark-tmp` | Scratch dir for the zipped/unzipped model; CI points it at the workspace. |
| `WEATHER_FEATURES_RETENTION_DAYS` | `batch_processing.py` | `180` | `weather_features` prune window, aligned with `raw_weather`'s TTL. |
| `GITHUB_SHA` | `ml_training.save_model` | `unknown` | Recorded as `model_registry.commit`. |
