# Batch 3 — frozen interfaces

Single source of truth for the four parallel Batch 3 workstreams. The lead owns this
file; specialists do **not** edit it. If the contract must change, stop and report to
the lead.

Branch: `feat/batch3` (never push to `main` while working).

## Scope

U3 radar playback, U4 data colour ramps, U5 clustered GeoJSON markers + wind arrows +
weather-code icons + basemap switcher, data-quality meter (roadmap D4), honest metrics —
temporal split + persistence/climatology baselines + skill score (roadmap D1 + M1),
prediction intervals (roadmap M3), R4 CI guardrails.

## Hard rules (from `docs/handoff.md`, paid for)

1. **Never run inference before the features are rebuilt.** Final acceptance is
   `gh workflow run ml-pipeline.yml --ref feat/batch3 -f job=all` (features → train →
   prune → inference). No `job=inference` run on its own. The saved model carries the
   feature list it was trained with.
2. **Every Mongo sort must be city-prefixed** (`{city:1, timestamp:-1}`); aggregates pass
   `allowDiskUse=True`; never add a global `{timestamp:-1}` index; Atlas M0 rejects
   external sorting. Code 292 = unindexed blocking sort.
3. Do not rename collections; do not split `weather_data`/`raw_weather`; do not change
   the prediction key `(city, source_timestamp, horizon_hours)`. All new prediction
   fields are **additive**.
4. Charts read oldest → newest; the API reverses descending queries.
5. **One writer per path.** Do not run `git commit`/`git push`; leave changes in the
   working tree. The lead integrates. Do not edit files outside your owned path.

## Ownership

| Workstream | Owner | Owned paths |
| --- | --- | --- |
| WS-A frontend | frontend-engineer | `frontend/**` |
| WS-B backend | backend-engineer | `backend/**` |
| WS-C ML | ml-engineer | `spark/spark-jobs/{ml_training.py,inference.py}`, `spark/config/spark_config.py`, `tests/unit/test_inference.py`, `tests/unit/test_ml_training.py` |
| WS-D CI | ci-engineer | `.github/**`, `scripts/prune_models.py`, `tests/integration/test_mongo_index_sort.py`, `tests/unit/test_inference_output_schema.py`, `tests/unit/test_storage_budget.py`, `tests/conftest.py` |

---

## Contract 1 — Data-quality meter (WS-B produces, WS-A consumes)

`GET /api/weather/quality?days=7` — `days` integer 1..30, default 7. Cached with the
existing `cached()` helper (TTL 300 s) so it gets an ETag/`Cache-Control` like the other
read routes.

```json
{
  "generated_at": "2026-09-25T12:00:00Z",
  "days": 7,
  "cities": [
    {
      "city": "Madrid",
      "expected_hours": 168,
      "observed_hours": 165,
      "completeness": 0.982,
      "max_gap_hours": 3.0,
      "null_rate": 0.012,
      "last_observed_at": "2026-09-25T11:00:00Z",
      "age_hours": 1.0,
      "status": "ok"
    }
  ]
}
```

- Source collection: **`weather_features`** (the model's actual input), window
  `[now - days, now]`, grouped per city. `expected_hours = days * 24`.
- `completeness = observed_hours / expected_hours` (clamped to 1.0).
- `max_gap_hours`: largest gap between consecutive observed hours in the window (0 if
  fewer than two points).
- `null_rate`: fraction of observed rows whose `temperature` is null.
- `age_hours = (now - last_observed_at)` in hours.
- `status` per metric, then the worst wins:
  completeness/age: `ok` = completeness ≥ 0.95 and age ≤ 2 h; `warn` = ≥ 0.80 and
  ≤ 6 h; else `bad`. If `last_observed_at` is null → `bad`.
- Read rows sorted **city-prefixed**; compute gaps/aggregates in Python (bounded:
  ≤ 30×24 rows/city). Any aggregate must pass `allowDiskUse=True`.
- Frontend zod: `weatherQualitySchema` / `WeatherQuality` (Contract 4).

## Contract 2 — Prediction intervals (WS-C writes, WS-B passes through, WS-A renders)

`weather_predictions` gains **additive** fields per document:

| Field | Type | Meaning |
| --- | --- | --- |
| `temp_lower` | float \| null | `predicted_temperature + lower_offset` |
| `temp_upper` | float \| null | `predicted_temperature + upper_offset` |
| `interval_level` | float \| null | nominal level, fixed 0.8 |

- The `_id` and `(city, source_timestamp, horizon_hours)` key are unchanged.
- `temp_lower`/`temp_upper` are null when the model has no interval (legacy doc).

API `Prediction` schema gains the same three nullable fields (default `null`);
`prediction_repo.prediction_from_doc` maps them.

## Contract 3 — Honest metrics + registry (WS-C writes, WS-B exposes, WS-A renders)

Temporal split replaces `randomSplit`. `ML_CONFIG["data_split"]` gains
`"kind": "temporal"`; ratios and `seed` stay. Fallback to `randomSplit` when the frame
has < 3 distinct `timestamp` values.

`model_registry` document gains **additive** blocks:

```jsonc
{
  "metrics": {
    // existing: rmse, mae, r2 (temperature) | auc_roc, auc_pr (rain) | n_test
    "persistence_rmse": 3.05,      // temperature: prediction = current observed temp
    "climatology_rmse": 3.41,      // temperature: train-window mean per city
    "skill_score": 0.53,           // 1 - rmse / persistence_rmse, temperature
    "brier": 0.11,                 // rain: Brier score of the model
    "persistence_brier": 0.14,     // rain: Brier of "will it rain = now raining"
    "prevalence": 0.18,            // rain: positive rate on test
    "coverage": 0.79               // fraction of test residuals within the interval
  },
  "split": { "kind": "temporal", "train_end": "ISO", "val_end": "ISO", "test_start": "ISO" },
  "interval": { "level": 0.8, "lower_offset": -1.9, "upper_offset": 2.1 },
  "data_snapshot": { "rows": 12000, "from": "ISO", "to": "ISO", "cities": 15,
                     "features": ["temperature", "humidity", "..."] },
  "commit": "GITHUB_SHA or \"unknown\""
}
```

- For rain, `skill_score = 1 - brier / persistence_brier`.
- Interval offsets: p10/p90 of `(target - prediction)` on the temporal **test** split;
  `coverage` = fraction of test residuals in `[lower_offset, upper_offset]`.
- All values pass through `_json_safe`; no NumPy/Java scalars in BSON.
- Existing `feature_importance`, `params`, `stage`, `schema_version` are unchanged.

API schemas:
- `ModelMetrics` gains `persistence_rmse`, `climatology_rmse`, `skill_score`, `brier`,
  `persistence_brier`, `prevalence`, `coverage` (all `float | None = None`).
- `ModelInfo` gains `split: dict | None`, `interval: dict | None`, `commit: str | None`.

`model_repo.list_models` sort becomes `[("model_name", 1), ("timestamp", -1)]`
(accepted follow-up; city-prefix rule does not apply — no city field).

## Contract 4 — Frontend schema/type additions (WS-A)

Exact names, in `frontend/src/api/schemas.ts`:

```ts
export const weatherQualitySchema = z.object({
  generated_at: isoDate,
  days: z.number(),
  cities: z.array(z.object({
    city: z.string(),
    expected_hours: z.number(),
    observed_hours: z.number(),
    completeness: z.number(),
    max_gap_hours: nullableNumber,
    null_rate: nullableNumber,
    last_observed_at: nullableString,
    age_hours: nullableNumber,
    status: z.enum(['ok', 'warn', 'bad']),
  })),
})
export type WeatherQuality = z.infer<typeof weatherQualitySchema>
```

`predictionSchema` gains `temp_lower`, `temp_upper`, `interval_level` (all
`nullableNumber` / `nullableNumber` for level).
`registryModelSchema.metrics` gains the seven Contract-3 metric keys (partial/nullable);
`registryModelSchema` gains `split`, `interval`, `commit` via
`.nullish().transform(v => v ?? null)`.

Query key: add `quality: ['weather','quality']` to `queryKeys`.

## R4 — CI guardrails (WS-D)

1. New CI job with a real `mongo:6` service container running
   `tests/integration/test_mongo_index_sort.py`. It must **not** use mongomock. Assert
   that the repo's city-prefixed sorts are index-backed (`explain()` plan has no
   `SORT` stage) and that a `{timestamp:-1}`-only sort is a blocking sort.
2. `tests/unit/test_inference_output_schema.py` imports `inference.output_schema_columns()`
   (WS-C adds this pure helper) and asserts the exact write-column contract incl. `_id`
   composition, `temp_lower`, `temp_upper`, `interval_level`.
3. `tests/unit/test_storage_budget.py`: steady-state estimate from the TTLs
   (`raw_weather` 180 d, `weather_predictions` 90 d, `weather_data` 30 d) × the measured
   growth rate must stay < 512 MB.
4. `ml-pipeline.yml`: `prune_models` step runs `if: always()` inside the train stage so a
   failed train still prunes.

## Verification (lead, after integration)

- `cd backend && ../.venv/bin/python -m pytest tests -q`
- `.venv/bin/python -m pytest tests -q`
- `cd frontend && npm run test && npm run typecheck && npm run lint && npm run build`
- Gates: `code-reviewer`, `qa-engineer`, `database-engineer` (quality aggregate + index
  test), `sre-engineer` (storage budget).
- Then `gh workflow run ml-pipeline.yml --ref feat/batch3 -f job=all` and verify the
  registry carries the new metrics and `weather_predictions` carries intervals.
