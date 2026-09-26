# Batch 4 — frozen interfaces

Single source of truth for the Batch 4 parallel workstreams. The lead owns this file;
specialists do **not** edit it. If the contract must change, stop and report to the lead.

Branch: `feat/batch4` (never commit/push while working; the lead integrates).

## Scope

M2 horizons `[1,3,6,12,24]` as config; M4 slice diagnostics + drift PSI (badge + workflow
warning); D2 analytics endpoints; U6 chart brushing + synced x-domain, command palette,
keyboard shortcuts, mobile affordances; U7 fresh-data pulse, shimmer skeletons, 120–240 ms
transitions honouring `prefers-reduced-motion`; recalibrated quality-meter age thresholds.
**L1 (15-min ingest) stays deferred** — do not touch the ingest cadence.

Decisions taken: **D2 is backend on-demand + the existing `cached()` TTL** (no new Mongo
collection, no new Spark/CI stage). `GET /api/predictions/latest` returns **all horizons**.

## Hard rules (from `docs/handoff.md`, paid for)

1. Any feature-semantics change needs `job=all` (features → train → prune → inference)
   before inference. The saved model carries its trained feature list.
2. Every Mongo sort is city-prefixed (`{city:1, timestamp:-1}` / the matching compound
   index); aggregates pass `allowDiskUse=True`; never add a global `{timestamp:-1}` index.
   Atlas M0 rejects external sorts (code 292).
3. Do not rename collections; do not split `weather_data`/`raw_weather`; the prediction key
   `(city, source_timestamp, horizon_hours)` is unchanged. New fields are **additive**.
4. One writer per path. Do not run `git commit`/`git push`; leave changes in the working
   tree. Do not edit files outside your owned path.
5. No new frontend runtime dependency. Reuse what is installed (React, TanStack Query, zod,
   lucide-react). Command palette is hand-rolled.

## Ownership

| WS | Owner | Owned paths |
| --- | --- | --- |
| A ML | ml-engineer | `spark/config/spark_config.py`, `spark/spark-jobs/{batch_processing,ml_training,inference}.py`, `tests/unit/test_{batch_processing,ml_training,inference}.py` |
| B backend core | backend-engineer | `backend/app/repositories/{prediction_repo,model_repo,weather_repo}.py`, `backend/app/routers/{predictions,models,weather}.py`, `backend/app/schemas/{predictions,models,quality}.py`, `backend/tests/{conftest,fakes,test_routers,test_registry_repo,test_weather_repo}.py` |
| C backend analytics | backend-engineer | `backend/app/routers/analytics.py` (new), `backend/app/repositories/analytics_repo.py` (new), `backend/app/schemas/analytics.py` (new), `backend/app/main.py`, `backend/tests/test_analytics.py` (new) |
| D1 frontend core | frontend-engineer | `frontend/**` **except** `frontend/src/features/analytics/**` |
| D2 frontend analytics | frontend-engineer | `frontend/src/features/analytics/**` only (all new files) |
| E CI | ci-engineer | `.github/workflows/ml-pipeline.yml`, `scripts/pipeline_summary.py`, `tests/unit/test_storage_budget.py` |

`backend/tests/conftest.py`+`fakes.py` are WS-B only. WS-C writes self-contained tests (its
own app/client fixture overriding `get_analytics_repo`); it must not edit conftest/fakes.

---

## Contract 1 — M2: horizons as config (WS-A produces)

`spark/config/spark_config.py`:

```python
FEATURES_CONFIG = {
    "window_sizes": [6, 12, 24],
    "lag_periods": [1, 2, 3, 6, 12],
    "target_horizons": [1, 3, 6, 12, 24],   # replaces "target_horizon"
}
ML_CONFIG["long_horizon_from"] = 12          # horizons >= this use a reduced grid
```

- `create_target_variable(df, horizons: list[int])` writes `target_temp_{h}h` and
  `target_will_rain_{h}h` for every `h` in one pass.
- `save_features_to_mongodb(df, horizons)`: `dropna` subset is the observation inputs only
  (`temperature, humidity, pressure, wind_speed`). **Targets must not be in the subset** —
  the current `target_temp_1h` drop discards the newest observation, forcing a healthy
  rebuild to land ~6 h old. Keep `dropDuplicates(["city","timestamp"])`.
- `ml_training.main`: cache the feature frame once; for each horizon dropna that horizon's
  label columns, `temporal_split`, train temperature + rain, `save_model(horizon=h, ...)`.
  For `h >= long_horizon_from`, use a reduced grid (first value of each param list).
- `inference.main`: load the latest features once; for each horizon load
  `temp_prediction_{h}h` / `rain_prediction_{h}h`, predict, build that horizon's rows, then
  **union all horizons into one upsert write**. A missing model for one horizon is skipped
  with a warning, never fatal. `_id = "{city}_{epoch(source_timestamp)}_{h}h"`.
- `build_output_df(pred_df, temp_meta, rain_meta, horizon)` gains the `horizon` parameter;
  `output_schema_columns()` is unchanged (columns are identical, only more rows).
- Per-horizon `data_snapshot` + `split` reflect the rows that horizon trained on.

## Contract 2 — Prediction API (WS-B produces, WS-D1 consumes)

- `PredictionRepository.latest_per_city(horizon: int | None = None)` groups by
  `(city, horizon_hours)` taking the newest `prediction_timestamp`; sort
  `[("city",1),("horizon_hours",1),("prediction_timestamp",-1)]` (index-backed by the
  existing `{city:1,horizon_hours:1,prediction_timestamp:-1}`); outer sort
  `[("city",1),("horizon_hours",1)]`.
- `PredictionRepository.for_city(city, limit=48, horizon: int | None = None)` filters by
  `horizon_hours` when provided.
- `GET /api/predictions/latest?horizon=` (optional int 1..48): omitted → all horizons
  (count = cities × horizons), provided → one row per city. `generated_at` stays max
  `prediction_timestamp`. Response shape is unchanged.
- `GET /api/predictions/{city}?limit=&horizon=`: optional filter, unchanged otherwise.
- Frontend: `queryKeys.latestPredictions` becomes `(horizon:number) => ['predictions','latest',horizon]`;
  add `useLatestPredictions(horizon)`; a horizon selector (default 1) on the station page.

## Contract 3 — M4 diagnostics: slices + drift PSI (WS-A produce, WS-B expose, WS-D1 render, WS-E warn)

`model_registry` gains additive `diagnostics`, computed on the **temporal test split**:

```jsonc
"diagnostics": {
  "by_city":        [{"label":"Madrid","n":24,"rmse":1.2,"mae":1.0,"bias":-0.1,"brier":null}, ...],
  "by_hour_of_day": [{"label":"0","n":10,"rmse":1.3,"mae":1.1,"bias":0.0,"brier":null}, ...],  // 24, local `hour`
  "by_rain_bucket": [{"label":"dry","n":40,...},{"label":"light","n":10,...},
                     {"label":"moderate","n":3,...},{"label":"heavy","n":0,...}],
  "drift_psi":      {"temperature":0.08,"humidity":0.31, "...": null}
}
```

- `SliceMetric` = `{label:string, n:int, rmse:float|null, mae:float|null, bias:float|null,
  brier:float|null}`. Temperature slices fill rmse/mae/bias; rain slices fill brier. A
  slice with no rows is emitted with `n:0` and nulls.
- `by_hour_of_day` uses the stored local `hour` column (`batch_processing.create_time_features`).
  Rain buckets on observed precipitation: `dry <0.1`, `light 0.1–2`, `moderate 2–8`,
  `heavy >=8` mm.
- `drift_psi`: per numeric feature, PSI of the **test** distribution vs the **train**
  distribution (10 equal-width bins from train; eps guard on zero bins). Null when the
  feature has too few rows. This is a train→test drift proxy, documented as such.
- Backend `ModelInfo` gains `diagnostics: ModelDiagnostics | None` where
  `ModelDiagnostics = {by_city: list[SliceMetric], by_hour_of_day: list[SliceMetric],
  by_rain_bucket: list[SliceMetric], drift_psi: dict[str,float|None]}`; missing/legacy
  registry docs → `None`, never invented.
- Frontend `registryModelSchema.diagnostics` nullish → `{...}` empty default. Badge from
  `maxDriftPsi` = max finite `drift_psi` values: `<0.1` **stable**, `≤0.25` **warn**,
  `>0.25` **drift**. Show it on `ModelsPage`.
- WS-E: `scripts/pipeline_summary.py` prints a `::warning`/summary line listing any model
  whose `maxDriftPsi > 0.25` (read from `model_registry`), plus the count of scored models.

## Contract 4 — D2 analytics endpoints (WS-C produces, WS-D2 consumes)

Prefix `/api/analytics`, all read-only, all cached with the existing `cached()` helper
(TTL 300 s → ETag/`Cache-Control`), all sources `weather_features` (local `hour`, city-
prefixed sorts, `allowDiskUse=True`). `days` params are validated and bounded. Every
schema is null-tolerant; no fabricated values.

1. `GET /api/analytics/daily?city=&days=90` (days 7..180)
   `{"city":str,"days":int,"generated_at":iso,"points":[{"date":"YYYY-MM-DD","tmin":f|null,
   "tmax":f|null,"tmean":f|null,"hdd":f|null,"cdd":f|null,"anomaly":f|null,"heatwave":bool}]}`
   hdd/cdd base 18 °C; anomaly = tmean − same-city day-of-year climatology mean (null when
   too few years); heatwave = ≥3 consecutive days with `tmax ≥ 35.0` °C.
2. `GET /api/analytics/climatology?city=`
   `{"city":str,"generated_at":iso,"basis_years":f,"series":[{"day_of_year":int,"tmean":f|null,
   "tmin":f|null,"tmax":f|null,"n":int}]}` (day_of_year 1..366).
3. `GET /api/analytics/wind-rose?city=&days=90` (days 7..365)
   `{"city":str,"days":int,"generated_at":iso,"sectors":[{"sector":int,"count":int,
   "mean_speed":f|null}]}` (16 sectors, `sector=round(dir/22.5)%16`, speed km/h).
4. `GET /api/analytics/diurnal?days=90` (days 7..365)
   `{"days":int,"generated_at":iso,"cells":[{"city":str,"hour":int,"tmean":f|null,"n":int}]}`
   (all canonical cities × local hour 0..23).
5. `GET /api/analytics/correlation?days=90&var=temperature` (var ∈ temperature|humidity|pressure|wind_speed)
   `{"days":int,"var":str,"generated_at":iso,"cities":[str],"matrix":[[f|null]]}` (daily means
   aligned by date; Pearson pairwise, null when <3 shared days).
6. `GET /api/analytics/error-by-hour?days=30` (days 7..90)
   `{"days":int,"generated_at":iso,"points":[{"horizon_hours":int,"hour":int,"n":int,
   "mae":f|null,"rmse":f|null,"bias":f|null,"persistence_mae":f|null}]}`
   joins `weather_predictions` to the observed `temperature` at the **target hour**
   (`source_timestamp + horizon_hours`, bucketed to the hour — feature timestamps are not
   minute-aligned), **not** at `prediction_timestamp` (that column is the Spark
   `current_timestamp()` inference-run time, `spark/spark-jobs/inference.py:257`). Rows with no
   target observation, or null predicted/hour, are excluded; persistence baseline =
   observed(target) − stored `observed_temperature` (source). Matches roadmap D3's
   "prediction vs observed at t+h".

WS-D2 page: feature id `analytics`, route path `analytics`, nav label `Analítica`,
`icon` from lucide-react, `order: 50`. It defines its own zod schemas + query hooks inside
`frontend/src/features/analytics/` and renders at least the diurnal heatmap, the daily
anomaly/HDD-CDD view and the wind rose. WS-D1 adds the registry import + nav entry.

## Contract 5 — Quality-meter age recalibration (WS-B)

Measured 2026-09-25 (see sre report): feature age is driven by the ~6 h feature-rebuild
cycle plus the target-drop that discards the newest observation; a healthy throttled
pipeline sawtooths ~6 h → ~12 h. Change in `backend/app/repositories/weather_repo.py`:

```python
QUALITY_OK_AGE_HOURS = 12.0     # ~2x the 6 h rebuild cycle (was 2.0)
QUALITY_WARN_AGE_HOURS = 18.0   # ~3x, i.e. one missed rebuild (was 6.0)
```

`QUALITY_OK_COMPLETENESS = 0.95` and `QUALITY_WARN_COMPLETENESS = 0.80` stay: the 0.98
completeness is real only because of the one-shot backfill, so the gates correctly flag the
broken cadence and must not be loosened. Update `docs/api-contract.md`'s quality section and
the Contract-1 comment accordingly.

## Contract 6 — U6 + U7 behaviour (WS-D1)

- Transitions 120–240 ms; all animation gated by `usePrefersReducedMotion` (reduce → no
  transition/pulse).
- Fresh-data pulse keyed on the data's age (reuse the existing freshness signal /
  `X-Data-Age-Seconds`), not on every render.
- Shimmer skeletons where data is loading (map, charts, table).
- Command palette: hand-rolled, opens on `Cmd/Ctrl-K`, lists routes + stations, Escape
  closes, focus-trapped, `aria-*` correct.
- Keyboard shortcuts: `/` focuses station search; `g` then section keys navigate; documented
  in the palette.
- Chart brushing + synced x-domain via the existing `useChartSync` (fix the noted leak: the
  shared domain must reset when a page unmounts).
- Mobile: bottom sheet for station detail, swipe between stations; do not regress the
  existing bottom nav.

## Contract 7 — Storage budget model (WS-E)

sre measured 2026-09-25 on live Atlas M0: avgObjSize `raw_weather` 1362 B, `weather_features`
3856 B, `weather_predictions` **437 B** (Batch-3 interval columns — the old 405 B is stale),
live city count **15** (includes Pasadena, which is absent from `FALLBACK_CITIES`; that
registry gap is a separate follow-up, not Batch 4). Predictions are keyed
`(city, source_timestamp, horizon)` and written `operationType=replace` (upsert), so an
inference run adds rows **only when the latest feature `source_timestamp` advances**: the
binding rate is the feature-rebuild cadence (`20 */6` = 4/day), not the hourly inference
schedule.

`tests/unit/test_storage_budget.py` must model that **structural** bound, not a physically
impossible hourly one:

- predictions rows/day = `FEATURE_REBUILDS_PER_DAY × cities × horizons`,
  `FEATURE_REBUILDS_PER_DAY = 8` (2× the declared 4/day as cadence-jitter headroom).
- with 15 cities × 5 horizons × 437 B × 90 d → **~368.7 MiB < 384 MiB (75 %)** and
  < 512 MiB cap (≈15.3 MiB headroom). It breaches at **≥14 rebuilds/day** (13/day = 383.5 MiB
  still under), so the constant is the tripwire: any cadence change (e.g. the deferred L1)
  requires a re-measure and re-triggers the check. Test encodes the true boundary.
- do **not** keep 405 B / 14 cities — that reports a false pass by 0.6 MiB.

No TTL or retention change: `weather_predictions` stays 90 d.

## Verification (lead, after integration)

- `cd backend && ../.venv/bin/python -m pytest tests -q`
- `.venv/bin/python -m pytest tests -q`
- `cd frontend && npm run test && npm run typecheck && npm run lint && npm run build`
- Gates: `code-reviewer` on the diff, `qa-engineer` (tests change in every WS),
  `database-engineer` (new sorts/aggregates are index-backed, no migration),
  `sre-engineer` (storage projection with 5× predictions).
- Final live acceptance: `gh workflow run ml-pipeline.yml --ref feat/batch4 -f job=all`,
  then verify `/api/predictions/latest` carries 5 horizons, `model_registry` carries
  `diagnostics`, and `/api/analytics/*` return 200.
