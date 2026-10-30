# App rethink — 3 surfaces, mobile-first, no vibe-code

Status: proposal (uncommitted decision). Branch `docs/app-rethink`. Reads after
`docs/handoff.md` and `docs/roadmap.md`; nothing here changes those two.

Goal, in the owner's words: the app "looks very vibe coded, feels bloated, some
tables/views or graphs feel like too much. Rethink what data is enough to make
the app useful and retain as much utility keeping it simple and scalable."
Main use is a **weather app** with an **MLOps extra**; `notes.md`'s future
differentiators (trips, aircraft, MongoDB-Charts-like graphs) must stay
possible but not drive today's build.

**One recommendation:** consolidate the five current surfaces
(`frontend/src/app/registry.ts`: overview, station, benchmark, models,
analytics) into **three mobile-first surfaces** — **Resumen** (weather now),
**Estación** (one station deep-dive), **Sistema** (the whole MLOps trust
surface). Everything else is a re-homing of what already works, not new product
surface.

**Rejected alternative:** keep five (or add) top-level surfaces. It loses
because (1) five items do not fit a 360 px bottom nav without shrinking labels,
(2) the MLOps signals (quality / models / benchmark / analytics) are the same
audience and are currently scattered across three pages plus a self-fetching
panel, and (3) the overview already fires six endpoint groups on mount — more
surfaces means more cold-miss fan-out against a 0.1 CPU Render instance and an
Atlas M0.

---

## 1. Inventory — what exists and what it's worth

### 1.1 Public API surface (26 routes, both `/api` and `/api/v1`)

Mounted once in `backend/app/main.py:_api_routes()`, served under `/api/v1`
(canonical) and `/api` (deprecated alias). Consumers verified by grep against
`frontend/src`.

| Route | Consumer (file) | Fields consumed | Worth | Verdict |
|---|---|---|---|---|
| `GET /health`, `/health/ready` | Render healthCheck; none in UI | `status/service/version/time` | Ops-critical | **KEEP** (do not touch shape) |
| `GET /cities` | `api/queries.ts useCities`, shell search, every page select | `name/latitude/longitude` | Registry ordering + selects | **KEEP** |
| `POST /cities` | `features/overview/queries.ts useAddCity` (CityManager) | `name/lat/lon` | Only city add; **unauthenticated write** | **DEFER** (gate it, §4) |
| `DELETE /cities/{name}` | `features/overview/queries.ts useRemoveCity` (CityManager) | — | Only city remove; **unauthenticated write** | **DEFER** (gate it, §4) |
| `GET /geo/search` | `useGeoSearch` (CityManager) | `name/lat/lon/country/admin1` | City-add picker | **KEEP** |
| `GET /weather/current` | `useStations` (OverviewPage, StationTable) | all `CurrentWeather` | National table + freshness | **KEEP** |
| `GET /weather/current/{city}` | `station/queries.ts useCurrentWeather` | all `CurrentWeather` | Station hero | **KEEP** |
| `GET /weather/history/{city}` | `useHistory` (HistoryChart) | `points[]` | Station chart | **KEEP** |
| `GET /weather/stats/{city}` | **none** — `useStats` defined (`station/queries.ts:29`), never called | `temperature/humidity/pressure/wind_speed/precipitation_total` | 0 in UI | **CUT** |
| `GET /weather/range/{city}` | **none** (grep) | keyset pages | 0 in UI | **CUT** |
| `GET /weather/series` | **none** (grep) | bulk projection | 0 in UI | **CUT** |
| `GET /weather/summary` | **none** (grep) | sparklines | 0 in UI | **CUT** |
| `GET /weather/quality` | `DataQualityMeter` (self-fetching, overview) | `cities[]{completeness,max_gap_hours,age_hours,status}` | MLOps signal, not weather | **MERGE** → Sistema |
| `GET /map/stations` | `useMapStations` (StationMap) | GeoJSON `temperature/weather_code/wind_speed/wind_direction` | Map pins/temp/wind | **KEEP** |
| `GET /predictions/latest` | `useLatestPredictions` (OverviewPage +1h column) | one row per city | +1h delta column | **MERGE** → Estación only |
| `GET /predictions/{city}` | `usePredictions` (station) | `Prediction[]` | Model overlay + interval | **KEEP** |
| `GET /benchmark` | `useBenchmarkSummary` (overview MAE stat + benchmark page) | `cities[]{model/aemet mae,n}` | Trust signal | **MERGE** → Sistema |
| `GET /benchmark/{city}` | `useBenchmark` (BenchmarkPage) | `series[] + metrics` | Model vs AEMET | **MERGE** → Sistema |
| `GET /models` | `useModels` (ModelsPage) | full registry | Trust signal | **MERGE** → Sistema |
| `GET /analytics/daily` | `AnalyticsPage` | `points[]{tmin/tmax/tmean/hdd/cdd/anomaly/heatwave}` | 1 chart | **MERGE** → Sistema |
| `GET /analytics/wind-rose` | `AnalyticsPage` | `sectors[]{count,mean_speed}` | 1 chart | **MERGE** → Sistema |
| `GET /analytics/diurnal` | `AnalyticsPage` | `cells[]{city,hour,tmean}` | 1 chart | **MERGE** → Sistema |
| `GET /analytics/climatology` | **none** — `useAnalyticsClimatology` defined, never rendered | DOY baseline | 0 in UI | **CUT** |
| `GET /analytics/correlation` | **none** — `useAnalyticsCorrelation` defined, never rendered | Pearson matrix | 0 in UI | **CUT** |
| `GET /analytics/error-by-hour` | **none** — `useAnalyticsErrorByHour` defined, never rendered | MAE/RMSE by (horizon, hour) | 0 in UI | **CUT** |
| `POST /refresh` | none in UI | — | Opt-in write, default off | **DEFER** (keep off; §4) |

Net: **13 KEEP/MERGE**, **6 CUT** (3 analytics + 3 weather + stats), **3
DEFER-gate**, and the 3 unused analytics endpoints are the clearest "half-built"
bloat signal — backend, repo, schema and query hook all exist, no component.

### 1.2 Frontend surfaces, panels, tables, charts

| Surface / panel | Endpoint(s) | Content | Worth | Verdict |
|---|---|---|---|---|
| **Overview** `OverviewPage.tsx` | cities + current + map-stations + latest-predictions + benchmark + quality (via `DataQualityMeter`) | 6 endpoint groups on mount | Headline + map + list are useful; the last 3 are not weather | **KEEP** map + list, **MOVE** quality/benchmark off it |
| — map `StationMap.tsx` | map/stations | 3 basemaps, 4 layer toggles (relieve/radar/terreno3D/edificios3D), radar scrubber, 2 legends | Temp + wind + radar are weather; hillshade/3D are demo chrome | **KEEP** temp/wind/radar; **CUT** terrain3D/buildings3D/hillshade from primary UI |
| — stat strip (5 `Stat`) | current + benchmark | monitored / avg temp / avg humidity / avg wind / MAE 24 h | MAE belongs in Sistema; the other 4 duplicate the table | **CUT** to 2 (estaciones + act.) |
| — `StationTable.tsx` | current + latest-predictions | 15 rows × 7 cols | 7 cols don't fit 360 px | **KEEP** as 4-field compact card list on mobile |
| — `StationSheet.tsx` | (props) | mobile bottom sheet duplicating `CurrentConditions` + prediction | Duplicates the Estación surface | **CUT** → deep-link to Estación |
| — `CityManager.tsx` | cities + geo + add/remove | add/remove stations | Operator action, not daily weather | **MOVE** → Sistema (or a collapsed "Gestionar" row) |
| — `DataQualityMeter.tsx` | quality | 3 readouts + 15-row table | MLOps signal | **MOVE** → Sistema |
| **Estación** `StationPage.tsx` | current/{city} + history + predictions/{city} | hero + chart + prediction | The core weather detail | **KEEP**, tighten |
| — `CurrentConditions.tsx` | current/{city} | hero (`text-5xl`) + 6 readouts | Useful, oversized | **KEEP**, hero → `text-2xl` max |
| — `HistoryChart.tsx` | history + predictions | temp/rain/wind + model overlay + interval band + brush + legend toggles | Best-in-app chart; keep as the one chart style | **KEEP** (canonical chart) |
| — `LatestPrediction.tsx` | predictions/{city} | 3 readouts + 6 metadata rows + last 6 runs | Metadata is registry noise for a weather user | **KEEP** the 3 readouts, **CUT** model-name/version/run list to Sistema |
| — `DerivedPanel.tsx` | (client `lib/derived.ts`) | q, dew point, u/v wind | ML-internal quantities, not weather | **CUT** (defer to Sistema if ever wanted) |
| **Benchmark** `BenchmarkPage.tsx` | benchmark/{city} + benchmark | chart + metrics table + residual + national table | Honest trust check | **MERGE** → Sistema |
| — `BenchmarkChart.tsx` / `ResidualChart.tsx` / `MetricsCompare.tsx` / `AemetNotice.tsx` | benchmark | observed/model/aemet | Useful, honest | **MERGE** → Sistema |
| **Modelos** `ModelsPage.tsx` | models | 11-col table + 8-row expandable detail | Raw registry dump; too dense | **MERGE** → Sistema, production-first |
| **Analítica** `AnalyticsPage.tsx` | diurnal + daily + wind-rose | 3 charts (of 6 endpoints) | Nice-to-have; not core weather | **MERGE** → Sistema, keep 3, cut 3 |
| **Shell** `AppShell.tsx` | — | header, 5-item nav, `StationSearch`, `UnitToggle`, `ThemeToggle`, footer, bottom nav, `CommandPalette`, `g`-shortcuts | Search + palette are power-user; 5 nav items are the bloat | **KEEP** search/palette/theme/units, **CUT** to 3 nav items |

URL state today (all via `useUrlState` / `useStationSelection`): `city`
(app-wide), `tab` (overview), `hours`/`var`/`horizon` (station), `range`
(benchmark), `days` (analytics). The 3-surface IA keeps every key; `tab`
becomes the Sistema tab selector.

---

## 2. Proposed 3-surface IA (mobile-first)

Names: **Resumen**, **Estación**, **Sistema** (Spanish, matches the existing
copy; "Sistema" beats "Modelos" because it holds quality + models + benchmark +
analytics, not just models).

### Bottom-nav contract

| Item | Path | Deep link | Back behaviour |
|---|---|---|---|
| Resumen | `/` | — | n/a (home) |
| Estación | `/estaciones` | `/estaciones?city=Madrid` (+ `var`, `hours`, `horizon`) | browser back unwinds `city`; nav re-selects Estación |
| Sistema | `/sistema` | `/sistema?tab=modelos` (or `calidad`/`benchmark`/`analitica`) | browser back unwinds `tab` |

Rules:

- **3 items only**, fixed bottom bar on `<md`, same header row on `≥md`
  (`AppShell.tsx` already renders both; only `NAV_ITEMS` changes).
- `city` stays the only app-wide param; `tab` (Sistema), `var`/`hours`/`horizon`
  (Estación) are surface-local and survive navigation (existing `useUrlState`
  preserves unknown params — verified in `useUrlState.ts`).
- No nested stacks, no modals for primary navigation. Tapping a station is a
  **link** to `/estaciones?city=X`, not a sheet.
- `--nav-h` + `--safe-bottom` offsets in `index.css` stay untouched — the
  mobile layout contract does not change, only the item count.

### What a first visit shows

`/` → map centred on Iberia (`IBERIA_CENTER [-3.9, 39.9]` in `StationMap.tsx`),
compact station list sorted by name, freshness one-liner. Default selection =
first registry city (existing `resolveCity`). **Three endpoint groups max**
(`/cities`, `/weather/current`, `/map/stations`). No `/predictions/latest`, no
`/benchmark`, no `/weather/quality` on the home screen.

### What a station deep-link shows

`/estaciones?city=Madrid` → hero (temp + weather label + 3 key readouts) →
`HistoryChart` (temp default, `48 h`) with model overlay + interval band →
`LatestPrediction` (temp prevista + prob. lluvia + interval). Unknown city falls
back to the first registry city (`resolveCity`).

### ASCII wireframes (phone, ~360–390 px)

**Resumen**

```
┌────────────────────────────────┐
│ meteoml            ⌕   ☾   (°C)│   header: search, theme, units
├────────────────────────────────┤
│                                │
│      MAPA  (temp + viento)     │   ~45vh, one collapsible "Capas"
│         ●  ●       ●           │   (radar on/off + relieve advanced)
│                                │
├────────────────────────────────┤
│ 15 estaciones · Act. 20:00  ●  │   freshness one-liner (no MAE stat)
│                                │
│ Madrid      21,4°  ☁   12 km/h │   compact list: temp + tiempo + viento
│ Barcelona   19,8°  ☀    8 km/h │   + edad; tap → /estaciones?city=X
│ Valencia    20,9°  ☁    5 km/h │
│ …                              │
├────────────────────────────────┤
│  Resumen    Estación    Sistema │   3-item bottom nav
└────────────────────────────────┘
```

**Estación**

```
┌────────────────────────────────┐
│ ← meteoml          ⌕    ☾  (°C)│
├────────────────────────────────┤
│ Madrid · 21,4°    ☁ Cubierto   │   hero (text-2xl, not text-5xl)
│ Sens 20,1° · Hum 61% · V 12 km/h NO │
│ Predicción +1h · 22,0° · lluvia 0,1 │
├────────────────────────────────┤
│ Evolución  [Temp|Lluvia|Viento]│
│ ┌────────────────────────────┐ │
│ │  HistoryChart (brush+zoom) │ │   the one canonical chart
│ │  modelo — banda — observado│ │
│ └────────────────────────────┘ │
│ Intervalo 20,9–23,1 °C (80%)   │
├────────────────────────────────┤
│  Resumen    Estación    Sistema │
└────────────────────────────────┘
```

**Sistema** (`?tab=modelos`)

```
┌────────────────────────────────┐
│ ← meteoml          ⌕    ☾  (°C)│
├────────────────────────────────┤
│ Sistema [Calidad|Modelos|Benchmark|Analítica] │ 4 tabs, one URL param
├────────────────────────────────┤
│ Modelos registrados (prod.)    │
│ temp_1h GBT   +1h  skill 0,53 ●│   production-first, skill/coverage/
│ temp_24h GBT  +24h skill −0,12 │   drift only; detail on expand
│ rain_1h RF    +1h  brier 0,21  │
│ …                              │
├────────────────────────────────┤
│  Resumen    Estación    Sistema │
└────────────────────────────────┘
```

### Desktop adaptation

- Top nav (3 items) replaces the bottom bar (`AppShell.tsx` already has the
  `sm:flex` header nav).
- **Resumen:** map left (~60 %) + compact list right (~40 %).
- **Estación:** `HistoryChart` left (~60 %) + `LatestPrediction` right (~40 %)
  (the existing `xl:grid-cols-[1.6fr_1fr]`).
- **Sistema:** horizontal tab bar + full-width panels; Models/Benchmark keep
  their current wide tables where legible.

### What the app *stops* showing, and why the rest is enough

Stops: MAE stat on home, data-quality meter on home, city manager on home, the
mobile station sheet, the derived-quantities panel, model-name/version/run-list
on the station prediction card, hillshade/terrain-3D/buildings-3D as primary
map controls, the 11-column models table, and 3 unrendered analytics endpoints.

Why enough: a weather user needs "what is it now, what was it, what's next,
across my stations" (Resumen + Estación); an MLOps user needs "is data fresh,
is the model honest, does it beat AEMET" (Sistema). Everything cut is either a
duplicate of those two jobs or a half-built endpoint.

### Data budget per surface (exact fields/endpoints)

| Surface | Endpoint(s) | Fields kept | Dropped from render |
|---|---|---|---|
| Resumen | `/cities` | `name, latitude, longitude` | — |
| | `/weather/current` | `city, temperature, weather_code, wind_speed, wind_direction, humidity, precipitation, observed_at` | `pressure, cloud_cover, apparent_temperature` (list), `lat/lon` (already on map) |
| | `/map/stations` | `geometry.coordinates, properties{city, temperature, weather_code, wind_speed, wind_direction, observed_at}` | `relative_humidity, precipitation, apparent_temperature` |
| Estación | `/weather/current/{city}` | full `CurrentWeather` (hero + readouts) | — |
| | `/weather/history/{city}?hours=48&limit=500` | `observed_at, temperature, precipitation, wind_speed, wind_direction` | `pressure, cloud_cover, apparent_temperature, humidity` (chart doesn't use them) |
| | `/predictions/{city}?limit=24&horizon=1` | `predicted_temperature, predicted_rain, temp_lower, temp_upper, interval_level, horizon_hours` | `temp_model_name/version, rain_model_name/version, source/prediction_timestamp` (→ Sistema) |
| Sistema | `/weather/quality?days=7` | `city, completeness, max_gap_hours, age_hours, status` | `expected_hours, observed_hours, null_rate` (keep in API, drop from UI) |
| | `/models` | `name, target, horizon_hours, metrics{skill_score, coverage}, diagnostics(drift), stage, created_at` | full 11-col + 8-row detail (fold into expand) |
| | `/benchmark` + `/benchmark/{city}?hours=24` | summary `{model.mae, aemet.mae, n}`; detail `series + metrics` | `residual_aemet` (show model residual only) |
| | `/analytics/diurnal`, `/analytics/daily`, `/analytics/wind-rose` | as today | climatology / correlation / error-by-hour (cut entirely) |

---

## 3. Anti-vibe-code design system

Do **not** change the font family — `notes.md` UI-03 is a pending user decision.
Design around the current `Inter Variable` (UI) + `JetBrains Mono Variable`
(numerals/charts), already loaded in `frontend/src/styles/index.css`.

### Scales (formalise what `index.css` + the UI already imply)

| Token | Value | Use |
|---|---|---|
| Type | `10/11/12/13/14/16/18/22` px | badges 10 · labels/hints 11 · body/table 12 · panel title 13 · readout 14 · page title 18 · single readout 22. **Ban** the `text-4xl/5xl` hero (`CurrentConditions.tsx`, `StationSheet.tsx`). |
| Spacing | 4 px grid: `4/8/12/16/20/24/32` | panel `p-4` (16) · header `py-3` (12) · row `py-2` (8) · gaps `1–5`. No `2.3rem`/`13px` off-scale values. |
| Radius | `--radius-panel: 0.25rem` (4) panels · `3px` controls · `2px` chart cells/ramp bars | nothing above 6 px; no `rounded-2xl`. |
| Borders | hairline `--line`, strong `--line-strong` | every container edge; no borderless floating cards. |
| Shadows | `--shadow: 0 1px 2px` only; sheet elevation `0 -8px 24px` once | no glows, no `shadow-lg` stacking. |

### Color roles (reuse existing tokens — no new palette)

From `index.css` + `lib/chart-theme.ts` + `lib/ramps.ts`:

- **Chrome neutrals:** `--bg/--panel/--panel-2/--line/--line-strong/--fg/--fg-2/--fg-3`.
- **Interaction only:** `--accent` (+ `--accent-fg`, `--accent-soft`). Never tint chrome with data colours.
- **Data series identity:** `--obs` (observed), `--model` (Spark GBT), `--aemet`. Reserved; never for chrome.
- **Semantic:** `--ok/--warn/--bad` — only paired with a label/icon (never colour alone).
- **Data ramps (quantitative magnitude only):** `--temp-1..7`, `--rain-1..5`, `--wind-1..4`. Never for series identity.

Resolution rule stays `lib/chart-theme.ts readChartPalette()`; MapLibre paint
values are the documented exception (hex literals in `chart-theme.ts`).

### Density, motion, states

- Density: table row ≈ 36 px (`py-2`), panel padding 16 px, mobile touch targets
  ≥ 44 px (already enforced for map controls in `index.css`).
- Motion budget: `120–240 ms` transitions, `240 ms` fade-rise, `1.4 s` shimmer —
  all gated by `prefers-reduced-motion` (already in `index.css`). Map `easeTo`
  durations collapse to 0 under reduced motion (`StationMap.tsx motionDuration`).
- Empty / loading / error: **only** `components/ui/Feedback.tsx`
  (`Skeleton`/`ChartSkeleton`/`TableSkeleton`/`MapSkeleton`/`LoadingBlock`/
  `EmptyState`/`ErrorState`). No new spinners, no inline `if (!x) return null`.

### Chart conventions (one style)

- One axis/tick/grid/cursor source: `lib/chart-theme.ts`
  (`CHART_MARGIN`, `chartAxisProps`, `chartGridProps`, `chartLineCursor`,
  `chartBarCursor`, `chartAxisLabel`).
- X-axis is always time, oldest → newest (`docs/api-contract.md` convention).
- Series colours from `ChartPalette` only; dashes fixed (`observed` solid,
  `model` `4 3`, `aemet` `2 3`).
- No splines / interpolation (`notes.md` UI-07 asks for it — **reject**: honest
  gaps read better than invented curvature; `connectNulls=false` is already the
  convention).
- One chart component for line/area (`HistoryChart`); bar+line mixing allowed
  only where already justified (temp + precip dual axis, daily anomaly
  HDD/CDD + anomaly line).

### Component inventory — keep vs remove

| File | Action |
|---|---|
| `components/ui/{Panel,Feedback,Readout,PageHeader,Segmented,Badge,Button}.tsx` | **KEEP** — the a11y primitives (radiogroup segmented, focus ring, `aria-pressed` legend) are the best part of the current UI |
| `components/charts/{HistoryChart,BenchmarkChart,ResidualChart,ChartTooltip,ChartBrush,SeriesLegend,chart-brush,history-rows}.tsx` | **KEEP** — converge onto `HistoryChart`'s style |
| `components/charts/Sparkline.tsx` | **REMOVE** — no consumer (grep: only its own file) |
| `components/shell/{CommandPalette,StationSearch,ThemeToggle,UnitToggle,shortcuts,command-palette}.tsx` | **KEEP** — power-user shortcuts are a differentiator, not bloat |
| `components/map/{StationMap,RadarControls,radar,station-geojson,weather-icons}.tsx` | **KEEP** but collapse the layer panel (below) |
| `components/DataQualityMeter.tsx` | **MOVE** to Sistema feature |
| `features/analytics/components/{DiurnalHeatmap,WindRose,DailyAnomalyChart}.tsx` | **KEEP** (move into Sistema) |
| `features/overview/StationSheet.tsx` + `features/overview/sheet.ts` | **REMOVE** — replaced by deep-link |
| `features/station/DerivedPanel.tsx` + `lib/derived.ts` | **REMOVE** from UI (keep `lib/derived.ts` only if a Sistema "datos" tab ever wants it) |

### Explicitly banned (the generated look)

1. **Gradients / glows** — no `bg-gradient-*`, no `drop-shadow`, no
   `backdrop-blur` overlays. (The one existing `backdrop-blur-[2px]` on
   `CommandPalette.tsx` goes; the `shimmer` skeleton gradient stays, it's a
   loading state, not decoration.)
2. **Icon soup** — one icon per row/cell. Drop the triple
   `weather-icon + Check + ArrowUpRight` cluster in `StationTable.tsx`; selection
   is shown by the left accent bar + `aria-selected`, not a check.
3. **Mixed chart styles** — `DailyAnomalyChart.tsx` re-declares local
   `axisProps` (hardcoded `'JetBrains Mono Variable, monospace'`) instead of
   `chart-theme.ts`. Converge to the shared helper.
4. **Oversized cards / heroes** — `text-5xl` temperature (`CurrentConditions.tsx`)
   and `text-2xl` sheet readout (`StationSheet.tsx`) collapse to `text-2xl` max.
5. **Inconsistent number formats** — models table shows RMSE/MAE to 3 decimals,
   benchmark to 2. Pick one precision per metric in `lib/format.ts` and hold it
   (temperature 1 dp, error 2 dp, skill 2 dp, percentages 0–1 dp).
6. **Half-built features** — any endpoint without a rendered consumer
   (climatology / correlation / error-by-hour / stats) is cut, not left dormant.

---

## 4. Security and scalability review

### Public surface today

The API is **entirely unauthenticated** and public (`main.py` — no auth
middleware). Read paths are fine for a public weather app; the two write paths
are not.

### Concrete gaps, with severity

| # | Gap | Evidence | Severity |
|---|---|---|---|
| 1 | `POST /cities` and `DELETE /cities/{name}` are unauthenticated writes. `DELETE` removes a station from the registry that ingestion reads, so it stops data for a city and silently degrades the quality meter + next feature rebuild. | `backend/app/routers/cities.py` (`create_city`, `delete_city`); UI `CityManager.tsx` is the only legitimate consumer | **High** |
| 2 | `POST /refresh` is unauthenticated; mitigated only by default-off `REFRESH_ENABLED=false` + a **per-process** limiter that resets on free-tier sleep. | `routers/refresh.py`, `core/rate_limit.py` | **Medium** (accepted, documented; do not default-on) |
| 3 | Security headers are set by the **Netlify proxy**, not the backend. Direct calls to Render get no CSP/HSTS/frame/mime headers. | `netlify.toml` sets `X-Frame-Options/X-Content-Type-Options/Referrer-Policy/Permissions-Policy`; `main.py` sets none; no CSP anywhere | **Medium** (read-only JSON limits impact, but the gap is real) |
| 4 | CORS `allow_credentials=True` + `allow_methods=["*"]` + `allow_headers=["*"]` | `main.py` CORSMiddleware | **Low** (origins explicit; tighten to explicit methods/headers) |
| 5 | All caching/limiting/semaphore state is **in-process single-instance**: `TTLCache` (`core/cache.py`), `RefreshRateLimiter` (`core/rate_limit.py`), `AemetService._cache` (`services/aemet.py`), and the module-level `_analytics_semaphore` (`routers/analytics.py`). Multi-worker/instance breaks every one silently. | the four files above | **Medium** if PLAT-03 "distributed" is pursued; documented but not enforced |
| 6 | `_analytics_semaphore` is **module-level** (shared across `create_app()` instances in one process, and across tests), not `app.state`. | `routers/analytics.py:34` | **Low** (cosmetic today; move to `app.state`) |
| 7 | `pytest`/`pytest-asyncio` shipped in the production image | `backend/requirements.txt` (no dev split) | **Low** (image bloat + supply chain) |
| 8 | No general read rate limit. Cold-miss fan-out is bounded by LRU (256) + single-flight + 300 s TTL, but a cold burst hits Atlas M0 as the backstop. | `core/cache.py`, `routers/analytics.py` semaphore | **Low** (cache + ETag cover it; M0 is the tripwire) |
| 9 | Secrets are env-only and not committed (`MONGO_URI`, `AEMET_API_KEY`); `RENDER_DEPLOY_HOOK` still pending user-side; `.env.production` is gitignored but contains `&` (parse, never `source`). | `render.yaml`, `docs/handoff.md` | **OK** (no action beyond the handoff note) |
| 10 | Input validation is solid: pydantic schemas, query bounds (`hours 1–168`, `limit` caps, `days 7–365`, city name ≤ 100), error envelope strips submitted values, no query string logged. | `routers/*`, `core/errors.py`, `core/logging.py` | **OK** |

### What must change before trips / aircraft

- **Auth boundary (PLAT-02) first.** Any per-user write (saved trips, saved
  aircraft overlays) needs authn + authz + an audit log; there is none today.
  The `security-engineer` gate is mandatory, per `notes.md` PLAT-02.
- **Server-side proxy for external keys.** Aircraft (OpenSky/ADS-B) and any
  trip/geocoding vendor need keys held server-side, never in the SPA, with key
  rotation + quota caps + rate limits + caching — the exact pattern already used
  for AEMET (`services/aemet.py`) and Open-Meteo geo (`services/geo.py`).
- **Background freshness, not per-view fetch.** Trips/aircraft must be
  pre-fetched by a cron (GitHub Actions) or a keep-warm path, never fetched per
  render against a sleeping free instance. Same rule as the existing refresh
  endpoint's "never a browser-triggered workflow dispatch" note.
- **Arbitrary-coordinate predictions (trips).** `weather_predictions` is keyed
  `(city, source_timestamp, horizon_hours)` and the API is city-keyed. Trips
  need a lat/lon prediction surface — a new contract, and a storage re-measure
  first (handoff: 15-min ingest already projects ~637 MiB vs the 512 MB cap;
  trips would multiply rows).
- **Distributed limiter/cache.** PLAT-03 (on-prem / multi-node) invalidates the
  single-instance assumptions in gap #5; a Mongo-backed counter + shared cache
  must replace the in-process state before any horizontal scale.

---

## 5. Legacy and dead weight

A parallel owner is deleting `kafka/` and `torchcast/`. In this worktree
`torchcast/` is **already gone** (no such directory) and `kafka/` is still
present (44 KB, 6 files: `kafka-producer/{Dockerfile,producer.py,requirements.txt}`,
`kafka-consumer/{Dockerfile,consumer.py,requirements.txt}`).

After that deletion, the remaining dead weight is:

| Item | Why dead | Action |
|---|---|---|
| `scripts/sync_to_atlas.py`, `scripts/sync_to_supabase.py` | no references anywhere; Supabase/Atlas-sync not in the stack (grep over `*.py/*.yml/*.md/*.toml` returns nothing) | **remove** |
| `tests/unit/test_producer.py`, `tests/unit/test_consumer.py` | test the deleted Kafka producer/consumer | **remove** with `kafka/` |
| `docker-compose.yml` Kafka broker/zookeeper/`kafka-producer`/`kafka-consumer`/volumes | Kafka is gone | **remove** (keep the local Mongo + API services) |
| `docker-compose.yml` `mlflow` service + `MLFLOW_TRACKING_URI` | MLflow is now opt-in/ephemeral and dropped from CI (`docs/model-lifecycle.md`) | **remove** or mark local-only |
| `README.md` tree refs (`kafka/`, `frontend/nginx.conf`, sync scripts) | stale — the frontend no longer ships nginx (Netlify), Kafka is gone | **fix** |
| `backend/app/routers/weather.py` `/stats`, `/range`, `/series`, `/summary` + `backend/app/services/stats.py` + `backend/app/schemas/weather.py` (`SeriesResponse`, `SummaryResponse`, `WeatherRangeResponse`, `WeatherStats`) + `frontend/src/api/schemas.ts` (`statsResponseSchema`) + `frontend/src/features/station/queries.ts` (`useStats`) | zero frontend consumers (grep) | **remove** in P2, update `docs/api-contract.md` |
| `frontend/src/components/charts/Sparkline.tsx` | no consumer | **remove** |
| `useAnalyticsClimatology` / `useAnalyticsCorrelation` / `useAnalyticsErrorByHour` in `frontend/src/features/analytics/queries.ts` + their `schemas.ts` blocks + the three backend routes + repo methods + tests | built, never rendered | **remove** (the clearest "half-built" bloat) |
| `docs/batch3-contract.md`, `docs/batch4-contract.md` | frozen-interface snapshots, superseded by `docs/api-contract.md` + `docs/roadmap.md` | **remove** after moving any still-unique hard rule into `handoff.md`/`roadmap.md` |
| `backend/requirements.txt` `pytest`/`pytest-asyncio` | prod image ships test deps | **split** into dev requirements |
| `weather_data` collection (0 rows) | vestigial — but **do NOT delete or "fix"** (handoff hard rule 3) | **leave in place** |

---

## 6. Phased plan

Effort: S < 1 day, M 1–3 days, L > 3 days.

### P0 — visual/IA consolidation (M)

What: collapse 5 → 3 surfaces; re-home quality/benchmark/models/analytics into
Sistema; replace `StationSheet` with the Estación deep-link; reduce map chrome;
cut the overview's 6-endpoint fan-out to 3.

Acceptance criteria:

- `NAV_ITEMS` = 3 (`frontend/src/app/registry.ts`); bottom nav renders 3 items
  at 360 px; no horizontal scroll at 360/768/1024/1440.
- Tapping a map pin or a list row navigates to `/estaciones?city=X`.
- Resumen mount fires only `/cities`, `/weather/current`, `/map/stations`
  (verify in network panel); `/benchmark`, `/predictions/latest`,
  `/weather/quality` no longer load on `/`.
- Estación drops `DerivedPanel` and the model-name/version/run-list; hero ≤
  `text-2xl`.
- Sistema (single surface) holds Calidad / Modelos / Benchmark / Analítica
  behind `?tab=`.
- `npm run typecheck && npm run lint && npm run test && npm run build` green;
  axe/a11y clean; finish-gate review passes (no banned patterns from §3).

Risks: deep-link regressions (URL `city` param); map touch regressions (tap
tolerance / cooperative gestures in `StationMap.tsx`); losing the +1h forecast
and MAE from the home screen (they move, not vanish — flag in the PR).

### P1 — data-freshness honesty + Sistema detail (M)

What: read `X-Data-Age-Seconds` in `apiGet` (currently unused — `client.ts`
never surfaces response headers) instead of payload-timestamp inference in
`useFreshness`; production-first Models table (skill/coverage/drift only);
trim models metadata; cut the 3 unrendered analytics endpoints end-to-end.

Acceptance criteria:

- Freshness pulse keys on the server header, not `observed_at` arithmetic.
- Sistema "Modelos" shows production first with skill + coverage + drift badge;
  staging/superseded collapsed.
- `/analytics/{climatology,correlation,error-by-hour}` + their schemas/queries/
  repo methods/tests removed; `docs/api-contract.md` updated.
- AEMET Spain-only notice (`AemetNotice.tsx`) still renders for Pasadena.

Risks: header plumbing touches the shared client (schema tests must keep
passing); removing analytics endpoints is safe only because grep proves zero
consumers — re-verify before cutting.

### P2 — cuts/renames + hardening (S–M)

What: remove `/stats`, `/range`, `/series`, `/summary` + `services/stats.py` +
schemas + `useStats`; remove `Sparkline`; remove sync scripts + Kafka tests;
prune `docker-compose.yml`; fix `README.md`; delete `batch3/batch4-contract.md`;
gate city writes; tighten CORS; split test deps.

Acceptance criteria:

- The four weather endpoints return 404; `docs/api-contract.md` reflects it;
  `ruff`/`black`/`pytest` (backend + root) green.
- `grep -r` shows no references to removed files.
- `POST/DELETE /api/cities` gated behind `CITY_WRITES_ENABLED` (default off) or
  an auth boundary; CityManager disabled accordingly.
- CORS methods/headers explicit; `pytest` out of the prod image.

Risks: coordinated frontend + backend + contract change; **do not** remove the
`/api` alias (handoff: Render health + Netlify proxy depend on it); **do not**
rename collections or the prediction key.

### Must NOT break (applies to every phase)

1. **Render health path** `GET /api/health` (and `/health/ready` 503 body shape).
2. **Netlify edge proxy** `/api/*` → onrender + SPA fallback (`netlify.toml`).
3. **Deep links** — `?city=`, `?tab=`, `?var`/`?hours`/`?horizon`/`?range`/`?days`
   (and their defaults).
4. **Mobile offsets** `--nav-h` + `--safe-bottom` in `index.css`.
5. **Handoff hard rules 1–5** — never run inference against stale features;
   city-prefixed sorts + `allowDiskUse`; no collection rename; charts oldest →
   newest; one writer per tree.

---

## 7. `notes.md` extension points in the 3-surface IA

`notes.md` is gitignored and lives in the parent checkout
(`/home/joman/Proyectos/weather-prediction-mlops/notes.md`). The entries below
are intent, not approved scope.

| Idea | Fits in | One-line constraint |
|---|---|---|
| UI-01 (map city popup) | Resumen map | Replace the deleted sheet with a 4-field popover (estado/temp/viento/lluvia) + "Abrir estación" → `/estaciones?city=X`; never duplicate the Estación surface. |
| UI-03 (typography) | Design system (§3) | Blocked on the user's font choice; apply once to `index.css` `--font-sans`/`--font-mono`, re-check widths/`line-height`. |
| UI-05 (map layer bugs) | Resumen map | Fix the rain "zoom not supported" overlay by capping `maxzoom` and attenuating out-of-range (already `maxzoom:10` in `radar` — verify + inline message); fix legend/dropdown overlap; these are UX bugs, not features. |
| UI-09 (mobile) | All three | The 3-surface IA *is* the mobile answer; legend must be collapsible and 320–390 px must not scroll horizontally. |
| UI-10 (wind) | Resumen map | Diagnose data-vs-scale before building (`notes.md`); the wind arrow + ramp already exist in `StationMap.tsx` — likely a ramp/visibility issue, not a new layer. |
| UI-11 (layer catalog) | Resumen map | Declarative layer catalog (id/type/source/attribution/zoom bounds) is the prerequisite for aircraft; start it here, not as a one-off. |
| PLAT-01 (trips) | Future surface or Resumen mode | Needs an arbitrary lat/lon prediction contract + storage re-measure (city-keyed today); do not start before the L1 storage decision in `handoff.md`. |
| PLAT-02 (user log / on-prem auth) | Cross-cutting | Authn/authz/audit gate before any write surface; mandatory `security-engineer` review. |
| PLAT-03 (Spark vs on-prem/sklearn) | Architecture | Do not fork the compute model now; the single-instance cache/limiter (§4 gap 5) must be resolved first. |
| PLAT-04 (streetview) | Depends on PLAT-01 | Defer until trips are real; source/licensing unresolved. |
| PLAT-05 (Kafka via Atlas) | Backend research | Research only; current ingest already works and Kafka/ is being deleted. |
| PLAT-06 (MongoDB-Charts-like graphs) | Sistema (Analítica) + the shared chart component | One chart base (`HistoryChart` + `chart-theme.ts`), no new chart library; catalogue of chart types before building. |
| PLAT-07 (aircraft, real-time ADS-B) | Resumen map as a UI-11 layer | External keyed API → server proxy + quota + background freshness; license/attribution review first. |

---

## Rejected alternatives (summary)

- **Five surfaces** (status quo) — rejected: bottom-nav overflow, MLOps
  scatter, 6-endpoint home-screen fan-out.
- **Full restyle / new component library** — rejected: working a11y primitives
  already exist in `components/ui/` and tokens in `index.css`; a restyle churns
  tested behaviour for no user value and risks the handoff hard rules.
- **Interpolate chart lines (UI-07 splines)** — rejected: honest gaps beat
  invented curvature; `connectNulls=false` stays.
