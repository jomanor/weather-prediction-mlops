import { z } from 'zod'

/**
 * Runtime schemas for the six `GET /api/analytics/*` responses (Batch 4,
 * Contract 4). Parsing is tolerant — missing keys coerce to `null` (or `0`
 * for sample counts) — but the resulting types are strict, so a drifting
 * backend fails loudly at the query boundary instead of rendering
 * `undefined`.
 *
 * The null-tolerant helpers mirror the ones in `@/api/schemas`; they are
 * module-private there, so the analytics feature keeps its own copy rather
 * than widening the shared API surface (Contract 4: the feature owns its
 * schemas).
 */

const nullableNumber = z
  .number()
  .nullish()
  .transform((value) => value ?? null)

const nullableString = z
  .string()
  .nullish()
  .transform((value) => value ?? null)

/** Counts are never fabricated: a missing count is zero rows, not `null`. */
const sampleCount = z
  .number()
  .nullish()
  .transform((value) => value ?? 0)

/** `GET /api/analytics/daily` — observed daily aggregates + anomaly/heatwave. */
export const dailyPointSchema = z.object({
  date: z.string(),
  tmin: nullableNumber,
  tmax: nullableNumber,
  tmean: nullableNumber,
  hdd: nullableNumber,
  cdd: nullableNumber,
  anomaly: nullableNumber,
  heatwave: z
    .boolean()
    .nullish()
    .transform((value) => value ?? false),
})
export type DailyPoint = z.infer<typeof dailyPointSchema>

export const analyticsDailySchema = z.object({
  city: z.string(),
  days: z.number(),
  generated_at: nullableString,
  points: z.array(dailyPointSchema),
})
export type AnalyticsDaily = z.infer<typeof analyticsDailySchema>

/** `GET /api/analytics/climatology` — day-of-year baseline per city. */
export const climatologyPointSchema = z.object({
  day_of_year: z.number(),
  tmean: nullableNumber,
  tmin: nullableNumber,
  tmax: nullableNumber,
  n: sampleCount,
})
export type ClimatologyPoint = z.infer<typeof climatologyPointSchema>

export const climatologySchema = z.object({
  city: z.string(),
  generated_at: nullableString,
  basis_years: nullableNumber,
  series: z.array(climatologyPointSchema),
})
export type Climatology = z.infer<typeof climatologySchema>

/** `GET /api/analytics/wind-rose` — 16 direction sectors, speed km/h. */
export const windRoseSectorSchema = z.object({
  sector: z.number(),
  count: sampleCount,
  mean_speed: nullableNumber,
})
export type WindRoseSector = z.infer<typeof windRoseSectorSchema>

export const windRoseSchema = z.object({
  city: z.string(),
  days: z.number(),
  generated_at: nullableString,
  sectors: z.array(windRoseSectorSchema),
})
export type WindRose = z.infer<typeof windRoseSchema>

/** `GET /api/analytics/diurnal` — every canonical city × local hour. */
export const diurnalCellSchema = z.object({
  city: z.string(),
  hour: z.number(),
  tmean: nullableNumber,
  n: sampleCount,
})
export type DiurnalCell = z.infer<typeof diurnalCellSchema>

export const diurnalSchema = z.object({
  days: z.number(),
  generated_at: nullableString,
  cells: z.array(diurnalCellSchema),
})
export type Diurnal = z.infer<typeof diurnalSchema>

export const ANALYTICS_VARIABLES = ['temperature', 'humidity', 'pressure', 'wind_speed'] as const
export type AnalyticsVariable = (typeof ANALYTICS_VARIABLES)[number]

/** `GET /api/analytics/correlation` — pairwise Pearson matrix over cities. */
export const correlationSchema = z.object({
  days: z.number(),
  var: z.string(),
  generated_at: nullableString,
  cities: z.array(z.string()),
  matrix: z.array(z.array(nullableNumber)),
})
export type Correlation = z.infer<typeof correlationSchema>

/** `GET /api/analytics/error-by-hour` — MAE/RMSE/bias by horizon and local hour. */
export const errorByHourPointSchema = z.object({
  horizon_hours: z.number(),
  hour: z.number(),
  n: sampleCount,
  mae: nullableNumber,
  rmse: nullableNumber,
  bias: nullableNumber,
  persistence_mae: nullableNumber,
})
export type ErrorByHourPoint = z.infer<typeof errorByHourPointSchema>

export const errorByHourSchema = z.object({
  days: z.number(),
  generated_at: nullableString,
  points: z.array(errorByHourPointSchema),
})
export type ErrorByHour = z.infer<typeof errorByHourSchema>
