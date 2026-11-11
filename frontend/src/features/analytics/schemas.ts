import { z } from 'zod'

/**
 * Runtime schemas for the `GET /api/analytics/*` responses the UI renders
 * (Batch 4, Contract 4). Parsing is tolerant — missing keys coerce to `null` (or
 * `0` for sample counts) — but the resulting types are strict, so a drifting
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
