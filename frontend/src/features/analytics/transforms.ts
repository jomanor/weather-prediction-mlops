import { compassPoint, isNum } from '@/lib/format'

import type { DailyPoint, DiurnalCell, WindRoseSector } from '@/features/analytics/schemas'

/**
 * Pure transforms between the analytics payloads and what the views render.
 * Kept out of the components so the non-trivial aggregation (wind-rose sectors,
 * the city × hour matrix, heatwave runs) is unit-testable.
 */

export const HOURS = Array.from({ length: 24 }, (_, hour) => hour)
export const SECTOR_COUNT = 16
const SECTOR_WIDTH = 360 / SECTOR_COUNT

/** `days` bounds per endpoint, mirroring Contract 4's validation. */
export interface DaysBound {
  min: number
  max: number
  default: number
}

export const ANALYTICS_BOUNDS = {
  daily: { min: 7, max: 180, default: 90 },
  windRose: { min: 7, max: 365, default: 90 },
  diurnal: { min: 7, max: 365, default: 90 },
  correlation: { min: 7, max: 365, default: 90 },
  errorByHour: { min: 7, max: 90, default: 30 },
} as const satisfies Record<string, DaysBound>

/** Clamp a caller-supplied `days` to the endpoint bound; non-finite → default. */
export function clampDays(days: number, bound: DaysBound): number {
  const value = Number.isFinite(days) ? Math.trunc(days) : bound.default
  return Math.min(bound.max, Math.max(bound.min, value))
}

export interface WindRosePetal {
  /** 0 = N, clockwise in 22.5° steps. */
  sector: number
  /** Compass label: N, NNE, NE… */
  label: string
  count: number
  meanSpeed: number | null
}

export interface WindRoseModel {
  petals: WindRosePetal[]
  maxCount: number
  total: number
  maxSpeed: number | null
}

/**
 * Normalise the backend's sparse sector list into all 16 petals. Counts are
 * summed per sector; the mean speed is count-weighted across duplicate sectors,
 * so a partial payload can never skew the rose towards a single row.
 */
export function buildWindRose(sectors: readonly WindRoseSector[]): WindRoseModel {
  const counts = Array.from({ length: SECTOR_COUNT }, () => 0)
  const speedSum = Array.from({ length: SECTOR_COUNT }, () => 0)
  const speedCount = Array.from({ length: SECTOR_COUNT }, () => 0)

  for (const sector of sectors) {
    const index = ((Math.round(sector.sector) % SECTOR_COUNT) + SECTOR_COUNT) % SECTOR_COUNT
    const count = Math.max(0, Math.trunc(sector.count))
    counts[index] += count
    if (count > 0 && isNum(sector.mean_speed)) {
      speedSum[index] += sector.mean_speed * count
      speedCount[index] += count
    }
  }

  const petals = counts.map((count, index) => ({
    sector: index,
    label: compassPoint(index * SECTOR_WIDTH),
    count,
    meanSpeed: speedCount[index] > 0 ? speedSum[index] / speedCount[index] : null,
  }))

  const speeds = petals.map((petal) => petal.meanSpeed).filter(isNum)
  return {
    petals,
    maxCount: counts.reduce((max, count) => Math.max(max, count), 0),
    total: counts.reduce((sum, count) => sum + count, 0),
    maxSpeed: speeds.length ? Math.max(...speeds) : null,
  }
}

export interface DiurnalRow {
  city: string
  /** tmean per local hour 0..23; `null` when the hour has no sample. */
  values: (number | null)[]
  counts: number[]
}

export interface DiurnalMatrix {
  cities: string[]
  rows: DiurnalRow[]
  min: number | null
  max: number | null
}

/**
 * Pivot the flat cell list into cities × 24 hours, preserving first-seen city
 * order (the backend emits the canonical order). Missing hours stay `null` so
 * the heatmap draws an explicit gap rather than a fabricated value.
 */
export function buildDiurnalMatrix(cells: readonly DiurnalCell[]): DiurnalMatrix {
  const rows = new Map<string, DiurnalRow>()
  const cities: string[] = []
  let min: number | null = null
  let max: number | null = null

  for (const cell of cells) {
    let row = rows.get(cell.city)
    if (!row) {
      row = {
        city: cell.city,
        values: Array.from({ length: HOURS.length }, () => null) as (number | null)[],
        counts: Array.from({ length: HOURS.length }, () => 0),
      }
      rows.set(cell.city, row)
      cities.push(cell.city)
    }
    const hour = Math.trunc(cell.hour)
    if (hour < 0 || hour >= HOURS.length) continue
    row.values[hour] = cell.tmean
    row.counts[hour] = cell.n
    if (isNum(cell.tmean)) {
      min = min === null ? cell.tmean : Math.min(min, cell.tmean)
      max = max === null ? cell.tmean : Math.max(max, cell.tmean)
    }
  }

  return {
    cities,
    rows: cities.map((city) => rows.get(city) as DiurnalRow),
    min,
    max,
  }
}

export interface HeatwaveRun {
  /** Inclusive indices into the `points` array. */
  start: number
  end: number
}

/** Contiguous runs of `heatwave: true` days (backend: ≥3 days at tmax ≥ 35 °C). */
export function heatwaveRuns(points: readonly DailyPoint[]): HeatwaveRun[] {
  const runs: HeatwaveRun[] = []
  let start = -1
  points.forEach((point, index) => {
    if (point.heatwave) {
      if (start < 0) start = index
    } else if (start >= 0) {
      runs.push({ start, end: index - 1 })
      start = -1
    }
  })
  if (start >= 0) runs.push({ start, end: points.length - 1 })
  return runs
}
