import { describe, expect, it } from 'vitest'

import type { DailyPoint } from '@/features/analytics/schemas'
import {
  ANALYTICS_BOUNDS,
  buildDiurnalMatrix,
  buildWindRose,
  clampDays,
  heatwaveRuns,
} from '@/features/analytics/transforms'

function dailyPoint(date: string, heatwave: boolean): DailyPoint {
  return {
    date,
    tmin: null,
    tmax: null,
    tmean: null,
    hdd: null,
    cdd: null,
    anomaly: null,
    heatwave,
  }
}

describe('clampDays', () => {
  it('clamps to the endpoint bound', () => {
    expect(clampDays(1, ANALYTICS_BOUNDS.daily)).toBe(7)
    expect(clampDays(400, ANALYTICS_BOUNDS.daily)).toBe(180)
    expect(clampDays(90, ANALYTICS_BOUNDS.daily)).toBe(90)
  })

  it('falls back to the default for non-finite input and truncates fractions', () => {
    expect(clampDays(Number.NaN, ANALYTICS_BOUNDS.daily)).toBe(90)
    expect(clampDays(45.9, ANALYTICS_BOUNDS.errorByHour)).toBe(45)
  })

  it('keeps the error-by-hour window inside its tighter bound', () => {
    expect(clampDays(365, ANALYTICS_BOUNDS.errorByHour)).toBe(90)
  })
})

describe('buildWindRose', () => {
  it('always yields 16 ordered petals with compass labels', () => {
    const model = buildWindRose([{ sector: 0, count: 10, mean_speed: 20 }])
    expect(model.petals).toHaveLength(16)
    expect(model.petals.map((petal) => petal.sector)).toEqual(
      Array.from({ length: 16 }, (_, sector) => sector),
    )
    expect(model.petals[0].label).toBe('N')
    expect(model.petals[4].label).toBe('E')
    expect(model.petals[8].label).toBe('S')
    expect(model.petals[12].label).toBe('O')
  })

  it('aggregates duplicate sectors and count-weights the mean speed', () => {
    const model = buildWindRose([
      { sector: 0, count: 10, mean_speed: 20 },
      { sector: 0, count: 5, mean_speed: 50 },
    ])
    expect(model.petals[0].count).toBe(15)
    expect(model.petals[0].meanSpeed).toBeCloseTo((20 * 10 + 50 * 5) / 15)
    expect(model.total).toBe(15)
    expect(model.maxCount).toBe(15)
  })

  it('defers to the API mean speed when a speed is null', () => {
    const model = buildWindRose([{ sector: 8, count: 4, mean_speed: null }])
    expect(model.petals[8].count).toBe(4)
    expect(model.petals[8].meanSpeed).toBeNull()
    expect(model.maxSpeed).toBeNull()
  })

  it('wraps out-of-range sectors and clamps negative counts', () => {
    const model = buildWindRose([
      { sector: 16, count: 3, mean_speed: 10 },
      { sector: 17, count: -5, mean_speed: 10 },
      { sector: -1, count: 2, mean_speed: 10 },
    ])
    expect(model.petals[0].count).toBe(3)
    expect(model.petals[1].count).toBe(0)
    expect(model.petals[15].count).toBe(2)
    expect(model.total).toBe(5)
  })
})

describe('buildDiurnalMatrix', () => {
  it('pivots cities × 24 hours and leaves missing hours null', () => {
    const matrix = buildDiurnalMatrix([
      { city: 'Madrid', hour: 0, tmean: 12, n: 3 },
      { city: 'Madrid', hour: 13, tmean: 28, n: 4 },
      { city: 'Vigo', hour: 13, tmean: null, n: 0 },
    ])
    expect(matrix.cities).toEqual(['Madrid', 'Vigo'])
    expect(matrix.rows[0].values).toHaveLength(24)
    expect(matrix.rows[0].values[0]).toBe(12)
    expect(matrix.rows[0].values[1]).toBeNull()
    expect(matrix.rows[0].counts[13]).toBe(4)
    expect(matrix.rows[1].values[13]).toBeNull()
    expect(matrix.min).toBe(12)
    expect(matrix.max).toBe(28)
  })

  it('ignores out-of-range hours without dropping the city', () => {
    const matrix = buildDiurnalMatrix([
      { city: 'Madrid', hour: 24, tmean: 30, n: 1 },
      { city: 'Madrid', hour: -1, tmean: 30, n: 1 },
    ])
    expect(matrix.cities).toEqual(['Madrid'])
    expect(matrix.rows[0].values.filter((value) => value !== null)).toEqual([])
  })
})

describe('heatwaveRuns', () => {
  it('returns inclusive runs, including an open-ended final run', () => {
    const runs = heatwaveRuns([
      dailyPoint('2026-07-01', false),
      dailyPoint('2026-07-02', true),
      dailyPoint('2026-07-03', true),
      dailyPoint('2026-07-04', true),
      dailyPoint('2026-07-05', false),
      dailyPoint('2026-07-06', true),
    ])
    expect(runs).toEqual([
      { start: 1, end: 3 },
      { start: 5, end: 5 },
    ])
  })

  it('returns nothing without a heatwave', () => {
    expect(heatwaveRuns([dailyPoint('2026-07-01', false)])).toEqual([])
  })
})
