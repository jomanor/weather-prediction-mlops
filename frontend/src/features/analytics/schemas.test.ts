import { describe, expect, it } from 'vitest'

import {
  analyticsDailySchema,
  climatologySchema,
  correlationSchema,
  diurnalSchema,
  errorByHourSchema,
  windRoseSchema,
} from '@/features/analytics/schemas'

describe('analyticsDailySchema (/analytics/daily)', () => {
  it('parses the frozen contract payload', () => {
    const parsed = analyticsDailySchema.parse({
      city: 'Madrid',
      days: 90,
      generated_at: '2026-09-25T12:00:00Z',
      points: [
        {
          date: '2026-07-01',
          tmin: 20.1,
          tmax: 36.4,
          tmean: 28.2,
          hdd: 0,
          cdd: 10.2,
          anomaly: 3.4,
          heatwave: true,
        },
      ],
    })
    expect(parsed.points[0].heatwave).toBe(true)
    expect(parsed.points[0].cdd).toBeCloseTo(10.2)
  })

  it('coerces missing nullable fields and defaults heatwave to false', () => {
    const parsed = analyticsDailySchema.parse({
      city: 'Vigo',
      days: 90,
      points: [{ date: '2026-09-01' }],
    })
    expect(parsed.generated_at).toBeNull()
    expect(parsed.points[0].tmean).toBeNull()
    expect(parsed.points[0].hdd).toBeNull()
    expect(parsed.points[0].anomaly).toBeNull()
    expect(parsed.points[0].heatwave).toBe(false)
  })

  it('rejects a non-boolean heatwave marker', () => {
    const result = analyticsDailySchema.safeParse({
      city: 'Madrid',
      days: 90,
      points: [{ date: '2026-07-01', heatwave: 'yes' }],
    })
    expect(result.success).toBe(false)
  })
})

describe('climatologySchema (/analytics/climatology)', () => {
  it('parses a daily baseline with a null-mean day', () => {
    const parsed = climatologySchema.parse({
      city: 'Sevilla',
      generated_at: '2026-09-25T12:00:00Z',
      basis_years: 12.5,
      series: [
        { day_of_year: 1, tmean: 11.2, tmin: 6.1, tmax: 16.3, n: 12 },
        { day_of_year: 366, tmean: null, tmin: null, tmax: null, n: 0 },
      ],
    })
    expect(parsed.basis_years).toBeCloseTo(12.5)
    expect(parsed.series[1].tmean).toBeNull()
    expect(parsed.series[1].n).toBe(0)
  })
})

describe('windRoseSchema (/analytics/wind-rose)', () => {
  it('keeps a null mean speed instead of inventing one', () => {
    const parsed = windRoseSchema.parse({
      city: 'Madrid',
      days: 90,
      generated_at: '2026-09-25T12:00:00Z',
      sectors: [
        { sector: 0, count: 12, mean_speed: 18.4 },
        { sector: 8, count: 3, mean_speed: null },
      ],
    })
    expect(parsed.sectors[0].mean_speed).toBeCloseTo(18.4)
    expect(parsed.sectors[1].mean_speed).toBeNull()
  })
})

describe('diurnalSchema (/analytics/diurnal)', () => {
  it('defaults the sample count and tolerates a missing temperature', () => {
    const parsed = diurnalSchema.parse({
      days: 90,
      cells: [{ city: 'Madrid', hour: 13 }],
    })
    expect(parsed.generated_at).toBeNull()
    expect(parsed.cells[0].tmean).toBeNull()
    expect(parsed.cells[0].n).toBe(0)
  })
})

describe('correlationSchema (/analytics/correlation)', () => {
  it('parses a pairwise matrix with null pairs', () => {
    const parsed = correlationSchema.parse({
      days: 90,
      var: 'temperature',
      cities: ['Madrid', 'Vigo'],
      matrix: [
        [1, null],
        [null, 1],
      ],
    })
    expect(parsed.matrix[0][0]).toBe(1)
    expect(parsed.matrix[0][1]).toBeNull()
  })
})

describe('errorByHourSchema (/analytics/error-by-hour)', () => {
  it('parses a point with a persistence baseline', () => {
    const parsed = errorByHourSchema.parse({
      days: 30,
      points: [
        {
          horizon_hours: 12,
          hour: 15,
          n: 40,
          mae: 1.8,
          rmse: 2.4,
          bias: -0.2,
          persistence_mae: 2.6,
        },
      ],
    })
    expect(parsed.points[0].persistence_mae).toBeCloseTo(2.6)
    expect(parsed.points[0].n).toBe(40)
  })

  it('coerces missing metric fields to null', () => {
    const parsed = errorByHourSchema.parse({ days: 30, points: [{ horizon_hours: 1, hour: 0 }] })
    expect(parsed.points[0].mae).toBeNull()
    expect(parsed.points[0].persistence_mae).toBeNull()
  })
})
