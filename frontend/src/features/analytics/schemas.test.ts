import { describe, expect, it } from 'vitest'

import {
  analyticsDailySchema,
  diurnalSchema,
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

