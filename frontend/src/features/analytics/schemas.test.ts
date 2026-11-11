import { describe, expect, it } from 'vitest'

import { diurnalSchema, windRoseSchema } from '@/features/analytics/schemas'

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

