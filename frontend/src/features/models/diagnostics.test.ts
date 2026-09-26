import { describe, expect, it } from 'vitest'

import { driftLabel, driftTone, maxDriftPsi } from '@/features/models/diagnostics'

describe('maxDriftPsi', () => {
  it('returns null when no diagnostics or no finite PSI is present', () => {
    expect(maxDriftPsi(null)).toBeNull()
    expect(maxDriftPsi(undefined)).toBeNull()
    expect(maxDriftPsi({ by_city: [], by_hour_of_day: [], by_rain_bucket: [], drift_psi: {} })).toBeNull()
    expect(
      maxDriftPsi({
        by_city: [],
        by_hour_of_day: [],
        by_rain_bucket: [],
        drift_psi: { temperature: null, humidity: null },
      }),
    ).toBeNull()
  })

  it('takes the maximum, ignoring nulls', () => {
    expect(
      maxDriftPsi({
        by_city: [],
        by_hour_of_day: [],
        by_rain_bucket: [],
        drift_psi: { temperature: 0.08, humidity: 0.31, pressure: null },
      }),
    ).toBe(0.31)
  })
})

describe('driftTone / driftLabel (Contract 3 thresholds)', () => {
  it('is stable below 0.1', () => {
    expect(driftTone(0)).toBe('ok')
    expect(driftTone(0.099)).toBe('ok')
    expect(driftLabel(0.099)).toBe('Estable')
  })

  it('warns in [0.1, 0.25] inclusive', () => {
    expect(driftTone(0.1)).toBe('warn')
    expect(driftTone(0.25)).toBe('warn')
    expect(driftLabel(0.25)).toBe('Aviso')
  })

  it('flags drift above 0.25', () => {
    expect(driftTone(0.2501)).toBe('bad')
    expect(driftLabel(1)).toBe('Deriva')
  })

  it('has no label for a missing value', () => {
    expect(driftTone(null)).toBe('neutral')
    expect(driftLabel(null)).toBeNull()
  })
})
