import { describe, expect, it } from 'vitest'

import {
  RAMP_DOMAINS,
  RAMP_STEPS,
  rampBreaks,
  rampColor,
  rampColorHex,
  rampHex,
  rampStep,
  rampToken,
} from '@/lib/ramps'

describe('rampStep', () => {
  it('quantises a value across the domain into 0..steps-1', () => {
    // temp: [-5, 40], 7 steps -> width 45/7
    expect(rampStep('temp', -5)).toBe(0)
    expect(rampStep('temp', 40)).toBe(6)
    expect(rampStep('temp', 0)).toBe(0) // 5/45 = 0.11 -> bin 0
    expect(rampStep('temp', 20)).toBe(3) // 25/45 = 0.55 -> bin 3
  })

  it('clamps out-of-domain and non-finite values instead of throwing', () => {
    expect(rampStep('rain', -10)).toBe(0)
    expect(rampStep('rain', 999)).toBe(RAMP_STEPS.rain - 1)
    expect(rampStep('wind', null)).toBe(0)
    expect(rampStep('wind', undefined)).toBe(0)
    expect(rampStep('wind', Number.NaN)).toBe(0)
    expect(rampStep('wind', Number.POSITIVE_INFINITY)).toBe(0)
  })

  it('guards a zero-width domain', () => {
    expect(rampStep('wind', 5, [5, 5])).toBe(0)
  })

  it('never returns a step outside the family size', () => {
    for (const name of ['temp', 'rain', 'wind'] as const) {
      const [min, max] = RAMP_DOMAINS[name]
      for (let i = 0; i <= 50; i += 1) {
        const value = min + ((max - min) * i) / 50
        const step = rampStep(name, value)
        expect(step).toBeGreaterThanOrEqual(0)
        expect(step).toBeLessThan(RAMP_STEPS[name])
      }
    }
  })
})

describe('rampBreaks', () => {
  it('returns the interior step thresholds', () => {
    const breaks = rampBreaks('rain', [0, 10])
    expect(breaks).toEqual([2, 4, 6, 8])
    expect(breaks).toHaveLength(RAMP_STEPS.rain - 1)
  })
})

describe('ramp token/hex resolution', () => {
  it('builds the matching CSS token', () => {
    expect(rampToken('temp', 0)).toBe('var(--temp-1)')
    expect(rampToken('temp', 6)).toBe('var(--temp-7)')
    expect(rampToken('rain', 2)).toBe('var(--rain-3)')
    expect(rampToken('wind', 4)).toBe('var(--wind-4)')
  })

  it('clamps out-of-range steps', () => {
    expect(rampToken('temp', -3)).toBe('var(--temp-1)')
    expect(rampToken('temp', 99)).toBe('var(--temp-7)')
    expect(rampHex('rain', 99)).toBe('#123f73')
  })

  it('falls back to the token literals when the sheet is unavailable', () => {
    // jsdom resolves no custom properties, so the fallback table is returned.
    expect(rampHex('temp', 0)).toBe('#0d0887')
    expect(rampHex('wind', 3)).toBe('#33445f')
    expect(rampColorHex('temp', 40)).toBe('#f0f921')
  })

  it('maps value + domain to a token and a hex', () => {
    expect(rampColor('rain', 10, [0, 10])).toBe('var(--rain-5)')
    expect(rampColorHex('wind', 0)).toBe('#aeb8c8')
  })
})
