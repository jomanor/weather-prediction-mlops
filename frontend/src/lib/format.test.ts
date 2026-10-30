import { describe, expect, it } from 'vitest'

import {
  ageInHours,
  compassPoint,
  convertTemperature,
  formatAgeHours,
  formatDegrees,
  formatNumber,
  formatSigned,
  formatTemperature,
  formatWithUnit,
  isNum,
  normalise,
} from '@/lib/format'

describe('isNum', () => {
  it('accepts finite numbers only', () => {
    expect(isNum(0)).toBe(true)
    expect(isNum(-1.5)).toBe(true)
    expect(isNum(null)).toBe(false)
    expect(isNum(undefined)).toBe(false)
    expect(isNum(Number.NaN)).toBe(false)
    expect(isNum(Number.POSITIVE_INFINITY)).toBe(false)
  })
})

describe('formatTemperature', () => {
  it('returns the empty placeholder for missing values', () => {
    expect(formatTemperature(null)).toBe('—')
    expect(formatTemperature(undefined)).toBe('—')
  })

  it('formats celsius with the es-ES decimal separator', () => {
    expect(formatTemperature(21.35)).toBe('21,4°')
  })

  it('converts to fahrenheit', () => {
    expect(formatTemperature(20, 'imperial')).toBe('68,0°')
  })
})

describe('convertTemperature', () => {
  it('keeps celsius in metric', () => {
    expect(convertTemperature(20, 'metric')).toBe(20)
  })
})

describe('compassPoint', () => {
  it('maps cardinal and ordinal directions', () => {
    expect(compassPoint(0)).toBe('N')
    expect(compassPoint(90)).toBe('E')
    expect(compassPoint(180)).toBe('S')
    expect(compassPoint(225)).toBe('SO')
    expect(compassPoint(359)).toBe('N')
  })

  it('returns the placeholder without a direction', () => {
    expect(compassPoint(null)).toBe('—')
  })
})

describe('formatSigned', () => {
  it('prefixes an explicit sign', () => {
    expect(formatSigned(1.234)).toBe('+1,23')
    expect(formatSigned(-1.234)).toBe('−1,23')
    expect(formatSigned(0)).toBe('0,00')
  })
})

describe('formatNumber', () => {
  it('uses the requested precision', () => {
    expect(formatNumber(3.14159, 3)).toBe('3,142')
    expect(formatNumber(null)).toBe('—')
  })
})

describe('normalise', () => {
  it('clamps into 0..1', () => {
    expect(normalise(5, 0, 10)).toBe(0.5)
    expect(normalise(-1, 0, 10)).toBe(0)
    expect(normalise(99, 0, 10)).toBe(1)
  })

  it('handles a zero-width range', () => {
    expect(normalise(5, 5, 5)).toBe(0.5)
  })
})


describe('chart formatting helpers', () => {
  it('formats an already-converted degree value', () => {
    expect(formatDegrees(21.4, 0)).toBe('21°')
    expect(formatDegrees(null)).toBe('—')
  })

  it('appends the unit with a space', () => {
    expect(formatWithUnit(21.44, '°C', 1)).toBe('21,4 °C')
    expect(formatWithUnit(2, 'mm', 1)).toBe('2,0 mm')
  })

  it('derives a human age from an ISO timestamp', () => {
    const now = new Date('2026-09-25T12:00:00Z').getTime()
    expect(ageInHours('2026-09-25T09:00:00Z', now)).toBe(3)
    expect(formatAgeHours('2026-09-25T09:00:00Z', now)).toBe('hace 3 h')
    expect(formatAgeHours('2026-09-25T11:30:00Z', now)).toBe('hace 30 min')
    expect(formatAgeHours('nope', now)).toBe('—')
  })
})
