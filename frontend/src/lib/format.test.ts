import { describe, expect, it } from 'vitest'

import {
  compassPoint,
  convertTemperature,
  formatNumber,
  formatSigned,
  formatTemperature,
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
