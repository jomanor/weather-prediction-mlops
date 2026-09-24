import { describe, expect, it } from 'vitest'

import { dewPoint, saturationVapourPressure, specificHumidity, windComponents } from '@/lib/derived'

describe('saturationVapourPressure', () => {
  it('matches the Bolton reference value at 0 °C', () => {
    expect(saturationVapourPressure(0)).toBeCloseTo(6.112, 3)
  })

  it('grows with temperature', () => {
    expect(saturationVapourPressure(30)).toBeGreaterThan(saturationVapourPressure(10))
  })
})

describe('specificHumidity', () => {
  it('derives ~7.2 g/kg at 20 °C, 50 % RH, 1013.25 hPa', () => {
    expect(specificHumidity(20, 50, 1013.25)).toBeCloseTo(7.21, 1)
  })

  it('rejects non-physical pressure', () => {
    expect(specificHumidity(20, 50, 0)).toBeNull()
  })
})

describe('dewPoint', () => {
  it('is ~9.3 °C at 20 °C and 50 % RH', () => {
    expect(dewPoint(20, 50)).toBeCloseTo(9.25, 1)
  })

  it('equals the dry-bulb temperature at saturation', () => {
    expect(dewPoint(15, 100)).toBeCloseTo(15, 5)
  })

  it('rejects zero humidity', () => {
    expect(dewPoint(15, 0)).toBeNull()
  })
})

describe('windComponents', () => {
  it('maps a 270° wind to a purely zonal eastward vector', () => {
    const { u, v } = windComponents(36, 270)
    expect(u).toBeCloseTo(10, 5)
    expect(v).toBeCloseTo(0, 5)
  })

  it('maps a 180° wind to a purely meridional southward vector', () => {
    const { u, v } = windComponents(36, 180)
    expect(u).toBeCloseTo(0, 5)
    expect(v).toBeCloseTo(10, 5)
  })
})
