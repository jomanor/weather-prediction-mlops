import { describe, expect, it } from 'vitest'

import type { City } from '@/api/schemas'
import {
  aemetCoveredCities,
  aemetCoveredNames,
  isAemetCovered,
  isOutsideAemetCoverage,
} from '@/lib/aemet-coverage'

const REGISTRY: City[] = [
  { name: 'Madrid', latitude: 40.42, longitude: -3.7 },
  { name: 'Vigo', latitude: 42.24, longitude: -8.72 },
  { name: 'Pasadena', latitude: 34.15, longitude: -118.14 },
]

describe('AEMET coverage rule', () => {
  it('treats longitudes west of -20° as outside coverage', () => {
    expect(isOutsideAemetCoverage(REGISTRY[2])).toBe(true)
    expect(isAemetCovered(REGISTRY[2])).toBe(false)
    expect(isOutsideAemetCoverage(REGISTRY[0])).toBe(false)
    expect(isAemetCovered(REGISTRY[0])).toBe(true)
  })

  it('keeps an unknown station covered rather than dropping it', () => {
    expect(isOutsideAemetCoverage(undefined)).toBe(false)
    expect(isAemetCovered(undefined)).toBe(true)
  })
})

describe('covered registry helpers', () => {
  it('filters the registry to covered stations, in order', () => {
    expect(aemetCoveredCities(REGISTRY).map((city) => city.name)).toEqual(['Madrid', 'Vigo'])
  })

  it('builds the name set that coordinate-less endpoints are matched against', () => {
    const names = aemetCoveredNames(REGISTRY)
    expect(names.has('Madrid')).toBe(true)
    expect(names.has('Pasadena')).toBe(false)
    expect(names.size).toBe(2)
  })
})
