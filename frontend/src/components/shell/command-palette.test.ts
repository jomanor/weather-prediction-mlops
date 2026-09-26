import { describe, expect, it } from 'vitest'

import { filterPaletteItems, normalizeTerm, type PaletteItem } from '@/components/shell/command-palette'

const ITEMS: PaletteItem[] = [
  { id: 'route:/', label: 'Red', group: 'Secciones', to: '/' },
  { id: 'route:/stations', label: 'Estación', group: 'Secciones', to: '/stations' },
  { id: 'route:/models', label: 'Modelos', group: 'Secciones', to: '/models' },
  { id: 'station:Sevilla', label: 'Sevilla', group: 'Estaciones', to: '/stations?city=Sevilla' },
  { id: 'station:Santander', label: 'Santander', group: 'Estaciones', to: '/stations?city=Santander' },
  { id: 'station:Madrid', label: 'Madrid', group: 'Estaciones', to: '/stations?city=Madrid' },
]

describe('normalizeTerm', () => {
  it('strips case and accents', () => {
    expect(normalizeTerm('  Estación ')).toBe('estacion')
    expect(normalizeTerm('MÁLAGA')).toBe('malaga')
  })
})

describe('filterPaletteItems', () => {
  it('lists sections and a few stations for an empty query', () => {
    const result = filterPaletteItems(ITEMS, '')
    expect(result.map((item) => item.id)).toEqual([
      'route:/',
      'route:/stations',
      'route:/models',
      'station:Sevilla',
      'station:Santander',
      'station:Madrid',
    ])
  })

  it('matches accents and case-insensitively', () => {
    const result = filterPaletteItems(ITEMS, 'estacion')
    expect(result.map((item) => item.label)).toEqual(['Estación'])
  })

  it('ranks prefix matches before substring matches', () => {
    const result = filterPaletteItems(ITEMS, 'sa')
    expect(result.map((item) => item.label)).toEqual(['Santander'])
  })

  it('filters by station name and respects the limit', () => {
    const result = filterPaletteItems(ITEMS, 's', 2)
    expect(result.length).toBe(2)
    expect(result.every((item) => normalizeTerm(item.label).includes('s'))).toBe(true)
  })

  it('returns nothing when there is no match', () => {
    expect(filterPaletteItems(ITEMS, 'zzz')).toEqual([])
  })
})
