import { describe, expect, it } from 'vitest'

import { FEATURES, NAV_ITEMS, ROUTES } from '@/app/registry'

describe('feature registry', () => {
  it('resolves every nav item to a registered route', () => {
    const registered = new Set(ROUTES.map((route) => route.to))
    for (const item of NAV_ITEMS) {
      expect(registered.has(item.to), `nav ${item.to} has no route`).toBe(true)
    }
  })

  it('has no duplicate ids or route paths', () => {
    expect(new Set(FEATURES.map((feature) => feature.id)).size).toBe(FEATURES.length)
    expect(new Set(ROUTES.map((route) => route.to)).size).toBe(ROUTES.length)
  })

  it('has no duplicate nav targets and every feature owns a route', () => {
    expect(new Set(NAV_ITEMS.map((item) => item.to)).size).toBe(NAV_ITEMS.length)
    for (const feature of FEATURES) {
      expect(feature.routes.length, `feature ${feature.id} has no routes`).toBeGreaterThan(0)
    }
  })

  it('exposes every manifest route exactly once', () => {
    const expected = FEATURES.reduce((total, feature) => total + feature.routes.length, 0)
    expect(ROUTES).toHaveLength(expected)
  })
})
