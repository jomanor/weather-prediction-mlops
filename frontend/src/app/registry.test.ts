import { describe, expect, it } from 'vitest'

import { FEATURES, NAV_ITEMS, REDIRECTS, ROUTES } from '@/app/registry'

describe('feature registry', () => {
  it('exposes exactly the four surfaces, Panel last', () => {
    expect(NAV_ITEMS.map((item) => [item.label, item.to])).toEqual([
      ['Resumen', '/'],
      ['Estación', '/stations'],
      ['Sistema', '/sistema'],
      /* The user-composed dashboard sits after the three read surfaces: it is
         something you build, not something the backend serves. */
      ['Panel', '/panel'],
    ])
    expect(FEATURES).toHaveLength(4)
  })

  it('routes /panel to its own lazy page, in one list', () => {
    const panel = ROUTES.find((route) => route.to === '/panel')
    expect(panel?.id).toBe('dashboard')
    /* One list: the nav is the manifests' own items, never a parallel array. */
    expect(NAV_ITEMS).toHaveLength(FEATURES.filter((feature) => feature.navItem).length)
  })

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

  it('redirects the absorbed pages to the surfaces that kept their content', () => {
    expect(REDIRECTS.map((redirect) => [redirect.from, redirect.to])).toEqual([
      ['/benchmark', '/sistema?tab=benchmark'],
      ['/models', '/sistema?tab=modelos'],
      /* Analítica is gone (audit 16): the heatmap and the wind rose live on the
         station page now, so the legacy path lands there, not on a dead tab. */
      ['/analytics', '/stations'],
    ])
    /* A redirect must never shadow a live route or a nav target. */
    const live = new Set(ROUTES.map((route) => route.to))
    for (const redirect of REDIRECTS) expect(live.has(redirect.from)).toBe(false)
  })
})
