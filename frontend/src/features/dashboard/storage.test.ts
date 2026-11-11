import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  DAY_OPTIONS,
  HOUR_OPTIONS,
  MAX_WIDGETS,
  WIDGET_TYPES,
  type DashboardWidget,
} from '@/features/dashboard/types'
import {
  DASHBOARD_KEY,
  DASHBOARD_VERSION,
  parseDashboard,
  readDashboard,
  writeDashboard,
} from '@/features/dashboard/storage'

/**
 * The dashboard layout is the only state the app keeps in the browser that the
 * user builds by hand, so the parser is the trust boundary: a stale blob, a
 * hand-edited type or a duplicated id must degrade to something renderable.
 */
function blob(widgets: unknown[], version: unknown = DASHBOARD_VERSION): string {
  return JSON.stringify({ version, widgets })
}

let nextId = 0
const makeId = () => `id-${(nextId += 1)}`

beforeEach(() => {
  localStorage.clear()
  nextId = 0
})

afterEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
})

describe('dashboard storage round-trip', () => {
  it('reads back exactly what it wrote, ids included', () => {
    const layout: DashboardWidget[] = [
      { id: 'a', type: 'diurnal', days: 30 },
      { id: 'b', type: 'wind-rose', city: 'Vigo', days: 180 },
      { id: 'c', type: 'series', city: 'Madrid', hours: 168 },
    ]

    writeDashboard(layout)

    expect(readDashboard()).toEqual(layout)
    /* The stored blob is the documented shape, version included. */
    expect(JSON.parse(localStorage.getItem(DASHBOARD_KEY) as string)).toEqual({
      version: 1,
      widgets: layout,
    })
  })

  it('keeps ids stable across a reload (parse twice, same ids)', () => {
    writeDashboard([{ id: 'keep-me', type: 'diurnal' }])
    const first = readDashboard()
    const second = readDashboard()

    expect(first.map((widget) => widget.id)).toEqual(['keep-me'])
    expect(second).toEqual(first)
  })

  it('reads an absent key as an empty dashboard', () => {
    expect(localStorage.getItem(DASHBOARD_KEY)).toBeNull()
    expect(readDashboard()).toEqual([])
  })
})

describe('dashboard storage is defensive', () => {
  it('returns an empty layout for malformed JSON instead of throwing', () => {
    localStorage.setItem(DASHBOARD_KEY, '{"version":1,"widgets":[')
    expect(() => readDashboard()).not.toThrow()
    expect(readDashboard()).toEqual([])
  })

  it.each([
    ['a JSON array root', '[]'],
    ['a JSON string root', '"nope"'],
    ['a null root', 'null'],
    ['a number root', '42'],
    ['a blob with no version', JSON.stringify({ widgets: [{ id: 'a', type: 'diurnal' }] })],
    ['a future version', blob([{ id: 'a', type: 'diurnal' }], 2)],
    ['a missing widgets array', JSON.stringify({ version: 1 })],
    ['a non-array widgets value', JSON.stringify({ version: 1, widgets: { a: 1 } })],
  ])('reads %s as an empty dashboard', (_label, raw) => {
    localStorage.setItem(DASHBOARD_KEY, raw)
    expect(readDashboard()).toEqual([])
  })

  it('drops an unknown widget type and keeps the known ones', () => {
    localStorage.setItem(
      DASHBOARD_KEY,
      blob([
        { id: 'a', type: 'radar' },
        { id: 'b', type: 'diurnal', days: 90 },
        { id: 'c' },
        { id: 'd', type: 7 },
        'not-a-widget',
        null,
      ]),
    )

    expect(readDashboard()).toEqual([{ id: 'b', type: 'diurnal', days: 90 }])
  })

  it('drops unknown keys and an option that is not one of the offered ones', () => {
    localStorage.setItem(
      DASHBOARD_KEY,
      blob([
        {
          id: 'a',
          type: 'wind-rose',
          city: '  Vigo  ',
          days: 45,
          hours: 12,
          bogus: 'x',
          onClick: 'alert(1)',
        },
      ]),
    )

    /* `bogus`/`onClick` never reach the rendered object, an invalid `days` falls
       back to the widget default and a valid city is trimmed. */
    expect(readDashboard()).toEqual([{ id: 'a', type: 'wind-rose', city: 'Vigo' }])
  })

  it('rejects a city that is not a non-empty string', () => {
    localStorage.setItem(
      DASHBOARD_KEY,
      blob([
        { id: 'a', type: 'series', city: '   ' },
        { id: 'b', type: 'series', city: 42 },
      ]),
    )

    expect(readDashboard()).toEqual([
      { id: 'a', type: 'series' },
      { id: 'b', type: 'series' },
    ])
  })

  it('normalises the blob on write, so junk cannot survive a save', () => {
    localStorage.setItem(DASHBOARD_KEY, blob([{ id: 'a', type: 'diurnal', junk: true }]))
    const layout = readDashboard()

    writeDashboard(layout)

    expect(JSON.parse(localStorage.getItem(DASHBOARD_KEY) as string)).toEqual({
      version: 1,
      widgets: [{ id: 'a', type: 'diurnal' }],
    })
  })
})

describe('dashboard storage ids', () => {
  it('generates an id for a widget that has none, and pins it on the next write', () => {
    localStorage.setItem(DASHBOARD_KEY, blob([{ type: 'diurnal' }]))

    const first = readDashboard()
    expect(first).toHaveLength(1)
    expect(first[0].id).not.toBe('')

    /* The page writes the hydrated layout straight back, which is what makes the
       generated id survive a reload (and the up/down focus stay put). */
    writeDashboard(first)
    expect(readDashboard()).toEqual(first)
  })

  it('replaces a duplicate id so React keys cannot collide', () => {
    localStorage.setItem(
      DASHBOARD_KEY,
      blob([
        { id: 'same', type: 'diurnal' },
        { id: 'same', type: 'series' },
      ]),
    )

    const layout = parseDashboard(localStorage.getItem(DASHBOARD_KEY), makeId)

    expect(layout.map((widget) => widget.id)).toEqual(['same', 'id-1'])
  })

  it('generates a distinct id per widget', () => {
    const layout = parseDashboard(blob(Array.from({ length: 5 }, () => ({ type: 'diurnal' }))))

    expect(layout).toHaveLength(5)
    expect(new Set(layout.map((widget) => widget.id)).size).toBe(5)
  })
})

describe('dashboard storage cap', () => {
  it(`keeps at most ${MAX_WIDGETS} widgets`, () => {
    const stored = Array.from({ length: 30 }, (_, index) => ({
      id: `w${index}`,
      type: WIDGET_TYPES[index % WIDGET_TYPES.length],
    }))
    localStorage.setItem(DASHBOARD_KEY, blob(stored))

    const layout = readDashboard()

    expect(layout).toHaveLength(MAX_WIDGETS)
    /* The cap truncates the tail; it never reshuffles what the user had. */
    expect(layout.map((widget) => widget.id)).toEqual(
      stored.slice(0, MAX_WIDGETS).map((widget) => widget.id),
    )
  })

  it('counts only valid widgets against the cap', () => {
    const mixed = [
      ...Array.from({ length: MAX_WIDGETS - 1 }, (_, index) => ({
        id: `ok${index}`,
        type: 'diurnal',
      })),
      ...Array.from({ length: 5 }, (_, index) => ({ id: `bad${index}`, type: 'radar' })),
      { id: 'last', type: 'series' },
    ]
    localStorage.setItem(DASHBOARD_KEY, blob(mixed))

    const layout = readDashboard()

    /* The dropped types must not eat a slot the user could fill. */
    expect(layout).toHaveLength(MAX_WIDGETS)
    expect(layout.at(-1)).toEqual({ id: 'last', type: 'series' })
  })
})

describe('dashboard storage never throws', () => {
  it('survives a storage that refuses to read', () => {
    vi.spyOn(localStorage, 'getItem').mockImplementation(() => {
      throw new Error('SecurityError')
    })

    expect(() => readDashboard()).not.toThrow()
    expect(readDashboard()).toEqual([])
  })

  it('survives a storage that refuses to write (quota)', () => {
    vi.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError')
    })

    expect(() => writeDashboard([{ id: 'a', type: 'diurnal' }])).not.toThrow()
  })
})

describe('dashboard offers', () => {
  it('accepts every documented window and no other number', () => {
    for (const days of DAY_OPTIONS) {
      expect(parseDashboard(blob([{ id: 'a', type: 'diurnal', days }]))).toEqual([
        { id: 'a', type: 'diurnal', days },
      ])
    }
    for (const hours of HOUR_OPTIONS) {
      expect(parseDashboard(blob([{ id: 'a', type: 'series', hours }]))).toEqual([
        { id: 'a', type: 'series', hours },
      ])
    }
    expect(parseDashboard(blob([{ id: 'a', type: 'diurnal', days: 7 }]))).toEqual([
      { id: 'a', type: 'diurnal' },
    ])
    expect(parseDashboard(blob([{ id: 'a', type: 'series', hours: 999 }]))).toEqual([
      { id: 'a', type: 'series' },
    ])
  })
})
