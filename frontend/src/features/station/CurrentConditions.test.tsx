import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CurrentWeather } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { CurrentConditions } from '@/features/station/CurrentConditions'

const NOW = new Date('2026-09-25T12:00:00Z').getTime()

function freshStation(): CurrentWeather {
  return {
    city: 'Madrid',
    latitude: 40.4168,
    longitude: -3.7038,
    temperature: 22.0,
    apparent_temperature: 21.5,
    humidity: 60,
    pressure: 1014,
    wind_speed: 10,
    wind_direction: 200,
    precipitation: 0,
    cloud_cover: 30,
    weather_code: 2,
    is_day: 1,
    observed_at: new Date(NOW - 5 * 60_000).toISOString(),
  }
}

function renderConditions(station = freshStation()) {
  return render(
    <PreferencesProvider>
      <CurrentConditions station={station} />
    </PreferencesProvider>,
  )
}

/** `PreferencesProvider` reads `matchMedia`; `setup.ts` installs a `vi.fn` for
 * it, and a previous test's `restoreAllMocks()` can take it away again. */
function stubMatchMedia() {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    (query: string) =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }) as unknown as MediaQueryList,
  )
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(NOW)
  stubMatchMedia()
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('CurrentConditions weather descriptor', () => {
  it('renders the daytime descriptor for is_day 1', () => {
    const { container } = renderConditions()
    expect(screen.getByText('Parcialmente nuboso')).toBeInTheDocument()
    expect(container.querySelector('.lucide-cloud-sun')).not.toBeNull()
  })

  it('renders the moon glyph and night label when the station reports is_day 0', () => {
    /* The viewer's clock says 12:00; the station says it is night. The station wins. */
    const { container } = renderConditions({ ...freshStation(), is_day: 0, weather_code: 0 })
    expect(screen.getByText('Cielo despejado (noche)')).toBeInTheDocument()
    expect(container.querySelector('.lucide-moon')).not.toBeNull()
    expect(container.querySelector('.lucide-sun')).toBeNull()
  })
})

/**
 * S10: the hero used to pair the age with a pulsing ok/neutral `StatusDot`. The
 * dot was the same freshness fact told a second time *and* in colour only, so it
 * is gone; the age text is what survives. This pins the absence so the dot
 * cannot come back as "just a small indicator", and it pins that the mount
 * renders no animation of its own (nothing to gate on `prefers-reduced-motion`).
 */
describe('CurrentConditions freshness is one line of text', () => {
  it('reports the observation age as text and renders no status dot', () => {
    const { container } = renderConditions()
    /* Regex, not the literal: `Intl.RelativeTimeFormat` spells out "minutos"
       with full ICU and abbreviates under small ICU. */
    expect(screen.getByText(/hace 5 min/)).toBeInTheDocument()
    /* `StatusDot` is the only `h-1.5` box in the primitive set. */
    expect(container.querySelector('[class*="h-1.5"]')).toBeNull()
  })

  it('carries no animation of its own', () => {
    const { container } = renderConditions()
    expect(container.querySelector('.animate-ping')).toBeNull()
    expect(container.querySelector('.animate-spin')).toBeNull()
  })

  it('still prints the absolute timestamp next to the age', () => {
    renderConditions()
    expect(screen.getByText('Registrado')).toBeInTheDocument()
    /* Day + month + clock, whatever the runner's timezone: the fact is that the
       timestamp is still there and still text. */
    expect(screen.getByText(/^\d{2} \w+, \d{2}:\d{2}$/)).toBeInTheDocument()
  })
})
