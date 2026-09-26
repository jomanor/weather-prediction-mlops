import { render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CurrentWeather } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { CurrentConditions } from '@/features/station/CurrentConditions'

const NOW = new Date('2026-09-25T12:00:00Z').getTime()

function setReducedMotion(reduce: boolean) {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    () =>
      ({
        matches: reduce,
        media: '(prefers-reduced-motion: reduce)',
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }) as unknown as MediaQueryList,
  )
}

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
    observed_at: new Date(NOW - 5 * 60_000).toISOString(),
  }
}

function renderConditions() {
  return render(
    <PreferencesProvider>
      <CurrentConditions station={freshStation()} />
    </PreferencesProvider>,
  )
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(NOW)
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('CurrentConditions fresh-data pulse', () => {
  it('pulses a fresh observation when motion is allowed', () => {
    setReducedMotion(false)
    const { container } = renderConditions()
    expect(container.querySelector('.animate-ping')).not.toBeNull()
  })

  it('does not pulse under reduced motion', () => {
    setReducedMotion(true)
    const { container } = renderConditions()
    expect(container.querySelector('.animate-ping')).toBeNull()
  })
})
