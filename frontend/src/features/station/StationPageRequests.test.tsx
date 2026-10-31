import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { CurrentWeather, SeriesResponse } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { StationPage } from '@/features/station/StationPage'
import { STATION_SERIES_FIELDS } from '@/features/station/queries'
import { StationSelectionProvider } from '@/hooks/useStationSelection'

/**
 * The station surface must make ONE observation request: `GET /weather/series`.
 * The transport is mocked, not the hooks, so the real query hooks run and every
 * request the page would make is recorded — including a regression that brings
 * back `/weather/current/{city}` or `/weather/history/{city}`.
 */
const mocks = vi.hoisted(() => ({
  paths: [] as string[],
  series: null as unknown,
  seriesError: null as Error | null,
}))

const SERIES_FIELDS = [
  'temperature',
  'apparent_temperature',
  'humidity',
  'pressure',
  'wind_speed',
  'wind_direction',
  'precipitation',
  'cloud_cover',
  'weather_code',
  'is_day',
]

const CITIES = [
  { name: 'Madrid', latitude: 40.42, longitude: -3.7 },
  { name: 'El Ejido', latitude: 36.78, longitude: -2.81 },
]

function point(city: string, observedAt: string, temperature: number): CurrentWeather {
  return {
    city,
    latitude: 40.42,
    longitude: -3.7,
    temperature,
    apparent_temperature: temperature - 0.5,
    humidity: 60,
    pressure: 1014,
    wind_speed: 10,
    wind_direction: 200,
    precipitation: 0.2,
    cloud_cover: 20,
    weather_code: 0,
    is_day: 1,
    observed_at: observedAt,
  }
}

function defaultSeries(city: string, hours: number): SeriesResponse {
  const points = [
    point(city, '2026-09-25T08:00:00Z', 18),
    point(city, '2026-09-25T09:00:00Z', 20),
    point(city, '2026-09-25T10:00:00Z', 22),
  ]
  return {
    hours,
    count: 1,
    cities: [
      {
        city,
        count: points.length,
        latest_timestamp: '2026-09-25T10:00:00Z',
        points,
      },
    ],
  }
}

vi.mock('@/api/client', () => ({
  apiGet: vi.fn(async (path: string) => {
    mocks.paths.push(path)
    if (path.startsWith('/weather/series')) {
      if (mocks.seriesError) throw mocks.seriesError
      const search = new URLSearchParams(path.slice(path.indexOf('?') + 1))
      return mocks.series ?? defaultSeries(search.get('cities') ?? '', Number(search.get('hours')))
    }
    if (path === '/cities') return CITIES
    if (path.startsWith('/predictions/')) return []
    throw new Error(`unexpected request: ${path}`)
  }),
  apiGetWithDataAge: vi.fn(),
  apiSend: vi.fn(),
}))

/* jsdom has no ResizeObserver; the real Recharts chart is rendered here. */
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver

function renderPage(search = '?city=Madrid') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/stations${search}`]}>
        <PreferencesProvider>
          <StationSelectionProvider>
            <StationPage />
          </StationSelectionProvider>
        </PreferencesProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

const seriesCalls = () => mocks.paths.filter((path) => path.startsWith('/weather/series'))
const seriesQuery = (path: string) => new URLSearchParams(path.slice(path.indexOf('?') + 1))

beforeEach(() => {
  mocks.paths.length = 0
  mocks.series = null
  mocks.seriesError = null
})

describe('station observation request budget', () => {
  it('fires one /weather/series and never /weather/current or /weather/history', async () => {
    renderPage()

    /* Hero painted from the newest series point: the request has landed. */
    expect(await screen.findByText('22,0')).toBeInTheDocument()

    expect(seriesCalls()).toHaveLength(1)
    expect(mocks.paths.filter((path) => path.startsWith('/weather/current'))).toEqual([])
    expect(mocks.paths.filter((path) => path.startsWith('/weather/history'))).toEqual([])
    expect(mocks.paths).toEqual(['/cities', expect.stringMatching(/^\/weather\/series\?/), expect.stringMatching(/^\/predictions\//)])
  })

  it('asks for the selected window and projects only the rendered columns', async () => {
    renderPage()

    await waitFor(() => expect(seriesCalls()).toHaveLength(1))
    const query = seriesQuery(seriesCalls()[0])

    /* URL default window, and the backend's `step_hours` default (1 h) left
       alone: no downsampling, so the point count matches `/history`. */
    expect(query.get('hours')).toBe('48')
    expect(query.get('step_hours')).toBeNull()
    expect(query.get('cities')).toBe('Madrid')

    /* Pinned literally: the hero renders every column of the point contract. */
    expect(query.get('fields')).toBe(SERIES_FIELDS.join(','))
    expect(query.get('fields')).toBe(STATION_SERIES_FIELDS.join(','))
  })

  it('refetches the same endpoint with the new window when hours changes', async () => {
    renderPage()
    await waitFor(() => expect(seriesCalls()).toHaveLength(1))

    await userEvent.click(screen.getByRole('radio', { name: '24 h' }))

    await waitFor(() => expect(seriesCalls()).toHaveLength(2))
    expect(seriesQuery(seriesCalls().at(-1) ?? '').get('hours')).toBe('24')
    /* Still one endpoint, still no per-endpoint history/current fallback. */
    expect(mocks.paths.filter((path) => path.startsWith('/weather/history'))).toEqual([])
  })

  it('encodes a city name with a space in the cities parameter', async () => {
    renderPage()
    await waitFor(() => expect(seriesCalls()).toHaveLength(1))

    /* The registry request and the series race; wait for the option itself. */
    await screen.findByRole('option', { name: 'El Ejido' })
    await userEvent.selectOptions(screen.getByLabelText('Estación'), 'El Ejido')

    await waitFor(() =>
      expect(seriesQuery(seriesCalls().at(-1) ?? '').get('cities')).toBe('El Ejido'),
    )
    expect(seriesCalls().at(-1)).toContain('cities=El%20Ejido')
  })
})

describe('station series empty and failure states', () => {
  it('renders a city with no rows as an empty state', async () => {
    mocks.series = {
      hours: 48,
      count: 1,
      cities: [{ city: 'Madrid', count: 0, latest_timestamp: null, points: [] }],
    }
    renderPage()

    expect(await screen.findByText('Sin observaciones recientes')).toBeInTheDocument()
    /* `/series` has no per-city 404, so nothing was retried through the old
       endpoints to "find" the observation. */
    expect(mocks.paths.filter((path) => path.startsWith('/weather/current'))).toEqual([])
    expect(mocks.paths.filter((path) => path.startsWith('/weather/history'))).toEqual([])
  })

  it('surfaces a failed series as an error with a retry, without falling back', async () => {
    mocks.seriesError = new Error('La API respondió 404')
    renderPage()

    /* `useApiQuery` retries once, so the failure state follows the second try. */
    expect(await screen.findByText('No se pudo cargar la observación', {}, { timeout: 3000 }))
      .toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reintentar' })).toBeInTheDocument()
    expect(mocks.paths.filter((path) => path.startsWith('/weather/current'))).toEqual([])
    expect(mocks.paths.filter((path) => path.startsWith('/weather/history'))).toEqual([])
  })
})
