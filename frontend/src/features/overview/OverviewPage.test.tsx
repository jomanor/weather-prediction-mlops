import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { PreferencesProvider } from '@/app/preferences'
import { OverviewPage } from '@/features/overview/OverviewPage'
import { StationSelectionProvider } from '@/hooks/useStationSelection'

/**
 * The Resumen mount budget is the point of this test. The transport is mocked,
 * not the hooks, so the real query hooks run and every request the page would
 * make is recorded: the list must be exactly `/cities`, `/weather/current` and
 * `/map/stations` — no `usePredictions`, no `/benchmark`, no `/weather/quality`.
 */
const mocks = vi.hoisted(() => ({
  paths: [] as string[],
  /* `X-Data-Age-Seconds` the fake transport reports; null = header absent. */
  dataAgeSeconds: null as number | null,
  /* When set, every station's `observed_at` is this many minutes old. */
  stationMinutesAgo: null as number | null,
  /* When set, `/weather/current` carries no stations at all. */
  emptyStations: false,
}))

const NOW = Date.now()

function station(city: string, minutesAgo: number) {
  const age = mocks.stationMinutesAgo ?? minutesAgo
  return {
    city,
    latitude: 40.42,
    longitude: -3.7,
    temperature: 21.5,
    apparent_temperature: 21,
    humidity: 40,
    pressure: 1015,
    wind_speed: 10,
    wind_direction: 180,
    precipitation: 0,
    cloud_cover: 10,
    weather_code: 0,
    is_day: 1,
    observed_at: new Date(NOW - age * 60_000).toISOString(),
  }
}

/** Payloads behind both transports; the path list records every request once. */
function payloadFor(path: string) {
  if (path === '/cities') {
    return [
      { name: 'Madrid', latitude: 40.42, longitude: -3.7 },
      { name: 'Sevilla', latitude: 37.39, longitude: -5.98 },
      /* The US station stays on Resumen; only Sistema filters it out. */
      { name: 'Pasadena', latitude: 34.15, longitude: -118.14 },
    ]
  }
  if (path === '/weather/current') {
    return {
      generated_at: new Date(NOW).toISOString(),
      stations: mocks.emptyStations
        ? []
        : [station('Madrid', 5), station('Sevilla', 65), station('Pasadena', 5)],
    }
  }
  if (path === '/map/stations') return null
  throw new Error(`unexpected request: ${path}`)
}

vi.mock('@/api/client', () => ({
  apiGet: vi.fn(async (path: string) => {
    mocks.paths.push(path.split('?')[0])
    return payloadFor(path)
  }),
  apiGetWithDataAge: vi.fn(async (path: string) => ({
    data: (mocks.paths.push(path.split('?')[0]), payloadFor(path)),
    dataAgeSeconds: mocks.dataAgeSeconds,
  })),
  apiSend: vi.fn(),
}))

/* The interactive map is MapLibre; the assertion here is the wiring, so the
   mock renders the pin as a button that calls the same `onSelect`. */
vi.mock('@/components/map/StationMap', () => ({
  StationMap: ({ onSelect }: { onSelect: (city: string) => void }) => (
    <button type="button" onClick={() => onSelect('Sevilla')}>
      pin Sevilla
    </button>
  ),
}))

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname + location.search}</output>
}

function renderOverview() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/']}>
        <PreferencesProvider>
          <StationSelectionProvider>
            <OverviewPage />
            <LocationProbe />
          </StationSelectionProvider>
        </PreferencesProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  mocks.paths.length = 0
  mocks.dataAgeSeconds = null
  mocks.stationMinutesAgo = null
  mocks.emptyStations = false
})

describe('Resumen mount budget', () => {
  it('fires only /cities, /weather/current and /map/stations', () => {
    renderOverview()
    expect(mocks.paths).toEqual(['/cities', '/weather/current', '/map/stations'])
    expect(mocks.paths).not.toContain('/predictions/latest')
    expect(mocks.paths).not.toContain('/benchmark')
    expect(mocks.paths).not.toContain('/weather/quality')
  })
})

describe('Resumen content', () => {
  it('shows one freshness line for the observed network', async () => {
    renderOverview()
    expect(await screen.findByText(/3 estaciones · actualizado a las \d{2}:\d{2}/)).toBeInTheDocument()
    /* Fresh data, so no stale suffix. */
    expect(screen.queryByText(/datos con más de 12 h/)).not.toBeInTheDocument()
  })

  it('keeps Pasadena in the station list and its link', async () => {
    renderOverview()
    const link = await screen.findByRole('link', { name: 'Pasadena' })
    expect(link).toHaveAttribute('href', '/stations?city=Pasadena')
  })

  it('does not render the data-quality meter, a city manager or the MAE', () => {
    renderOverview()
    expect(screen.queryByText('Calidad de datos')).not.toBeInTheDocument()
    expect(screen.queryByText('Añadir estación')).not.toBeInTheDocument()
    expect(screen.queryByText(/MAE/)).not.toBeInTheDocument()
  })

  it('opens the station page when a map pin is selected', async () => {
    renderOverview()
    await userEvent.click(await screen.findByRole('button', { name: 'pin Sevilla' }))
    expect(screen.getByTestId('location')).toHaveTextContent('/stations?city=Sevilla')
  })

  it('opens the station page from the compact list', async () => {
    renderOverview()
    expect(await screen.findByRole('link', { name: 'Sevilla' })).toHaveAttribute(
      'href',
      '/stations?city=Sevilla',
    )
    await userEvent.click(screen.getByRole('link', { name: 'Sevilla' }))
    expect(screen.getByTestId('location')).toHaveTextContent('/stations?city=Sevilla')
  })
})

/**
 * F1/F6: `X-Data-Age-Seconds` is a floor, never live authority. `/weather/current`
 * is cacheable, so a 304 revalidation replays the header stored at the last 200
 * and React Query resets the baseline: trusting the header alone froze the age
 * while a stalled pipeline kept the payload identical. The rendered age is now
 * `max(carried header, observed_at math)`, and the '>12 h' suffix keys on that
 * result rather than on the presence of a payload timestamp.
 */
describe('Resumen freshness (server age can only raise the estimate)', () => {
  it('keeps the payload age when a replayed header would look fresher', async () => {
    /* Stalled pipeline: payload frozen for 20 h, header replayed at 5 min on
       every 304. The old header-wins rule painted this fresh; max() does not. */
    mocks.stationMinutesAgo = 20 * 60
    mocks.dataAgeSeconds = 300
    renderOverview()
    expect(await screen.findByText(/3 estaciones/)).toBeInTheDocument()
    expect(screen.getByText(/datos con más de 12 h/)).toBeInTheDocument()
  })

  it('raises a fresh-looking payload to the server age when that is larger', async () => {
    /* Client clock: 5 min old. Server: 13 h, past the window. */
    mocks.dataAgeSeconds = 13 * 60 * 60
    renderOverview()
    expect(await screen.findByText(/datos con más de 12 h/)).toBeInTheDocument()
  })

  it('falls back to the payload timestamps when the header is absent', async () => {
    mocks.stationMinutesAgo = 20 * 60
    mocks.dataAgeSeconds = null
    renderOverview()
    expect(await screen.findByText(/datos con más de 12 h/)).toBeInTheDocument()
  })

  it('shows the stale line from the server age alone', async () => {
    /* F6: with no stations there is no payload timestamp, so `newest` is null
       while the age exists. The suffix must still render. */
    mocks.emptyStations = true
    mocks.dataAgeSeconds = 13 * 60 * 60
    renderOverview()
    /* `0 estaciones` also renders while loading, so wait for the age itself. */
    expect(await screen.findByText(/datos con más de 12 h/)).toBeInTheDocument()
    expect(screen.getByText(/0 estaciones/)).toBeInTheDocument()
  })

  it('shows no stale line while both sources agree it is fresh', async () => {
    mocks.dataAgeSeconds = 300
    renderOverview()
    expect(await screen.findByText(/3 estaciones/)).toBeInTheDocument()
    expect(screen.queryByText(/datos con más de 12 h/)).not.toBeInTheDocument()
  })
})
