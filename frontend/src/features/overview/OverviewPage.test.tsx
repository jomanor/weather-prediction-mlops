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
const mocks = vi.hoisted(() => ({ paths: [] as string[] }))

const NOW = Date.now()

function station(city: string, minutesAgo: number) {
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
    observed_at: new Date(NOW - minutesAgo * 60_000).toISOString(),
  }
}

vi.mock('@/api/client', () => ({
  apiGet: vi.fn(async (path: string) => {
    mocks.paths.push(path.split('?')[0])
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
        stations: [station('Madrid', 5), station('Sevilla', 65), station('Pasadena', 5)],
      }
    }
    if (path === '/map/stations') return null
    throw new Error(`unexpected request: ${path}`)
  }),
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
