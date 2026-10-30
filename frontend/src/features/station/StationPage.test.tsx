import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { currentWeatherSchema, predictionSchema } from '@/api/schemas'
import type { CurrentWeather, Prediction, WeatherPoint } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { StationPage } from '@/features/station/StationPage'
import { StationSelectionProvider } from '@/hooks/useStationSelection'

interface MockCurrent {
  data?: CurrentWeather
  isLoading: boolean
}
interface MockHistory {
  data?: { city: string; hours: number; points: WeatherPoint[] }
  isLoading: boolean
  isError: boolean
  error: Error | null
}

const mocks = vi.hoisted(() => ({
  current: { data: undefined, isLoading: true } as MockCurrent,
  history: { data: undefined, isLoading: true, isError: false, error: null } as MockHistory,
  predictions: { data: [] as Prediction[], isLoading: true },
}))

vi.mock('@/api/queries', () => ({
  useCities: () => ({
    data: [
      { name: 'Madrid', latitude: 40.42, longitude: -3.7 },
      { name: 'Sevilla', latitude: 37.39, longitude: -5.98 },
      { name: 'Pasadena', latitude: 34.15, longitude: -118.14 },
    ],
  }),
}))

vi.mock('@/features/station/queries', () => ({
  PREDICTION_HORIZONS: [1, 3, 6, 12, 24],
  useCurrentWeather: () => ({
    ...mocks.current,
    isError: false,
    error: null,
    refetch: vi.fn(),
    isFetching: false,
  }),
  useHistory: () => mocks.history,
  usePredictions: () => mocks.predictions,
}))

/* jsdom has no ResizeObserver; Recharts' ResponsiveContainer needs one. */
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver

function renderPage(search: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/${search}`]}>
        <PreferencesProvider>
          <StationSelectionProvider>
            <StationPage />
          </StationSelectionProvider>
        </PreferencesProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

function loadStationData() {
  const point: WeatherPoint = currentWeatherSchema.parse({
    city: 'Madrid',
    temperature: 22,
    humidity: 60,
    pressure: 1014,
    wind_speed: 10,
    wind_direction: 200,
    precipitation: 0.2,
    observed_at: '2026-09-25T10:00:00Z',
  })
  mocks.history = {
    data: { city: 'Madrid', hours: 48, points: [point] },
    isLoading: false,
    isError: false,
    error: null,
  }
  mocks.current = {
    data: currentWeatherSchema.parse({ ...point, apparent_temperature: 21.5, cloud_cover: 20 }),
    isLoading: false,
  }
  mocks.predictions = {
    data: [
      predictionSchema.parse({
        city: 'Madrid',
        source_timestamp: '2026-09-25T09:00:00Z',
        prediction_timestamp: '2026-09-25T10:00:00Z',
        horizon_hours: 1,
        predicted_temperature: 22.4,
        observed_temperature: 22,
        predicted_rain: 0.1,
        temp_lower: 21.2,
        temp_upper: 23.6,
        interval_level: 0.8,
        temp_model_name: 'temp_1h_gbt',
        temp_model_version: '20260925093000',
      }),
    ],
    isLoading: false,
  }
}

beforeEach(() => {
  mocks.current = { data: undefined, isLoading: true }
  mocks.history = { data: undefined, isLoading: true, isError: false, error: null }
  mocks.predictions = { data: [], isLoading: true }
})

describe('StationPage city validation', () => {
  it('falls back to the first station for an unknown URL city', () => {
    renderPage('?city=Unknown')
    expect(screen.getByLabelText('Estación')).toHaveValue('Madrid')
  })

  it('keeps a known URL city', () => {
    renderPage('?city=Sevilla')
    expect(screen.getByLabelText('Estación')).toHaveValue('Sevilla')
  })

  it('keeps the US station selectable: hiding it is a Sistema-only rule', () => {
    renderPage('?city=Pasadena')
    expect(screen.getByLabelText('Estación')).toHaveValue('Pasadena')
    const select = screen.getByLabelText('Estación')
    expect(within(select).getByRole('option', { name: 'Pasadena' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { level: 1, name: 'Pasadena' })).toBeInTheDocument()
  })
})

describe('StationPage content', () => {
  it('keeps the observed chart and the single prediction card with age and interval', () => {
    loadStationData()
    renderPage('?city=Madrid')

    expect(screen.getByRole('heading', { level: 1, name: 'Madrid' })).toBeInTheDocument()
    expect(screen.getByText('Evolución observada')).toBeInTheDocument()
    expect(screen.getByText('Predicción')).toBeInTheDocument()
    expect(screen.getByText('Intervalo')).toBeInTheDocument()
    expect(screen.getByText('Nivel')).toBeInTheDocument()
  })

  it('drops the derived-quantities panel', () => {
    loadStationData()
    renderPage('?city=Madrid')
    expect(screen.queryByText('Variables derivadas')).not.toBeInTheDocument()
    expect(screen.queryByText('Punto de rocío')).not.toBeInTheDocument()
  })

  it('keeps model lineage out of the station surface', () => {
    loadStationData()
    renderPage('?city=Madrid')
    expect(screen.queryByText(/temp_1h_gbt/)).not.toBeInTheDocument()
    expect(screen.queryByText(/20260925093000/)).not.toBeInTheDocument()
    expect(screen.queryByText('Últimas ejecuciones')).not.toBeInTheDocument()
    expect(screen.queryByText('Versión')).not.toBeInTheDocument()
  })
})
