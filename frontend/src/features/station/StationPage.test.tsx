import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { currentWeatherSchema, predictionSchema } from '@/api/schemas'
import type { CurrentWeather, Prediction, SeriesResponse, WeatherPoint } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { StationPage } from '@/features/station/StationPage'
import { StationSelectionProvider } from '@/hooks/useStationSelection'
import { formatDateTime } from '@/lib/format'

interface MockSeries {
  data?: SeriesResponse
  isLoading: boolean
  isError: boolean
  error: Error | null
}

const mocks = vi.hoisted(() => ({
  series: { data: undefined, isLoading: true, isError: false, error: null } as MockSeries,
  /* Every `useStationSeries(city, hours)` call, so the window switch is provable. */
  seriesArgs: [] as Array<{ city: string; hours: number }>,
  chart: { points: [] as WeatherPoint[] },
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
  useStationSeries: (city: string, hours: number) => {
    mocks.seriesArgs.push({ city, hours })
    return { ...mocks.series, refetch: vi.fn(), isFetching: false }
  },
  usePredictions: () => mocks.predictions,
}))

/* The chart's own rendering is covered by HistoryChart.test.tsx; here the
   contract that matters is *what the page hands it* — one series, ascending. */
vi.mock('@/components/charts/HistoryChart', () => ({
  HistoryChart: ({ points }: { points: WeatherPoint[] }) => {
    mocks.chart.points = points
    return (
      <div data-testid="history-chart">
        {points.map((point) => point.observed_at).join('|')}
      </div>
    )
  },
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

function point(observedAt: string, overrides: Partial<CurrentWeather> = {}): WeatherPoint {
  return currentWeatherSchema.parse({
    city: 'Madrid',
    temperature: 22,
    humidity: 60,
    pressure: 1014,
    wind_speed: 10,
    wind_direction: 200,
    precipitation: 0.2,
    observed_at: observedAt,
    ...overrides,
  })
}

/* Ascending, as `/weather/series` serves it: oldest first, newest last. */
const OLDEST = '2026-09-25T08:00:00Z'
const MIDDLE = '2026-09-25T09:00:00Z'
const NEWEST = '2026-09-25T10:00:00Z'

function seriesResponse(points: WeatherPoint[]): SeriesResponse {
  return {
    hours: 48,
    count: 1,
    cities: [
      {
        city: 'Madrid',
        count: points.length,
        latest_timestamp: points.at(-1)?.observed_at ?? null,
        points,
      },
    ],
  }
}

function loadStationData() {
  mocks.series = {
    data: seriesResponse([
      point(OLDEST, { temperature: 18 }),
      point(MIDDLE, { temperature: 20 }),
      point(NEWEST, { temperature: 22, apparent_temperature: 21.5, cloud_cover: 20 }),
    ]),
    isLoading: false,
    isError: false,
    error: null,
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
  mocks.series = { data: undefined, isLoading: true, isError: false, error: null }
  mocks.seriesArgs.length = 0
  mocks.chart.points = []
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

/**
 * One `/weather/series` request feeds both halves of the surface: the hero is
 * the *newest* point and the chart is the same points oldest -> newest. These
 * tests pin the derivation, not the transport (see StationPageRequests).
 */
describe('StationPage derives hero and chart from one series', () => {
  it('shows the newest point in the hero, not the first one', () => {
    loadStationData()
    renderPage('?city=Madrid')

    /* 22.0 °C is the newest point; 18.0 °C is the oldest. */
    expect(screen.getByText('22,0')).toBeInTheDocument()
    expect(screen.queryByText('18,0')).not.toBeInTheDocument()
    /* `Registrado` must read the newest timestamp, never the window start. */
    expect(screen.getByText(formatDateTime(NEWEST))).toBeInTheDocument()
    expect(screen.queryByText(formatDateTime(OLDEST))).not.toBeInTheDocument()
  })

  it('hands the chart the series oldest -> newest', () => {
    loadStationData()
    renderPage('?city=Madrid')

    expect(screen.getByTestId('history-chart')).toHaveTextContent(
      [OLDEST, MIDDLE, NEWEST].join('|'),
    )
    expect(mocks.chart.points.map((entry) => entry.observed_at)).toEqual([
      OLDEST,
      MIDDLE,
      NEWEST,
    ])
  })

  it('asks for the series with the selected window and refetches when it changes', async () => {
    loadStationData()
    renderPage('?city=Madrid')

    /* URL default is 48 h. */
    expect(mocks.seriesArgs.at(-1)).toEqual({ city: 'Madrid', hours: 48 })

    await userEvent.click(screen.getByRole('radio', { name: '24 h' }))

    expect(mocks.seriesArgs.at(-1)).toEqual({ city: 'Madrid', hours: 24 })
    expect(screen.getByRole('radio', { name: '24 h' })).toHaveAttribute('aria-checked', 'true')
  })

  it('renders an empty series as an empty state, not as a stale hero', () => {
    mocks.series = {
      data: seriesResponse([]),
      isLoading: false,
      isError: false,
      error: null,
    }
    renderPage('?city=Madrid')

    expect(screen.getByText('Sin observaciones recientes')).toBeInTheDocument()
    expect(screen.getByText(/No hay observaciones de Madrid en las últimas 48 h/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reintentar' })).toBeInTheDocument()
    expect(screen.queryByText('Observación')).not.toBeInTheDocument()
  })

  it('keeps the failure state (and its retry) for both halves', () => {
    mocks.series = {
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("City 'Madrid' not found"),
    }
    renderPage('?city=Madrid')

    expect(screen.getByText('No se pudo cargar la observación')).toBeInTheDocument()
    /* One failed request, both halves reporting it: hero and chart panel. */
    expect(screen.getAllByText("City 'Madrid' not found")).toHaveLength(2)
    expect(screen.getByRole('button', { name: 'Reintentar' })).toBeInTheDocument()
    expect(screen.getByText('Sin histórico')).toBeInTheDocument()
  })

  it('shows the loading state while the series is in flight', () => {
    renderPage('?city=Madrid')
    /* The prediction card owns another loading block; the point here is that the
       observed half is a placeholder, not an empty or stale hero. */
    expect(screen.getAllByRole('status').length).toBeGreaterThan(0)
    expect(screen.queryByText('Observación')).not.toBeInTheDocument()
    expect(screen.queryByTestId('history-chart')).not.toBeInTheDocument()
  })
})
