import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { City } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { AnalyticsPanel } from '@/features/system/components/AnalyticsPanel'
import { StationSelectionProvider } from '@/hooks/useStationSelection'

/**
 * Analítica is a Sistema tab: the selector and the diurnal matrix must stay
 * inside AEMET coverage. `/analytics/diurnal` returns the server-side canonical
 * city list, which still contains the US station, so the panel filters it.
 */
const REGISTRY: City[] = [
  { name: 'Madrid', latitude: 40.42, longitude: -3.7 },
  { name: 'Vigo', latitude: 42.24, longitude: -8.72 },
  { name: 'Pasadena', latitude: 34.15, longitude: -118.14 },
]

const DIURNAL = {
  days: 90,
  cells: [
    { city: 'Madrid', hour: 0, tmean: 12, n: 30 },
    { city: 'Pasadena', hour: 0, tmean: 21, n: 30 },
    { city: 'Vigo', hour: 0, tmean: 11, n: 30 },
    { city: 'Pasadena', hour: 13, tmean: 27, n: 30 },
  ],
}

const mocks = vi.hoisted(() => ({ daily: [] as string[], wind: [] as string[] }))

vi.mock('@/api/queries', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/queries')>()
  return {
    ...actual,
    useCities: () => ({ data: REGISTRY, isError: false, error: null }),
  }
})

vi.mock('@/features/analytics/queries', () => ({
  useAnalyticsDiurnal: () => ({
    data: DIURNAL,
    isLoading: false,
    isError: false,
    isFetching: false,
    error: null,
    refetch: vi.fn(),
  }),
  useAnalyticsDaily: (city: string) => {
    mocks.daily.push(city)
    return {
      data: { city, days: 90, points: [] },
      isLoading: false,
      isError: false,
      isFetching: false,
      error: null,
      refetch: vi.fn(),
    }
  },
  useAnalyticsWindRose: (city: string) => {
    mocks.wind.push(city)
    return {
      data: { city, days: 90, sectors: [] },
      isLoading: false,
      isError: false,
      isFetching: false,
      error: null,
      refetch: vi.fn(),
    }
  },
}))

/* Recharts needs a sized container; the matrix is what this test asserts. */
vi.mock('@/features/analytics/components/DailyAnomalyChart', () => ({
  DailyAnomalyChart: () => <p>grafico diario</p>,
}))

function renderPanel(entry = '/sistema?tab=analitica') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[entry]}>
        <PreferencesProvider>
          <StationSelectionProvider>
            <AnalyticsPanel />
          </StationSelectionProvider>
        </PreferencesProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

const options = () => screen.getAllByRole('option').map((option) => option.textContent)

beforeEach(() => {
  mocks.daily.length = 0
  mocks.wind.length = 0
})

describe('AnalyticsPanel city coverage', () => {
  it('offers only AEMET-covered stations in the city selector', () => {
    renderPanel()
    expect(screen.getByLabelText('Estación')).toBeInTheDocument()
    expect(options()).toEqual(['Madrid', 'Vigo'])
  })

  it('falls back to a Spanish station for a Pasadena deep link', () => {
    renderPanel('/sistema?tab=analitica&city=Pasadena')
    expect(screen.getByLabelText('Estación')).toHaveValue('Madrid')
    expect(mocks.daily).toContain('Madrid')
    expect(mocks.wind).toContain('Madrid')
    expect(mocks.daily).not.toContain('Pasadena')
  })

  it('drops the Pasadena row from the diurnal matrix', () => {
    renderPanel()
    const rows = screen.getAllByRole('rowheader').map((row) => row.textContent)
    expect(rows).toEqual(['Madrid', 'Vigo'])
    expect(screen.queryByText('Pasadena')).not.toBeInTheDocument()
    /* Two cells survive the filter (Madrid 0 h, Vigo 0 h). */
    expect(screen.getByText('2 celdas')).toBeInTheDocument()
  })
})
