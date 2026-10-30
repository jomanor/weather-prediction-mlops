import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { City } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { BenchmarkPanel } from '@/features/system/components/BenchmarkPanel'
import { StationSelectionProvider } from '@/hooks/useStationSelection'

/**
 * Benchmark is a Sistema tab, so it follows the same coverage rule as Calidad:
 * AEMET only covers Spain, and the US station must not be offered or selected
 * here even through a `?city=Pasadena` deep link. The station itself stays on
 * Resumen and Estación.
 */
const REGISTRY: City[] = [
  { name: 'Madrid', latitude: 40.42, longitude: -3.7 },
  { name: 'Sevilla', latitude: 37.39, longitude: -5.98 },
  { name: 'Pasadena', latitude: 34.15, longitude: -118.14 },
]

const SUMMARY = {
  generated_at: '2026-09-25T12:00:00Z',
  aemet_configured: true,
  cities: [
    { city: 'Madrid', n: 24, model: { mae: 1.1, n: 24 }, aemet: { mae: 1.4, n: 24 } },
    { city: 'Pasadena', n: 24, model: { mae: 2.0, n: 24 }, aemet: { mae: null, n: 0 } },
  ],
}

const BENCHMARK = {
  city: 'Madrid',
  hours: 24,
  generated_at: '2026-09-25T12:00:00Z',
  aemet: { available: true, error: null, issued_at: null },
  series: [],
  metrics: { model: { mae: 1.1, n: 24 }, aemet: { mae: 1.4, n: 24 } },
}

const mocks = vi.hoisted(() => ({ requested: [] as string[] }))

vi.mock('@/api/queries', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/queries')>()
  return {
    ...actual,
    useCities: () => ({ data: REGISTRY, isError: false, error: null }),
    useBenchmarkSummary: () => ({
      data: SUMMARY,
      isLoading: false,
      isError: false,
      error: null,
    }),
  }
})

vi.mock('@/features/benchmark/queries', () => ({
  useBenchmark: (city: string) => {
    mocks.requested.push(city)
    return { data: BENCHMARK, isLoading: false, isError: false, error: null }
  },
}))

/* Recharts needs a sized container; the assertion is the wiring, not the SVG. */
vi.mock('@/components/charts/BenchmarkChart', () => ({
  BenchmarkChart: () => <p>grafico benchmark</p>,
}))
vi.mock('@/components/charts/ResidualChart', () => ({
  ResidualChart: () => <p>grafico residuo</p>,
}))

function renderPanel(entry = '/sistema?tab=benchmark') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[entry]}>
        <PreferencesProvider>
          <StationSelectionProvider>
            <BenchmarkPanel />
          </StationSelectionProvider>
        </PreferencesProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

const options = () =>
  screen.getAllByRole('option').map((option) => option.textContent)

beforeEach(() => {
  mocks.requested.length = 0
})

describe('BenchmarkPanel city coverage', () => {
  it('offers only AEMET-covered stations in the city selector', () => {
    renderPanel()
    expect(screen.getByLabelText('Estación')).toBeInTheDocument()
    expect(options()).toEqual(['Madrid', 'Sevilla'])
    expect(options()).not.toContain('Pasadena')
  })

  it('falls back to a Spanish station for a Pasadena deep link', () => {
    renderPanel('/sistema?tab=benchmark&city=Pasadena')
    expect(screen.getByLabelText('Estación')).toHaveValue('Madrid')
    expect(mocks.requested).toContain('Madrid')
    expect(mocks.requested).not.toContain('Pasadena')
  })

  it('keeps a covered deep link and drops Pasadena from the national table', () => {
    renderPanel('/sistema?tab=benchmark&city=Sevilla')
    expect(screen.getByLabelText('Estación')).toHaveValue('Sevilla')
    expect(mocks.requested).toContain('Sevilla')
    /* The summary endpoint still lists the US station; Sistema hides it. */
    const table = screen.getByText('Spark GBT MAE').closest('table') as HTMLElement
    expect(within(table).getByText('Madrid')).toBeInTheDocument()
    expect(within(table).queryByText('Pasadena')).not.toBeInTheDocument()
  })
})
