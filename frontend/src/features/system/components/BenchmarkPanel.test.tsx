import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
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

const mocks = vi.hoisted(() => ({
  requested: [] as string[],
  /* A test that needs more than one AEMET-covered station sets this. */
  summary: null as typeof SUMMARY | null,
}))

vi.mock('@/api/queries', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/queries')>()
  return {
    ...actual,
    useCities: () => ({ data: REGISTRY, isError: false, error: null }),
    useBenchmarkSummary: () => ({
      data: mocks.summary ?? SUMMARY,
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
  mocks.summary = null
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

describe('BenchmarkPanel national table semantics', () => {
  it('publishes the selected station as a selected row of a grid', () => {
    renderPanel()

    /* `aria-selected` on a bare <tr> is dropped by screen readers: the table is
       a single-select grid, so the state is announced. The interaction is
       unchanged — tab to the row, Enter or Space selects. */
    const grid = screen.getByRole('grid', { name: 'MAE por estación' })
    const rows = within(grid).getAllByRole('row')
    const body = rows.slice(1)

    expect(body).toHaveLength(1)
    expect(body[0]).toHaveAttribute('aria-selected', 'true')
    expect(body[0]).toHaveTextContent('Madrid')
    expect(body[0]).toHaveAttribute('tabindex', '0')
  })

  it('prints MAE at the same 2 dp as the registry tab', () => {
    renderPanel()
    const grid = screen.getByRole('grid', { name: 'MAE por estación' })

    /* PRECISION.error: the row, the verdict and ModelsPanel agree on digits. */
    expect(within(grid).getByText('1,10')).toBeInTheDocument()
    expect(within(grid).getByText('1,40')).toBeInTheDocument()
    expect(screen.queryByText('1,100')).not.toBeInTheDocument()
  })
})

/* A second AEMET-covered station, so "the selected row" is a choice and not the
 * only row in the table. */
const TWO_CITIES = {
  ...SUMMARY,
  cities: [
    ...SUMMARY.cities,
    { city: 'Sevilla', n: 30, model: { mae: 1.6, n: 30 }, aemet: { mae: 1.2, n: 30 } },
  ],
}

function gridRows(): HTMLElement[] {
  const grid = screen.getByRole('grid', { name: 'MAE por estación' })
  return within(grid).getAllByRole('row').slice(1) as HTMLElement[]
}

describe('BenchmarkPanel national table selection', () => {
  it('marks the chosen station and leaves the other rows unselected', () => {
    mocks.summary = TWO_CITIES
    renderPanel()

    /* Madrid is the resolved city, so exactly one row is selected. Hardcoding
       `aria-selected` to true would make the grid claim two selections. */
    expect(gridRows().map((row) => row.getAttribute('aria-selected'))).toEqual(['true', 'false'])
    expect(gridRows()[0]).toHaveTextContent('Madrid')
    expect(gridRows()[1]).toHaveTextContent('Sevilla')
  })

  it('selects the row it was tabbed to on Enter', async () => {
    mocks.summary = TWO_CITIES
    renderPanel()
    const sevilla = gridRows()[1]

    sevilla.focus()
    expect(sevilla).toHaveFocus()
    await userEvent.keyboard('{Enter}')

    expect(screen.getByLabelText('Estación')).toHaveValue('Sevilla')
    expect(mocks.requested).toContain('Sevilla')
    expect(gridRows()[1]).toHaveAttribute('aria-selected', 'true')
    expect(gridRows()[0]).toHaveAttribute('aria-selected', 'false')
  })

  it('selects on Space as well, without scrolling or navigating away', async () => {
    mocks.summary = TWO_CITIES
    renderPanel()

    gridRows()[1].focus()
    await userEvent.keyboard(' ')

    expect(screen.getByLabelText('Estación')).toHaveValue('Sevilla')
  })

  it('prints the verdict difference at the same 2 dp as the table', () => {
    renderPanel()

    /* Model 1,10 vs AEMET 1,40: the gap is 0,3 and it is printed as 0,30, the
       same PRECISION.error the two MAE columns use. */
    expect(
      screen.getByText('El modelo Spark GBT reduce el MAE en 0,30 °C frente a AEMET.'),
    ).toBeInTheDocument()
  })
})
