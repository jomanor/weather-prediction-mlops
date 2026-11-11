import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useWeatherQuality } from '@/api/queries'
import { DataQualityMeter } from '@/features/system/components/DataQualityMeter'

/* The registry is the only source of coordinates, so it decides coverage. */
const REGISTRY = [
  { name: 'Madrid', latitude: 40.42, longitude: -3.7 },
  { name: 'Vigo', latitude: 42.24, longitude: -8.72 },
  { name: 'Sevilla', latitude: 37.39, longitude: -5.98 },
  { name: 'Pasadena', latitude: 34.15, longitude: -118.14 },
]

vi.mock('@/api/queries', () => ({
  useWeatherQuality: vi.fn(),
  useCities: () => ({ data: REGISTRY }),
}))

const mocked = vi.mocked(useWeatherQuality)

function mockQuery(overrides: Record<string, unknown>) {
  mocked.mockReturnValue(
    {
      isLoading: false,
      isError: false,
      isFetching: false,
      error: null,
      refetch: vi.fn(),
      data: undefined,
      ...overrides,
    } as unknown as ReturnType<typeof useWeatherQuality>,
  )
}

const DATA = {
  generated_at: '2026-09-25T12:00:00Z',
  days: 7,
  cities: [
    {
      city: 'Madrid',
      expected_hours: 168,
      observed_hours: 165,
      completeness: 0.982,
      max_gap_hours: 3,
      null_rate: 0.012,
      last_observed_at: '2026-09-25T11:00:00Z',
      age_hours: 1,
      status: 'ok' as const,
    },
    {
      city: 'Vigo',
      expected_hours: 168,
      observed_hours: 140,
      completeness: 0.833,
      max_gap_hours: 12,
      null_rate: 0.02,
      last_observed_at: '2026-09-25T06:00:00Z',
      age_hours: 6,
      status: 'warn' as const,
    },
    {
      city: 'Sevilla',
      expected_hours: 168,
      observed_hours: 40,
      completeness: 0.238,
      max_gap_hours: 60,
      null_rate: 0.4,
      last_observed_at: null,
      age_hours: null,
      status: 'bad' as const,
    },
    {
      city: 'Pasadena',
      expected_hours: 168,
      observed_hours: 12,
      completeness: 0.071,
      max_gap_hours: 96,
      null_rate: 0.5,
      last_observed_at: null,
      age_hours: null,
      status: 'bad' as const,
    },
  ],
}

describe('DataQualityMeter', () => {
  beforeEach(() => {
    mocked.mockReset()
  })

  it('pairs every status colour with a text label and an icon', () => {
    mockQuery({ data: DATA })
    const { container } = render(<DataQualityMeter />)

    expect(screen.getByText('Calidad de datos')).toBeInTheDocument()
    expect(screen.getByText('Madrid')).toBeInTheDocument()
    expect(screen.getByText('Correcto')).toBeInTheDocument()
    expect(screen.getByText('Aviso')).toBeInTheDocument()
    // "Crítico" is the worst-status header badge and the Sevilla row.
    expect(screen.getAllByText('Crítico').length).toBeGreaterThanOrEqual(2)
    // Status is never encoded by colour alone: labels exist as text.
    expect(container.querySelectorAll('svg.lucide').length).toBeGreaterThanOrEqual(3)
  })

  it('sorts the worst stations first and shows the national coverage', () => {
    mockQuery({ data: DATA })
    render(<DataQualityMeter />)
    const cities = screen.getAllByText(/Madrid|Vigo|Sevilla/).map((node) => node.textContent)
    expect(cities[0]).toBe('Sevilla')
    // 345 observed / 504 expected -> 68.5 %
    expect(screen.getByText('68,5%')).toBeInTheDocument()
  })

  it('leaves Pasadena out of the rows and of every aggregate', () => {
    mockQuery({ data: DATA })
    render(<DataQualityMeter />)

    /* Outside AEMET coverage: no row, and it cannot become the worst status. */
    expect(screen.queryByText('Pasadena')).not.toBeInTheDocument()
    /* Aggregates are computed over the three Spanish rows only. */
    expect(screen.getByText('68,5%')).toBeInTheDocument()
    const readout = (label: string) => screen.getByText(label).parentElement?.textContent ?? ''
    expect(readout('Estaciones')).toContain('3')
    expect(readout('Con aviso')).toContain('2')
    expect(screen.getByText('Solo estaciones con cobertura AEMET.')).toBeInTheDocument()
  })

  it('drops the per-station table when every station is in range', () => {
    const healthy = {
      ...DATA,
      cities: DATA.cities
        .filter((city) => city.status === 'ok')
        .map((city) => ({ ...city, completeness: 0.99, status: 'ok' as const })),
    }
    mockQuery({ data: healthy })
    render(<DataQualityMeter />)

    /* A row per station saying "Correcto" said nothing the summary did not. */
    expect(screen.queryByRole('table')).toBeNull()
    expect(screen.queryByRole('columnheader', { name: 'Estación' })).toBeNull()

    /* The accessible status text stays: the counts, the hint and the worst badge. */
    expect(screen.getByText('Con aviso')).toBeInTheDocument()
    expect(screen.getByText('todo en rango')).toBeInTheDocument()
    expect(screen.getByText('Correcto')).toBeInTheDocument()
    expect(screen.getByText('Solo estaciones con cobertura AEMET.')).toBeInTheDocument()
  })

  it('renders a retry action on error', () => {
    mockQuery({ isError: true, error: new Error('boom') })
    render(<DataQualityMeter />)
    expect(screen.getByText('No se pudo consultar la calidad')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reintentar' })).toBeInTheDocument()
  })

  it('renders skeleton rows while loading', () => {
    mockQuery({ isLoading: true })
    render(<DataQualityMeter />)
    expect(screen.queryByText('Madrid')).not.toBeInTheDocument()
  })
})
