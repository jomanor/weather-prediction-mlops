import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useWeatherQuality } from '@/api/queries'
import { DataQualityMeter } from '@/components/DataQualityMeter'

vi.mock('@/api/queries', () => ({ useWeatherQuality: vi.fn() }))

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
