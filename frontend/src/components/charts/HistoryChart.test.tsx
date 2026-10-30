import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { currentWeatherSchema, predictionSchema } from '@/api/schemas'
import type { WeatherPoint } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { HistoryChart } from '@/components/charts/HistoryChart'
import { readChartPalette } from '@/lib/chart-theme'

const palette = readChartPalette(true)

/* jsdom has no ResizeObserver; Recharts' ResponsiveContainer needs one. */
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver

function point(observed_at: string, temperature: number, precipitation = 0): WeatherPoint {
  return currentWeatherSchema.parse({
    city: 'Madrid',
    observed_at,
    temperature,
    precipitation,
  })
}

const POINTS = [
  point('2026-09-25T09:00:00Z', 20),
  point('2026-09-25T10:00:00Z', 21, 1.2),
  point('2026-09-25T11:00:00Z', 22),
]

const PREDICTIONS = [
  predictionSchema.parse({
    city: 'Madrid',
    source_timestamp: '2026-09-25T09:00:00Z',
    prediction_timestamp: '2026-09-25T11:00:00Z',
    predicted_temperature: 22.5,
    temp_lower: 21,
    temp_upper: 24,
    interval_level: 0.8,
  }),
]

function renderChart(props: Partial<Parameters<typeof HistoryChart>[0]> = {}) {
  return render(
    <PreferencesProvider>
      <HistoryChart
        points={POINTS}
        predictions={PREDICTIONS}
        palette={palette}
        intervalLevel={0.8}
        {...props}
      />
    </PreferencesProvider>,
  )
}

describe('HistoryChart series toggles', () => {
  it('renders one toggle per temperature series, all pressed', () => {
    renderChart()
    for (const label of ['Temperatura', 'Modelo', 'Precipitación', 'Banda 80%']) {
      expect(screen.getByRole('button', { name: label })).toHaveAttribute('aria-pressed', 'true')
    }
  })

  it('hides and restores a series from the keyboard', async () => {
    renderChart()
    const toggle = screen.getByRole('button', { name: 'Modelo' })
    toggle.focus()
    await userEvent.keyboard('{Enter}')
    expect(toggle).toHaveAttribute('aria-pressed', 'false')
    await userEvent.keyboard('{Enter}')
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
  })

  it('never hides the last visible series', async () => {
    renderChart({ variable: 'wind' })
    const toggle = screen.getByRole('button', { name: 'Viento' })
    await userEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
  })

  it('resets visibility when the variable changes', async () => {
    const { rerender } = renderChart()
    await userEvent.click(screen.getByRole('button', { name: 'Temperatura' }))
    expect(screen.getByRole('button', { name: 'Temperatura' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )
    rerender(
      <PreferencesProvider>
        <HistoryChart points={POINTS} predictions={PREDICTIONS} palette={palette} variable="wind" />
      </PreferencesProvider>,
    )
    expect(screen.getByRole('button', { name: 'Viento' })).toHaveAttribute('aria-pressed', 'true')
    rerender(
      <PreferencesProvider>
        <HistoryChart
          points={POINTS}
          predictions={PREDICTIONS}
          palette={palette}
          variable="temperature"
        />
      </PreferencesProvider>,
    )
    expect(screen.getByRole('button', { name: 'Temperatura' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
  })
})

describe('HistoryChart empty state', () => {
  it('renders an explicit message instead of an empty grid', () => {
    renderChart({ points: [], predictions: [] })
    expect(screen.getByText('Sin observaciones en el rango')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Temperatura' })).not.toBeInTheDocument()
  })
})
