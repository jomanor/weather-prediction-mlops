import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { currentWeatherSchema, predictionSchema } from '@/api/schemas'
import type { WeatherPoint } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { HistoryChart } from '@/components/charts/HistoryChart'
import { ChartSyncProvider, useChartSync } from '@/hooks/useChartSync'
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

/* --- Zoom harness ------------------------------------------------------- */

const HOUR_MS = 3_600_000
/** Seven hourly rows: a six-hour extent, wide enough to zoom into. */
const ZOOM_POINTS = Array.from({ length: 7 }, (_, index) =>
  point(new Date(Date.UTC(2026, 8, 25, 9 + index)).toISOString(), 20 + index),
)
const ZOOM_START = new Date('2026-09-25T09:00:00Z').getTime()
const ZOOM_END = ZOOM_START + 6 * HOUR_MS

/** Renders the shared chart domain, so the sync contract is observable. */
function DomainProbe() {
  const { domain } = useChartSync()
  return <output data-testid="domain">{domain ? `${domain[0]}:${domain[1]}` : 'none'}</output>
}

function sharedDomain(): [number, number] | null {
  const text = screen.getByTestId('domain').textContent ?? 'none'
  if (text === 'none') return null
  const [start, end] = text.split(':').map(Number)
  return [start, end]
}

function renderZoomable(props: Partial<Parameters<typeof HistoryChart>[0]> = {}) {
  return render(
    <PreferencesProvider>
      <ChartSyncProvider>
        <DomainProbe />
        <HistoryChart points={ZOOM_POINTS} palette={palette} {...props} />
      </ChartSyncProvider>
    </PreferencesProvider>,
  )
}

describe('HistoryChart zoom', () => {
  it('renders the zoom toolbar with its hint', () => {
    renderZoomable()
    expect(screen.getByRole('group', { name: 'Zoom del gráfico' })).toBeInTheDocument()
    for (const label of ['Alejar', 'Acercar', 'Restablecer']) {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument()
    }
    expect(
      screen.getByText('Rueda para zoom · arrastra para mover · doble clic para restablecer'),
    ).toBeInTheDocument()
    /* Touch keeps scrolling the page vertically; only horizontal drags pan. */
    expect(screen.getByTestId('chart-surface').className).toContain('touch-pan-y')
  })

  it('starts unzoomed, with nothing to undo', () => {
    renderZoomable()
    expect(sharedDomain()).toBeNull()
    expect(screen.getByRole('button', { name: 'Alejar' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Restablecer' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Acercar' })).toBeEnabled()
  })

  it('narrows the shared domain from the toolbar and resets it', async () => {
    renderZoomable()

    await userEvent.click(screen.getByRole('button', { name: 'Acercar' }))
    const zoomed = sharedDomain()
    expect(zoomed).not.toBeNull()
    /* One step shrinks the six-hour window by the 1.25 zoom factor. */
    expect(zoomed![1] - zoomed![0]).toBeCloseTo((6 * HOUR_MS) / 1.25, 0)
    expect((zoomed![0] + zoomed![1]) / 2).toBeCloseTo((ZOOM_START + ZOOM_END) / 2, 0)
    expect(screen.getByRole('button', { name: 'Alejar' })).toBeEnabled()

    const reset = screen.getByRole('button', { name: 'Restablecer' })
    expect(reset).toBeEnabled()
    await userEvent.click(reset)
    expect(sharedDomain()).toBeNull()
    expect(screen.getByRole('button', { name: 'Alejar' })).toBeDisabled()
  })

  it('zooms in from the keyboard', async () => {
    renderZoomable()
    screen.getByRole('button', { name: 'Acercar' }).focus()
    await userEvent.keyboard('{Enter}')
    expect(sharedDomain()).not.toBeNull()
  })

  it('zooms with the wheel over the plot', () => {
    renderZoomable()
    fireEvent.wheel(screen.getByTestId('chart-surface'), { deltaY: -120 })
    expect(sharedDomain()).not.toBeNull()
  })

  it('resets the zoom on a double click over the plot', async () => {
    renderZoomable()
    await userEvent.click(screen.getByRole('button', { name: 'Acercar' }))
    expect(sharedDomain()).not.toBeNull()
    fireEvent.doubleClick(screen.getByTestId('chart-surface'))
    expect(sharedDomain()).toBeNull()
    expect(screen.getByRole('button', { name: 'Restablecer' })).toBeDisabled()
  })

  it('pans with a pointer drag and ignores a plain click', async () => {
    renderZoomable()
    const surface = screen.getByTestId('chart-surface')
    /* jsdom reports a zero-sized box; the pan maths needs the real width. */
    surface.getBoundingClientRect = () =>
      ({ width: 200, height: 260, top: 0, left: 0, right: 200, bottom: 260 }) as DOMRect

    /* A press that does not move is a click, not a pan. */
    await userEvent.pointer([
      { target: surface, coords: { clientX: 100 }, keys: '[MouseLeft>]' },
      { target: surface, coords: { clientX: 100 }, keys: '[/MouseLeft]' },
    ])
    expect(sharedDomain()).toBeNull()

    await userEvent.click(screen.getByRole('button', { name: 'Acercar' }))
    const [zoomedStart, zoomedEnd] = sharedDomain()!

    /* Dragging left pulls later time into view, until the right edge clamps. */
    await userEvent.pointer([
      { target: surface, coords: { clientX: 100 }, keys: '[MouseLeft>]' },
      { target: surface, coords: { clientX: 40 } },
      { target: surface, coords: { clientX: 40 }, keys: '[/MouseLeft]' },
    ])

    const panned = sharedDomain()
    expect(panned).not.toBeNull()
    expect(panned![1] - panned![0]).toBeCloseTo(zoomedEnd - zoomedStart, 0)
    expect(panned![0]).toBeGreaterThan(zoomedStart)
    expect(panned![1]).toBeLessThanOrEqual(ZOOM_END + 1)
  })

  it('never zooms past the two-hour minimum span', async () => {
    renderZoomable()
    const zoomIn = screen.getByRole('button', { name: 'Acercar' })
    for (let step = 0; step < 12; step += 1) await userEvent.click(zoomIn)
    const zoomed = sharedDomain()!
    expect(zoomed[1] - zoomed[0]).toBeCloseTo(2 * HOUR_MS, 0)
    expect(zoomIn).toBeDisabled()
  })

  it('drops the zoom when the variable changes', async () => {
    const { rerender } = renderZoomable()
    await userEvent.click(screen.getByRole('button', { name: 'Acercar' }))
    expect(sharedDomain()).not.toBeNull()

    rerender(
      <PreferencesProvider>
        <ChartSyncProvider>
          <DomainProbe />
          <HistoryChart points={ZOOM_POINTS} palette={palette} variable="wind" />
        </ChartSyncProvider>
      </PreferencesProvider>,
    )
    expect(sharedDomain()).toBeNull()
  })

  it('disables zooming when the range is already two hours wide', () => {
    renderChart()
    expect(screen.getByRole('button', { name: 'Acercar' })).toBeDisabled()
  })
})

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
