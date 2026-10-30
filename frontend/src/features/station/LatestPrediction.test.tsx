import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { predictionSchema, type Prediction } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { LatestPrediction, PREDICTION_STALE_HOURS } from '@/features/station/LatestPrediction'

function makePrediction(overrides: Record<string, unknown>): Prediction {
  return predictionSchema.parse({
    city: 'Madrid',
    source_timestamp: '2026-09-25T09:00:00Z',
    prediction_timestamp: '2026-09-25T10:00:00Z',
    horizon_hours: 1,
    predicted_temperature: 22.1,
    observed_temperature: 21,
    ...overrides,
  })
}

function renderLatest(predictions: Prediction[], horizon = 1) {
  return render(
    <PreferencesProvider>
      <LatestPrediction predictions={predictions} horizon={horizon} />
    </PreferencesProvider>,
  )
}

describe('LatestPrediction interval readout (Contract 2)', () => {
  it('renders the interval range and its nominal level', () => {
    renderLatest([
      makePrediction({ temp_lower: 20.1, temp_upper: 24.1, interval_level: 0.8 }),
    ])

    expect(screen.getByText('Intervalo')).toBeInTheDocument()
    expect(screen.getByText(/20,1°.*24,1°/)).toBeInTheDocument()
    expect(screen.getByText('Nivel')).toBeInTheDocument()
    expect(screen.getByText('80%')).toBeInTheDocument()
  })

  it('hides the interval rows for a legacy prediction with null fields', () => {
    renderLatest([makePrediction({})])

    expect(screen.queryByText('Intervalo')).not.toBeInTheDocument()
    expect(screen.queryByText('Nivel')).not.toBeInTheDocument()
    // The rest of the panel still renders real values.
    expect(screen.getByText('Temp. prevista')).toBeInTheDocument()
  })

  it('does not fabricate a level when the interval has bounds but no level', () => {
    renderLatest([makePrediction({ temp_lower: 20.1, temp_upper: 24.1 })])
    expect(screen.getByText('Intervalo')).toBeInTheDocument()
    expect(screen.queryByText('Nivel')).not.toBeInTheDocument()
  })
})


describe('LatestPrediction picks the newest run and horizon', () => {
  it('shows the newest run row for the selected horizon, not the oldest row', () => {
    /* The API returns history oldest → newest, so the older run comes first. */
    renderLatest(
      [
        makePrediction({
          source_timestamp: '2026-09-25T08:00:00Z',
          prediction_timestamp: '2026-09-25T08:05:00Z',
          horizon_hours: 3,
          predicted_temperature: 15,
        }),
        makePrediction({
          source_timestamp: '2026-09-25T10:00:00Z',
          prediction_timestamp: '2026-09-25T10:05:00Z',
          horizon_hours: 3,
          predicted_temperature: 25,
        }),
      ],
      3,
    )

    expect(screen.getByText('25,0°')).toBeInTheDocument()
    expect(screen.queryByText('15,0°')).not.toBeInTheDocument()
    expect(screen.getByText('+3 h')).toBeInTheDocument()
  })

  it('falls back to the newest row when the selected horizon is missing', () => {
    renderLatest(
      [
        makePrediction({
          prediction_timestamp: '2026-09-25T10:01:00Z',
          horizon_hours: 1,
          predicted_temperature: 21,
        }),
        makePrediction({
          prediction_timestamp: '2026-09-25T10:06:00Z',
          horizon_hours: 6,
          predicted_temperature: 26,
        }),
      ],
      3,
    )

    // Horizon 3 was not written; the newest prediction_timestamp wins safely.
    expect(screen.getByText('26,0°')).toBeInTheDocument()
    expect(screen.getByText('+6 h')).toBeInTheDocument()
  })

  it('renders the empty state when there are no predictions', () => {
    renderLatest([], 3)
    expect(screen.getByText('Sin predicciones para esta estación')).toBeInTheDocument()
  })
})

describe('LatestPrediction age and stale badge', () => {
  const NOW = new Date('2026-09-25T12:00:00Z').getTime()

  afterEach(() => vi.useRealTimers())

  it('shows how old the origin observation is next to the horizon', () => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
    renderLatest([
      makePrediction({ source_timestamp: '2026-09-25T09:00:00Z', prediction_timestamp: '2026-09-25T10:00:00Z' }),
    ])
    expect(screen.getByText('hace 3 h')).toBeInTheDocument()
    expect(screen.queryByText('Antigua')).not.toBeInTheDocument()
  })

  it('marks a prediction older than 24 h as "Antigua" without hiding it', () => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
    renderLatest([
      makePrediction({
        source_timestamp: new Date(NOW - (PREDICTION_STALE_HOURS + 6) * 3_600_000).toISOString(),
      }),
    ])
    expect(screen.getByText(`hace ${PREDICTION_STALE_HOURS + 6} h`)).toBeInTheDocument()
    expect(screen.getByText('Antigua')).toBeInTheDocument()
    // The real prediction values stay visible.
    expect(screen.getByText('Temp. prevista')).toBeInTheDocument()
  })
})
