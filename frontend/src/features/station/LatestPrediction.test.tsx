import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { predictionSchema, type Prediction } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { LatestPrediction } from '@/features/station/LatestPrediction'

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

function renderLatest(predictions: Prediction[]) {
  return render(
    <PreferencesProvider>
      <LatestPrediction predictions={predictions} />
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
