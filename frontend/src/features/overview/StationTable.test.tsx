import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import type { CurrentWeather, Prediction } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { StationTable } from '@/features/overview/StationTable'
import { StationSelectionProvider, useStationSelection } from '@/hooks/useStationSelection'

const STATION: CurrentWeather = {
  city: 'Madrid',
  latitude: 40.42,
  longitude: -3.7,
  temperature: 21.5,
  apparent_temperature: 21,
  humidity: 40,
  pressure: 1015,
  wind_speed: 10,
  wind_direction: 180,
  precipitation: 0,
  cloud_cover: 10,
  weather_code: 0,
  observed_at: '2026-09-25T10:00:00Z',
}

/** Stands in for the map and the charts, which read the same selection. */
function SelectionProbe() {
  const { selectedCity } = useStationSelection()
  return <p data-testid="probe">{selectedCity ?? 'none'}</p>
}

function renderTable() {
  return render(
    <MemoryRouter>
      <PreferencesProvider>
        <StationSelectionProvider>
          <StationTable
            rows={[{ name: 'Madrid', station: STATION }]}
            predictions={new Map<string, Prediction>()}
          />
          <SelectionProbe />
        </StationSelectionProvider>
      </PreferencesProvider>
    </MemoryRouter>,
  )
}

describe('StationTable cross-filtering', () => {
  it('propagates a row click through the shared station selection', async () => {
    renderTable()
    const row = screen.getByRole('row', { name: /Madrid/ })
    expect(row).toHaveAttribute('aria-selected', 'false')

    await userEvent.click(row)

    expect(row).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByTestId('probe')).toHaveTextContent('Madrid')
  })

  it('selects the station from the keyboard', async () => {
    renderTable()
    const row = screen.getByRole('row', { name: /Madrid/ })
    row.focus()
    await userEvent.keyboard('{Enter}')
    expect(screen.getByTestId('probe')).toHaveTextContent('Madrid')
  })

  it('keeps the surface params when opening a station', () => {
    render(
      <MemoryRouter initialEntries={['/overview?tab=estaciones']}>
        <PreferencesProvider>
          <StationSelectionProvider>
            <StationTable
              rows={[{ name: 'Madrid', station: STATION }]}
              predictions={new Map<string, Prediction>()}
            />
          </StationSelectionProvider>
        </PreferencesProvider>
      </MemoryRouter>,
    )
    expect(screen.getByRole('link', { name: /Abrir la estación Madrid/ })).toHaveAttribute(
      'href',
      '/stations?tab=estaciones&city=Madrid',
    )
  })
})


describe('StationTable data age and sorting', () => {
  const older = {
    ...STATION,
    city: 'Sevilla',
    temperature: 25,
    observed_at: new Date(Date.now() - 3 * 3_600_000).toISOString(),
  }
  const newer = {
    ...STATION,
    city: 'Madrid',
    temperature: 20,
    observed_at: new Date(Date.now() - 10 * 60_000).toISOString(),
  }

  function renderRows() {
    return render(
      <MemoryRouter>
        <PreferencesProvider>
          <StationSelectionProvider>
            <StationTable
              rows={[
                { name: 'Sevilla', station: older },
                { name: 'Madrid', station: newer },
              ]}
              predictions={new Map<string, Prediction>()}
            />
          </StationSelectionProvider>
        </PreferencesProvider>
      </MemoryRouter>,
    )
  }

  it('surfaces the age of each row from its own observed_at', () => {
    renderRows()
    const sevilla = screen.getByRole('row', { name: /Sevilla/ })
    expect(within(sevilla).getByText(/hace 3 h/)).toBeInTheDocument()
    const madrid = screen.getByRole('row', { name: /Madrid/ })
    expect(within(madrid).getByText(/hace 10 min/)).toBeInTheDocument()
  })

  it('sorts by temperature and keeps the selection working', async () => {
    renderRows()
    const before = screen.getAllByRole('row').slice(1).map((row) => row.textContent)
    expect(before[0]).toContain('Madrid')

    await userEvent.click(screen.getByRole('button', { name: /Temp/ }))

    const after = screen.getAllByRole('row').slice(1).map((row) => row.textContent)
    expect(after[0]).toContain('Sevilla')

    await userEvent.click(screen.getByRole('row', { name: /Sevilla/ }))
    expect(screen.getByRole('row', { name: /Sevilla/ })).toHaveAttribute('aria-selected', 'true')
  })
})
