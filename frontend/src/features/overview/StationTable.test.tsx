import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import type { CurrentWeather } from '@/api/schemas'
import { PreferencesProvider } from '@/app/preferences'
import { StationTable, type StationRow } from '@/features/overview/StationTable'

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

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname + location.search}</output>
}

function renderTable(rows: StationRow[] = [{ name: 'Madrid', station: STATION }], search = '/') {
  return render(
    <MemoryRouter initialEntries={[search]}>
      <PreferencesProvider>
        <StationTable rows={rows} />
        <LocationProbe />
      </PreferencesProvider>
    </MemoryRouter>,
  )
}

describe('StationTable navigation', () => {
  it('opens the station page from the row link', async () => {
    renderTable()
    const link = screen.getByRole('link', { name: 'Madrid' })
    expect(link).toHaveAttribute('href', '/stations?city=Madrid')

    await userEvent.click(link)

    expect(screen.getByTestId('location')).toHaveTextContent('/stations?city=Madrid')
  })

  it('opens the station page from the keyboard', async () => {
    renderTable()
    const link = screen.getByRole('link', { name: 'Madrid' })
    link.focus()
    await userEvent.keyboard('{Enter}')
    expect(screen.getByTestId('location')).toHaveTextContent('/stations?city=Madrid')
  })

  it('keeps the params the surface carries', () => {
    renderTable(undefined, '/?hours=72')
    expect(screen.getByRole('link', { name: 'Madrid' })).toHaveAttribute(
      'href',
      '/stations?hours=72&city=Madrid',
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
    return renderTable([
      { name: 'Sevilla', station: older },
      { name: 'Madrid', station: newer },
    ])
  }

  it('surfaces the age of each row from its own observed_at', () => {
    renderRows()
    const sevilla = screen.getByRole('row', { name: /Sevilla/ })
    expect(within(sevilla).getByText(/hace 3 h/)).toBeInTheDocument()
    const madrid = screen.getByRole('row', { name: /Madrid/ })
    expect(within(madrid).getByText(/hace 10 min/)).toBeInTheDocument()
  })

  it('sorts by temperature', async () => {
    renderRows()
    const before = screen.getAllByRole('row').slice(1).map((row) => row.textContent)
    expect(before[0]).toContain('Madrid')

    await userEvent.click(screen.getByRole('button', { name: /Temp/ }))

    const after = screen.getAllByRole('row').slice(1).map((row) => row.textContent)
    expect(after[0]).toContain('Sevilla')
  })
})
