import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { PreferencesProvider } from '@/app/preferences'
import { StationPage } from '@/features/station/StationPage'
import { StationSelectionProvider } from '@/hooks/useStationSelection'

vi.mock('@/api/queries', () => ({
  useCities: () => ({ data: [{ name: 'Madrid' }, { name: 'Sevilla' }] }),
}))

vi.mock('@/features/station/queries', () => ({
  PREDICTION_HORIZONS: [1, 3, 6, 12, 24],
  useCurrentWeather: () => ({
    isError: false,
    isLoading: true,
    data: undefined,
    error: null,
    refetch: vi.fn(),
    isFetching: false,
  }),
  useHistory: () => ({ isLoading: true, isError: false, error: null, data: undefined }),
  usePredictions: () => ({ isLoading: true, data: [] }),
}))

function renderPage(search: string) {
  return render(
    <MemoryRouter initialEntries={[`/${search}`]}>
      <PreferencesProvider>
        <StationSelectionProvider>
          <StationPage />
        </StationSelectionProvider>
      </PreferencesProvider>
    </MemoryRouter>,
  )
}

describe('StationPage city validation', () => {
  it('falls back to the first station for an unknown URL city', () => {
    renderPage('?city=Unknown')
    expect(screen.getByLabelText('Estación')).toHaveValue('Madrid')
  })

  it('keeps a known URL city', () => {
    renderPage('?city=Sevilla')
    expect(screen.getByLabelText('Estación')).toHaveValue('Sevilla')
  })
})
