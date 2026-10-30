import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { PreferencesProvider } from '@/app/preferences'
import { AppRoutes } from '@/app/routes'
import { SystemPage } from '@/features/system/SystemPage'
import { StationSelectionProvider } from '@/hooks/useStationSelection'

/* The four tab panels are their own surfaces with their own queries; this test
   covers the tab contract (default, switch, URL) and the legacy redirects. */
vi.mock('@/features/system/components/DataQualityMeter', () => ({
  DataQualityMeter: () => <p>panel calidad</p>,
}))
vi.mock('@/features/system/components/ModelsPanel', () => ({
  ModelsPanel: () => <p>panel modelos</p>,
}))
vi.mock('@/features/system/components/BenchmarkPanel', () => ({
  BenchmarkPanel: () => <p>panel benchmark</p>,
}))
vi.mock('@/features/system/components/AnalyticsPanel', () => ({
  AnalyticsPanel: () => <p>panel analitica</p>,
}))

vi.mock('@/api/queries', () => ({
  useCities: () => ({ data: [] }),
}))

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname + location.search}</output>
}

function renderSystem(entry = '/sistema') {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <PreferencesProvider>
        <StationSelectionProvider>
          <SystemPage />
          <LocationProbe />
        </StationSelectionProvider>
      </PreferencesProvider>
    </MemoryRouter>,
  )
}

describe('SystemPage tabs', () => {
  it('defaults to Calidad and keeps the URL clean', () => {
    renderSystem()
    expect(screen.getByRole('heading', { name: 'Sistema' })).toBeInTheDocument()
    expect(screen.getByText('panel calidad')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Calidad' })).toBeChecked()
    expect(screen.getByTestId('location')).toHaveTextContent('/sistema')
  })

  it('reads the tab from the URL', () => {
    renderSystem('/sistema?tab=benchmark')
    expect(screen.getByText('panel benchmark')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Benchmark' })).toBeChecked()
  })

  it('switches panel and URL together', async () => {
    renderSystem()
    await userEvent.click(screen.getByRole('radio', { name: 'Modelos' }))
    expect(screen.getByText('panel modelos')).toBeInTheDocument()
    expect(screen.queryByText('panel calidad')).not.toBeInTheDocument()
    expect(screen.getByTestId('location')).toHaveTextContent('/sistema?tab=modelos')

    await userEvent.click(screen.getByRole('radio', { name: 'Analítica' }))
    expect(screen.getByText('panel analitica')).toBeInTheDocument()
    expect(screen.getByTestId('location')).toHaveTextContent('/sistema?tab=analitica')
  })

  it('falls back to Calidad for an unknown tab', () => {
    renderSystem('/sistema?tab=nope')
    expect(screen.getByText('panel calidad')).toBeInTheDocument()
  })

  it('is reachable from the keyboard', async () => {
    renderSystem()
    const calidad = screen.getByRole('radio', { name: 'Calidad' })
    calidad.focus()
    await userEvent.keyboard('{ArrowRight}')
    expect(screen.getByText('panel modelos')).toBeInTheDocument()
    expect(screen.getByTestId('location')).toHaveTextContent('/sistema?tab=modelos')
  })
})

describe('legacy paths redirect into Sistema', () => {
  const CASES = [
    ['/benchmark', '/sistema?tab=benchmark', 'panel benchmark'],
    ['/models', '/sistema?tab=modelos', 'panel modelos'],
    ['/analytics', '/sistema?tab=analitica', 'panel analitica'],
  ] as const

  for (const [from, to, panel] of CASES) {
    it(`redirects ${from} to ${to}`, async () => {
      render(
        <MemoryRouter initialEntries={[from]}>
          <PreferencesProvider>
            <StationSelectionProvider>
              <AppRoutes />
              <LocationProbe />
            </StationSelectionProvider>
          </PreferencesProvider>
        </MemoryRouter>,
      )
      expect(await screen.findByText(panel)).toBeInTheDocument()
      expect(screen.getByTestId('location')).toHaveTextContent(to)
    })
  }
})
