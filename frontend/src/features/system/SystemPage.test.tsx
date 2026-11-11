import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { PreferencesProvider } from '@/app/preferences'
import { AppRoutes } from '@/app/routes'
import { SystemPage } from '@/features/system/SystemPage'
import { StationSelectionProvider } from '@/hooks/useStationSelection'

/* The three tab panels are their own surfaces with their own queries; this test
   covers the tab contract (default, switch, URL, the removed Analítica tab) and
   the legacy redirects. */
vi.mock('@/features/system/components/DataQualityMeter', () => ({
  DataQualityMeter: () => <p>panel calidad</p>,
}))
vi.mock('@/features/system/components/ModelsPanel', () => ({
  ModelsPanel: () => <p>panel modelos</p>,
}))
vi.mock('@/features/system/components/BenchmarkPanel', () => ({
  BenchmarkPanel: () => <p>panel benchmark</p>,
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

    await userEvent.click(screen.getByRole('radio', { name: 'Benchmark' }))
    expect(screen.getByText('panel benchmark')).toBeInTheDocument()
    expect(screen.getByTestId('location')).toHaveTextContent('/sistema?tab=benchmark')
  })

  it('offers three tabs and no Analítica (deleted, audit 16)', () => {
    renderSystem()
    expect(screen.getAllByRole('radio').map((radio) => radio.textContent)).toEqual([
      'Calidad',
      'Modelos',
      'Benchmark',
    ])
    expect(screen.queryByRole('radio', { name: 'Analítica' })).toBeNull()
  })

  it('falls back to Calidad for an unknown tab', () => {
    renderSystem('/sistema?tab=nope')
    expect(screen.getByText('panel calidad')).toBeInTheDocument()
  })

  /* A stale bookmark must land on a real tab, not on an empty page. */
  it('resolves a legacy ?tab=analitica to Calidad instead of rendering nothing', () => {
    renderSystem('/sistema?tab=analitica')
    expect(screen.getByText('panel calidad')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Calidad' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.queryByRole('radio', { name: 'Analítica' })).toBeNull()
    expect(screen.getByRole('heading', { name: 'Sistema' })).toBeInTheDocument()
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
  /* `/analytics` no longer points at Sistema at all: its two surviving views
     live on the station page now, so the redirect is asserted in the registry
     test (`app/registry.test.ts`) rather than by rendering a second feature. */
  const CASES = [
    ['/benchmark', '/sistema?tab=benchmark', 'panel benchmark'],
    ['/models', '/sistema?tab=modelos', 'panel modelos'],
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
