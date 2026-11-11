import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { AppShell } from '@/app/AppShell'
import { PreferencesProvider } from '@/app/preferences'

vi.mock('@/api/queries', () => ({
  useCities: () => ({ data: [{ name: 'Madrid' }], isLoading: false }),
}))

function renderShell() {
  return render(
    <PreferencesProvider>
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<p>contenido</p>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </PreferencesProvider>,
  )
}

/**
 * Finding 19: the shell mounted `StationSearch` twice — once for the phone row
 * and once for the desktop bar — so the results list id was duplicated and two
 * global `/` listeners were live. One mount, and CSS decides where it sits.
 */
describe('AppShell', () => {
  it('mounts the station search exactly once', () => {
    const { container } = renderShell()

    expect(screen.getAllByRole('combobox', { name: 'Buscar estación' })).toHaveLength(1)
    expect(container.querySelectorAll('[aria-controls="station-search-results"]')).toHaveLength(1)
    /* One theme group, one unit group: the shell has one of each. */
    expect(screen.getAllByRole('radiogroup', { name: 'Tema' })).toHaveLength(1)
    expect(screen.getAllByRole('radiogroup', { name: 'Unidades' })).toHaveLength(1)
  })

  it('keeps the four sections and the routed content', () => {
    renderShell()

    expect(screen.getAllByRole('navigation', { name: 'Secciones' })).toHaveLength(2)
    expect(screen.getByText('contenido')).toBeInTheDocument()
    for (const label of ['Resumen', 'Estación', 'Sistema', 'Panel']) {
      expect(screen.getAllByRole('link', { name: label }).length).toBeGreaterThanOrEqual(1)
    }
  })
})
