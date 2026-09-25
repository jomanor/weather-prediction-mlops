import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { StationSearch } from '@/components/shell/StationSearch'

vi.mock('@/api/queries', () => ({
  useCities: () => ({ data: [{ name: 'Madrid' }, { name: 'Sevilla' }], isLoading: false }),
}))

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname + location.search}</output>
}

function renderSearch(entry = '/benchmark?range=48') {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <StationSearch />
      <LocationProbe />
    </MemoryRouter>,
  )
}

describe('StationSearch', () => {
  it('keeps the surface params when a station is chosen', async () => {
    renderSearch()
    await userEvent.click(screen.getByRole('combobox', { name: 'Buscar estación' }))
    await userEvent.click(screen.getByRole('button', { name: 'Madrid' }))
    expect(screen.getByTestId('location')).toHaveTextContent('/stations?range=48&city=Madrid')
  })

  it('ignores the / shortcut with modifiers and inside editable fields', () => {
    renderSearch()
    const input = screen.getByRole('combobox', { name: 'Buscar estación' })

    fireEvent.keyDown(document.body, { key: '/', ctrlKey: true })
    expect(input).not.toHaveFocus()
    fireEvent.keyDown(document.body, { key: '/', metaKey: true })
    expect(input).not.toHaveFocus()

    const textarea = document.createElement('textarea')
    document.body.appendChild(textarea)
    fireEvent.keyDown(textarea, { key: '/' })
    expect(input).not.toHaveFocus()

    fireEvent.keyDown(document.body, { key: '/' })
    expect(input).toHaveFocus()

    textarea.remove()
  })
})
