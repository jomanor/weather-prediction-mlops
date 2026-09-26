import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { CommandPalette } from '@/components/shell/CommandPalette'

vi.mock('@/api/queries', () => ({
  useCities: () => ({ data: [{ name: 'Madrid' }, { name: 'Sevilla' }] }),
}))

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname + location.search}</output>
}

function renderPalette() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <CommandPalette />
      <LocationProbe />
    </MemoryRouter>,
  )
}

function openPalette() {
  fireEvent.keyDown(window, { key: 'k', metaKey: true })
}

describe('CommandPalette', () => {
  it('opens on Cmd/Ctrl-K and closes on Escape', () => {
    renderPalette()
    expect(screen.queryByRole('dialog')).toBeNull()

    openPalette()
    const dialog = screen.getByRole('dialog', { name: 'Paleta de comandos' })
    expect(dialog).toHaveAttribute('aria-modal', 'true')

    fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Escape' })
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('exposes combobox/listbox ARIA and moves the active option with arrows', () => {
    renderPalette()
    openPalette()

    const input = screen.getByRole('combobox', { name: 'Buscar secciones y estaciones' })
    expect(input).toHaveFocus()
    expect(input).toHaveAttribute('aria-controls', 'command-palette-list')
    expect(input).toHaveAttribute('aria-activedescendant', 'palette-option-0')

    fireEvent.keyDown(input, { key: 'ArrowDown' })
    expect(input).toHaveAttribute('aria-activedescendant', 'palette-option-1')
    fireEvent.keyDown(input, { key: 'ArrowUp' })
    expect(input).toHaveAttribute('aria-activedescendant', 'palette-option-0')

    const options = screen.getAllByRole('option')
    expect(options.length).toBeGreaterThan(1)
    expect(options[0]).toHaveAttribute('aria-selected', 'true')
  })

  it('navigates to the highlighted item on Enter', () => {
    renderPalette()
    openPalette()

    fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' })

    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getByTestId('location')).toHaveTextContent('/')
  })

  it('traps Tab inside the dialog', () => {
    renderPalette()
    openPalette()
    const input = screen.getByRole('combobox')
    const close = screen.getByRole('button', { name: 'Cerrar la paleta' })

    close.focus()
    fireEvent.keyDown(close, { key: 'Tab' })
    expect(input).toHaveFocus()

    fireEvent.keyDown(input, { key: 'Tab', shiftKey: true })
    expect(close).toHaveFocus()
  })

  it('restores focus to the opener when closed', () => {
    renderPalette()
    const opener = document.createElement('button')
    document.body.appendChild(opener)
    opener.focus()

    openPalette()
    expect(screen.getByRole('combobox')).toHaveFocus()

    fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Escape' })
    expect(opener).toHaveFocus()
    opener.remove()
  })
})
