import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { Segmented } from '@/components/ui/Segmented'

const OPTIONS = [
  { value: 24, label: '24 h' },
  { value: 48, label: '48 h' },
]

describe('Segmented', () => {
  it('marks the active option as checked', () => {
    render(<Segmented value={24} onChange={() => {}} options={OPTIONS} label="Rango" />)
    expect(screen.getByRole('radio', { name: '24 h' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: '48 h' })).toHaveAttribute('aria-checked', 'false')
  })

  it('emits the selected value', async () => {
    const onChange = vi.fn()
    render(<Segmented value={24} onChange={onChange} options={OPTIONS} label="Rango" />)
    await userEvent.click(screen.getByRole('radio', { name: '48 h' }))
    expect(onChange).toHaveBeenCalledWith(48)
  })

  it('moves and selects with the arrow keys over a single tab stop', async () => {
    const onChange = vi.fn()
    render(<Segmented value={24} onChange={onChange} options={OPTIONS} label="Rango" />)
    const first = screen.getByRole('radio', { name: '24 h' })
    const second = screen.getByRole('radio', { name: '48 h' })

    expect(first).toHaveAttribute('tabindex', '0')
    expect(second).toHaveAttribute('tabindex', '-1')

    first.focus()
    await userEvent.keyboard('{ArrowRight}')
    expect(onChange).toHaveBeenLastCalledWith(48)
    expect(second).toHaveFocus()

    await userEvent.keyboard('{ArrowLeft}')
    expect(onChange).toHaveBeenLastCalledWith(24)
    expect(first).toHaveFocus()
  })
})
