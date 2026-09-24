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
})
