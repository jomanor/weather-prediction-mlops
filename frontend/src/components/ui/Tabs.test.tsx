import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { Tabs, type TabOption } from '@/components/ui/Tabs'

type Section = 'calidad' | 'modelos' | 'benchmark'

const OPTIONS: TabOption<Section>[] = [
  { value: 'calidad', label: 'Calidad' },
  { value: 'modelos', label: 'Modelos' },
  { value: 'benchmark', label: 'Benchmark' },
]

function Harness({ onChange }: { onChange?: (value: Section) => void } = {}) {
  const [tab, setTab] = useState<Section>('calidad')
  return (
    <Tabs
      value={tab}
      onChange={(value) => {
        setTab(value)
        onChange?.(value)
      }}
      options={OPTIONS}
      label="Secciones de Sistema"
    >
      <p>Panel {tab}</p>
    </Tabs>
  )
}

describe('Tabs', () => {
  it('exposes a labelled tablist with the active tab selected', () => {
    render(<Harness />)
    expect(screen.getByRole('tablist', { name: 'Secciones de Sistema' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Calidad' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Modelos' })).toHaveAttribute('aria-selected', 'false')
  })

  it('keeps one tab stop and moves it with the selection', () => {
    render(<Harness />)
    expect(screen.getByRole('tab', { name: 'Calidad' })).toHaveAttribute('tabindex', '0')
    expect(screen.getByRole('tab', { name: 'Modelos' })).toHaveAttribute('tabindex', '-1')
  })

  it('links the panel to the active tab', () => {
    render(<Harness />)
    const panel = screen.getByRole('tabpanel')
    const active = screen.getByRole('tab', { name: 'Calidad' })
    expect(panel).toHaveAttribute('aria-labelledby', active.id)
    expect(active).toHaveAttribute('aria-controls', panel.id)
    expect(panel).toHaveTextContent('Panel calidad')
  })

  it('moves and selects with the arrow keys, wrapping at the ends', async () => {
    render(<Harness />)
    const first = screen.getByRole('tab', { name: 'Calidad' })
    first.focus()

    await userEvent.keyboard('{ArrowRight}')
    expect(screen.getByRole('tab', { name: 'Modelos' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Modelos' })).toHaveFocus()
    expect(screen.getByRole('tabpanel')).toHaveTextContent('Panel modelos')

    await userEvent.keyboard('{ArrowRight}')
    await userEvent.keyboard('{ArrowRight}')
    expect(screen.getByRole('tab', { name: 'Calidad' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Calidad' })).toHaveFocus()

    await userEvent.keyboard('{ArrowLeft}')
    expect(screen.getByRole('tab', { name: 'Benchmark' })).toHaveAttribute('aria-selected', 'true')
  })

  it('jumps to the ends with Home and End', async () => {
    render(<Harness />)
    screen.getByRole('tab', { name: 'Calidad' }).focus()

    await userEvent.keyboard('{End}')
    expect(screen.getByRole('tab', { name: 'Benchmark' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Benchmark' })).toHaveFocus()

    await userEvent.keyboard('{Home}')
    expect(screen.getByRole('tab', { name: 'Calidad' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Calidad' })).toHaveFocus()
  })

  it('selects on click and reports the new value', async () => {
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Modelos' }))
    expect(onChange).toHaveBeenCalledWith('modelos')
    expect(screen.getByRole('tab', { name: 'Modelos' })).toHaveAttribute('aria-selected', 'true')
  })
})
