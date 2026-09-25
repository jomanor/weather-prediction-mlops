import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ComponentProps } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { RadarControls } from '@/components/map/RadarControls'
import type { RadarFrame } from '@/components/map/radar'

const FRAMES: RadarFrame[] = [
  { time: 1000, path: '/radar/p1', kind: 'past' },
  { time: 1200, path: '/radar/p2', kind: 'past' },
  { time: 1400, path: '/radar/n1', kind: 'nowcast' },
]

function renderControls(overrides: Partial<ComponentProps<typeof RadarControls>> = {}) {
  const props = {
    frames: FRAMES,
    index: 0,
    onIndex: vi.fn(),
    playing: false,
    onTogglePlay: vi.fn(),
    ...overrides,
  }
  return { ...render(<RadarControls {...props} />), props }
}

describe('RadarControls', () => {
  it('toggles playback and scrubs frames', async () => {
    const { props } = renderControls()
    await userEvent.click(screen.getByRole('button', { name: 'Reproducir radar' }))
    expect(props.onTogglePlay).toHaveBeenCalledTimes(1)

    fireEvent.change(screen.getByLabelText('Fotograma del radar'), { target: { value: '2' } })
    expect(props.onIndex).toHaveBeenCalledWith(2)
    expect(screen.getByText('1/3')).toBeInTheDocument()
  })

  it('shows the pause affordance and nowcast badge while playing ahead', async () => {
    renderControls({ index: 2, playing: true })
    expect(screen.getByRole('button', { name: 'Pausar radar' })).toBeInTheDocument()
    expect(screen.getByText('Previsión')).toBeInTheDocument()
    expect(screen.getByText('3/3')).toBeInTheDocument()
  })

  it('disables playback under reduced motion but keeps the scrubber usable', () => {
    renderControls({ reducedMotion: true })
    expect(
      screen.getByRole('button', { name: 'Reproducción desactivada por reducción de movimiento' }),
    ).toBeDisabled()
    expect(screen.getByLabelText('Fotograma del radar')).toBeEnabled()
  })

  it('disables playback with a single frame', () => {
    renderControls({ frames: [FRAMES[0]] })
    expect(screen.getByRole('button', { name: 'Reproducir radar' })).toBeDisabled()
  })
})
