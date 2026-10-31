import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { MapLayers } from '@/app/preferences'
import { MapLayersPanel, MobileMapLayers } from '@/components/map/MapLayersControl'

const LAYERS: MapLayers = { hillshade: false, radar: false, terrain3d: false, buildings3d: true }

function props(overrides: Partial<Parameters<typeof MapLayersPanel>[0]> = {}) {
  return {
    mapLayers: LAYERS,
    onLayerChange: vi.fn(),
    basemap: 'positron' as const,
    onBasemapChange: vi.fn(),
    radarUnavailable: false,
    ...overrides,
  }
}

describe('MapLayersPanel (>= sm)', () => {
  it('keeps the desktop panel, hidden below the breakpoint', () => {
    render(<MapLayersPanel {...props()} />)

    /* CSS, not a JS media query: desktop shows it, the phone hides it. */
    const panel = screen.getByText('Capas').parentElement as HTMLElement
    expect(panel.className).toContain('hidden')
    expect(panel.className).toContain('sm:block')
    expect(panel.className).toContain('w-44')

    /* Identical content: four layers, the basemap and both legends. */
    for (const label of ['Relieve', 'Radar de lluvia', 'Terreno 3D', 'Edificios 3D']) {
      expect(screen.getByLabelText(label)).toBeInTheDocument()
    }
    expect(screen.getByLabelText('Base')).toBeInTheDocument()
    expect(screen.getByText('Temperatura')).toBeInTheDocument()
    expect(screen.getByText('Viento (flecha)')).toBeInTheDocument()
    /* Ramp ends carry the same domain values as before the extraction. */
    expect(screen.getByText('icono = temp.')).toBeInTheDocument()
    expect(screen.getByText('-5°')).toBeInTheDocument()
    expect(screen.getByText('40°')).toBeInTheDocument()
    expect(screen.getByText('tamaño = vel.')).toBeInTheDocument()
    expect(screen.getByText('0')).toBeInTheDocument()
    expect(screen.getByText('60 km/h')).toBeInTheDocument()
  })

  it('reports a layer toggle with its key', async () => {
    const onLayerChange = vi.fn()
    render(<MapLayersPanel {...props({ onLayerChange })} />)

    await userEvent.click(screen.getByLabelText('Radar de lluvia'))

    expect(onLayerChange).toHaveBeenCalledWith('radar', true)
  })
})

describe('MobileMapLayers (< sm)', () => {
  it('starts closed with a 44 px "Capas" trigger', () => {
    render(<MobileMapLayers {...props()} />)

    const trigger = screen.getByRole('button', { name: /^capas$/i })
    expect(trigger).toHaveAttribute('aria-expanded', 'false')
    expect(trigger).toHaveAttribute('aria-haspopup', 'dialog')
    /* Tap target, not a 14 px checkbox row. */
    expect(trigger.className).toContain('min-h-11')
    expect(trigger.className).toContain('sm:hidden')
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('opens the sheet with the same content and focuses its close button', async () => {
    render(<MobileMapLayers {...props()} />)

    await userEvent.click(screen.getByRole('button', { name: /^capas$/i }))

    const sheet = screen.getByRole('dialog', { name: 'Capas del mapa' })
    expect(screen.getByRole('button', { name: /^capas$/i })).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('button', { name: 'Cerrar capas' })).toHaveFocus()

    for (const label of ['Relieve', 'Radar de lluvia', 'Terreno 3D', 'Edificios 3D']) {
      expect(within(sheet).getByLabelText(label)).toBeInTheDocument()
    }
    const select = within(sheet).getByLabelText('Base')
    expect(select).toHaveValue('positron')
    /* 16 px on touch so focusing the select does not zoom the page. */
    expect(select.className).toContain('text-[16px]')
    expect(within(sheet).getByText('Temperatura')).toBeInTheDocument()
    expect(within(sheet).getByText('Viento (flecha)')).toBeInTheDocument()
  })

  it('clears the bottom nav and the iOS safe area', async () => {
    render(<MobileMapLayers {...props()} />)
    await userEvent.click(screen.getByRole('button', { name: /^capas$/i }))

    const sheet = screen.getByRole('dialog', { name: 'Capas del mapa' })
    expect(sheet.style.marginBottom).toContain('var(--nav-h)')
    expect(sheet.style.marginBottom).toContain('var(--safe-bottom)')
    /* Scrolls instead of growing past the viewport on a small phone. */
    expect(sheet.className).toContain('overflow-y-auto')
    expect(sheet.style.maxHeight).toContain('100svh')
    /* The backdrop is phone-only and covers the whole viewport. */
    const backdrop = sheet.parentElement as HTMLElement
    expect(backdrop.className).toContain('fixed')
    expect(backdrop.className).toContain('inset-0')
    expect(backdrop.className).toContain('sm:hidden')
  })

  it('closes on a layer selection and reports it', async () => {
    const onLayerChange = vi.fn()
    render(<MobileMapLayers {...props({ onLayerChange })} />)
    await userEvent.click(screen.getByRole('button', { name: /^capas$/i }))

    await userEvent.click(screen.getByLabelText('Terreno 3D'))

    expect(onLayerChange).toHaveBeenCalledWith('terrain3d', true)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('closes on a basemap selection and reports it', async () => {
    const onBasemapChange = vi.fn()
    render(<MobileMapLayers {...props({ onBasemapChange })} />)
    await userEvent.click(screen.getByRole('button', { name: /^capas$/i }))

    await userEvent.selectOptions(screen.getByLabelText('Base'), 'topo')

    expect(onBasemapChange).toHaveBeenCalledWith('topo')
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('closes on Escape and hands focus back to the trigger', async () => {
    render(<MobileMapLayers {...props()} />)
    const trigger = screen.getByRole('button', { name: /^capas$/i })
    await userEvent.click(trigger)

    await userEvent.keyboard('{Escape}')

    expect(screen.queryByRole('dialog')).toBeNull()
    expect(trigger).toHaveFocus()
  })

  it('closes on a backdrop tap but not on a tap inside the sheet', async () => {
    render(<MobileMapLayers {...props()} />)
    await userEvent.click(screen.getByRole('button', { name: /^capas$/i }))

    await userEvent.click(screen.getByText('Temperatura'))
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    const backdrop = screen.getByRole('dialog').parentElement as HTMLElement
    await userEvent.pointer({ keys: '[MouseLeft]', target: backdrop })
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('renders the radar warning inside the sheet', async () => {
    render(<MobileMapLayers {...props({ radarUnavailable: true })} />)
    await userEvent.click(screen.getByRole('button', { name: /^capas$/i }))

    expect(screen.getByText('Radar no disponible ahora.')).toBeInTheDocument()
  })
})

describe('both surfaces together', () => {
  it('never duplicates an id', () => {
    render(
      <>
        <MapLayersPanel {...props()} />
        <MobileMapLayers {...props()} />
      </>,
    )

    const ids = screen.getAllByLabelText('Base').map((node) => node.id)
    expect(new Set(ids).size).toBe(ids.length)
  })
})
