import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PreferencesProvider, usePreferences } from '@/app/preferences'

const LIGHT: Record<string, string> = { '--obs': '#0f141a' }
const DARK: Record<string, string> = { '--obs': '#f1f3f6' }

/** The chart palette reads tokens from `getComputedStyle`, so mirror the class. */
function mockComputedStyle() {
  vi.spyOn(window, 'getComputedStyle').mockImplementation(
    () =>
      ({
        getPropertyValue: (name: string) =>
          (document.documentElement.classList.contains('dark') ? DARK : LIGHT)[name] ?? '',
      }) as unknown as CSSStyleDeclaration,
  )
}

function Probe() {
  const { isDark, palette, setTheme } = usePreferences()
  return (
    <>
      <span data-testid="obs">{palette.observed}</span>
      <span data-testid="is-dark">{String(isDark)}</span>
      <button onClick={() => setTheme('dark')}>Tema oscuro</button>
      <button onClick={() => setTheme('light')}>Tema claro</button>
    </>
  )
}

describe('PreferencesProvider chart palette', () => {
  beforeEach(() => {
    mockComputedStyle()
    document.documentElement.classList.remove('dark')
    localStorage.setItem('meteoml.theme', 'light')
  })

  afterEach(() => {
    vi.restoreAllMocks()
    document.documentElement.classList.remove('dark')
    localStorage.clear()
  })

  it('recomputes the palette from the theme that gets applied', async () => {
    render(
      <PreferencesProvider>
        <Probe />
      </PreferencesProvider>,
    )
    expect(screen.getByTestId('obs')).toHaveTextContent('#0f141a')

    await userEvent.click(screen.getByRole('button', { name: 'Tema oscuro' }))
    expect(document.documentElement).toHaveClass('dark')
    expect(screen.getByTestId('obs')).toHaveTextContent('#f1f3f6')

    await userEvent.click(screen.getByRole('button', { name: 'Tema claro' }))
    expect(document.documentElement).not.toHaveClass('dark')
    expect(screen.getByTestId('obs')).toHaveTextContent('#0f141a')
  })
})

describe('PreferencesProvider map layers', () => {
  /* Same reason as the basemap suite: a previous suite restores all mocks and
     clears the global matchMedia stub from test/setup.ts. */
  beforeEach(() => {
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))
  })

  afterEach(() => {
    localStorage.clear()
  })

  function LayersProbe() {
    const { mapLayers, setMapLayers } = usePreferences()
    return (
      <>
        <output data-testid="layers">{JSON.stringify(mapLayers)}</output>
        <button onClick={() => setMapLayers({ ...mapLayers, radar: !mapLayers.radar })}>
          Radar
        </button>
      </>
    )
  }

  /* Finding 16: there is one elevation setting, so the removed keys must not be
     resurrected by a stored preference — no migration, they are just dropped. */
  it('drops the removed terrain3d/buildings3d keys from stored preferences', async () => {
    localStorage.setItem(
      'meteoml.map',
      JSON.stringify({ hillshade: true, radar: true, terrain3d: true, buildings3d: true }),
    )

    render(
      <PreferencesProvider>
        <LayersProbe />
      </PreferencesProvider>,
    )

    expect(JSON.parse(screen.getByTestId('layers').textContent ?? '{}')).toEqual({
      hillshade: true,
      radar: true,
    })

    /* …and a write serialises only the two surviving keys. */
    await userEvent.click(screen.getByRole('button', { name: 'Radar' }))
    expect(JSON.parse(localStorage.getItem('meteoml.map') ?? '{}')).toEqual({
      hillshade: true,
      radar: false,
    })
  })

  it('defaults both overlays off with no stored preference', () => {
    render(
      <PreferencesProvider>
        <LayersProbe />
      </PreferencesProvider>,
    )
    expect(JSON.parse(screen.getByTestId('layers').textContent ?? '{}')).toEqual({
      hillshade: false,
      radar: false,
    })
  })
})

function BasemapProbe() {
  const { basemap, setBasemap } = usePreferences()
  return (
    <>
      <span data-testid="basemap">{basemap}</span>
      <button onClick={() => setBasemap('topo')}>Relieve</button>
    </>
  )
}

describe('PreferencesProvider basemap', () => {
  beforeEach(() => {
    /* The chart-palette suite restores all mocks in its `afterEach`, which
       clears the global matchMedia mock set up in test/setup.ts. */
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }))
  })

  afterEach(() => {
    localStorage.clear()
  })

  it('defaults to the light basemap when the OS is not dark', () => {
    render(
      <PreferencesProvider>
        <BasemapProbe />
      </PreferencesProvider>,
    )
    expect(screen.getByTestId('basemap')).toHaveTextContent('positron')
  })

  it('restores a stored choice and persists a new one', async () => {
    localStorage.setItem('meteoml.basemap', 'dark')
    render(
      <PreferencesProvider>
        <BasemapProbe />
      </PreferencesProvider>,
    )
    expect(screen.getByTestId('basemap')).toHaveTextContent('dark')

    await userEvent.click(screen.getByRole('button', { name: 'Relieve' }))
    expect(screen.getByTestId('basemap')).toHaveTextContent('topo')
    expect(localStorage.getItem('meteoml.basemap')).toBe('topo')
  })
})
