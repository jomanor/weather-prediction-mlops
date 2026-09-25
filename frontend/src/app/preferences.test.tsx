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
