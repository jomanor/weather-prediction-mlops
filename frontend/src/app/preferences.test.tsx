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
