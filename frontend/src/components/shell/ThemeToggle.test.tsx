import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { PreferencesProvider } from '@/app/preferences'
import { NAV_ITEMS } from '@/app/registry'
import { ThemeToggle } from '@/components/shell/ThemeToggle'

/**
 * The three theme options are icon-only, so the `title` is their accessible
 * name. "Sistema" was also the name of a navigation section: a screen reader
 * heard the same word for a theme and for a page.
 */
describe('ThemeToggle', () => {
  it('names each option explicitly instead of reusing a nav label', () => {
    render(
      <PreferencesProvider>
        <ThemeToggle />
      </PreferencesProvider>,
    )

    expect(screen.getByRole('radiogroup', { name: 'Tema' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Tema claro' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Tema oscuro' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Tema del sistema' })).toBeInTheDocument()

    /* The collision that motivated the rename: "Sistema" is a section, and the
       theme group must not contain an option by that name. */
    expect(NAV_ITEMS.map((item) => item.label)).toContain('Sistema')
    expect(screen.queryByRole('radio', { name: 'Sistema' })).toBeNull()
  })
})
