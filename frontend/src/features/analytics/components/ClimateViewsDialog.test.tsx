import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ClimateViewsDialog } from '@/features/analytics/components/ClimateViewsDialog'

/**
 * The station page's climate modal (audit 16): the two views the deleted
 * Analítica tab kept, behind a dialog that must be operable by keyboard and
 * must not fetch anything until it is opened.
 */
const mocks = vi.hoisted(() => ({
  diurnalDays: [] as number[],
  wind: [] as Array<{ city: string; days: number }>,
}))

const WIND_ROSE = {
  city: 'Madrid',
  days: 90,
  generated_at: null,
  sectors: [{ sector: 0, count: 10, mean_speed: 20 }],
}

const DIURNAL = {
  days: 90,
  generated_at: null,
  cells: [
    { city: 'Madrid', hour: 0, tmean: 12, n: 30 },
    { city: 'Vigo', hour: 13, tmean: 19, n: 30 },
  ],
}

vi.mock('@/api/queries', () => ({
  useCities: () => ({
    data: [
      { name: 'Madrid', latitude: 40.42, longitude: -3.7 },
      { name: 'Vigo', latitude: 42.24, longitude: -8.72 },
    ],
  }),
}))

vi.mock('@/features/analytics/queries', () => ({
  useAnalyticsDiurnal: (days: number) => {
    mocks.diurnalDays.push(days)
    return {
      data: DIURNAL,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    }
  },
  useAnalyticsWindRose: (city: string, days: number) => {
    mocks.wind.push({ city, days })
    return {
      data: WIND_ROSE,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    }
  },
}))

beforeEach(() => {
  mocks.diurnalDays.length = 0
  mocks.wind.length = 0
  document.body.style.overflow = ''
})

function renderDialog() {
  return render(<ClimateViewsDialog city="Madrid" />)
}

const trigger = () => screen.getByRole('button', { name: 'Clima' })
const dialog = () => screen.getByRole('dialog', { name: 'Clima de la estación' })

describe('ClimateViewsDialog closed', () => {
  it('renders the trigger only, and fetches nothing', () => {
    renderDialog()

    expect(trigger()).toHaveAttribute('aria-expanded', 'false')
    expect(trigger()).toHaveAttribute('aria-haspopup', 'dialog')
    expect(screen.queryByRole('dialog')).toBeNull()
    /* Opening a station page must not fire analytics nobody asked for. */
    expect(mocks.diurnalDays).toEqual([])
    expect(mocks.wind).toEqual([])
  })
})

describe('ClimateViewsDialog open', () => {
  it('is a labelled modal holding the heatmap and the wind rose, and focuses its close button', async () => {
    renderDialog()
    await userEvent.click(trigger())

    const modal = dialog()
    /* Named by its title, so a screen reader announces what opened. */
    expect(within(modal).getByRole('heading', { level: 2, name: 'Clima de la estación' })).toBeInTheDocument()
    expect(modal).toHaveAttribute('aria-modal', 'true')
    expect(screen.getByRole('button', { name: 'Cerrar clima' })).toHaveFocus()
    expect(trigger()).toHaveAttribute('aria-expanded', 'true')

    /* The selected station and the window are stated, not implied. */
    expect(within(modal).getByText('Madrid · últimos 90 días')).toBeInTheDocument()

    /* Both kept views render as their own standalone components. */
    expect(within(modal).getAllByRole('rowheader').map((row) => row.textContent)).toEqual([
      'Madrid',
      'Vigo',
    ])
    expect(within(modal).getByRole('img', { name: /Rosa de los vientos/ })).toBeInTheDocument()

    /* The enter animation is the token-gated utility, never `transition-all`. */
    expect(modal.className).toContain('dialog-enter')
    expect(modal.className).not.toContain('transition-all')
  })

  it('re-queries both views when the window changes', async () => {
    renderDialog()
    await userEvent.click(trigger())

    await userEvent.click(screen.getByRole('radio', { name: '180 d' }))

    expect(mocks.diurnalDays.at(-1)).toBe(180)
    expect(mocks.wind.at(-1)).toEqual({ city: 'Madrid', days: 180 })
    expect(screen.getByText('Madrid · últimos 180 días')).toBeInTheDocument()
  })

  it('closes on Escape and returns focus to the trigger', async () => {
    renderDialog()
    await userEvent.click(trigger())

    await userEvent.keyboard('{Escape}')

    expect(screen.queryByRole('dialog')).toBeNull()
    expect(trigger()).toHaveFocus()
  })

  it('closes on the close button and returns focus to the trigger', async () => {
    renderDialog()
    await userEvent.click(trigger())

    await userEvent.click(screen.getByRole('button', { name: 'Cerrar clima' }))

    expect(screen.queryByRole('dialog')).toBeNull()
    expect(trigger()).toHaveFocus()
  })

  it('closes on a backdrop tap but not on a tap inside the dialog', async () => {
    renderDialog()
    await userEvent.click(trigger())

    await userEvent.click(screen.getByText('Madrid · últimos 90 días'))
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    const backdrop = dialog().parentElement as HTMLElement
    fireEvent.pointerDown(backdrop)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('keeps Tab inside the dialog', async () => {
    renderDialog()
    await userEvent.click(trigger())

    /* The last control in DOM order wraps back to the first, so the focus ring
       never lands on the page behind the overlay. */
    const last = screen.getByRole('radio', { name: '180 d' })
    last.focus()
    await userEvent.keyboard('{Tab}')
    expect(screen.getByRole('button', { name: 'Cerrar clima' })).toHaveFocus()

    await userEvent.keyboard('{Shift>}{Tab}{/Shift}')
    expect(last).toHaveFocus()
  })

  it('locks the page scroll only while open', async () => {
    document.body.style.overflow = 'auto'
    renderDialog()
    await userEvent.click(trigger())
    expect(document.body.style.overflow).toBe('hidden')

    await userEvent.keyboard('{Escape}')
    expect(document.body.style.overflow).toBe('auto')
  })
})
