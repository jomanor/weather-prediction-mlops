import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { PreferencesProvider } from '@/app/preferences'
import { AppRoutes } from '@/app/routes'
import { DashboardPage } from '@/features/dashboard/DashboardPage'
import { DASHBOARD_KEY, readDashboard } from '@/features/dashboard/storage'

/**
 * Panel: the dashboard the user composes, stored in this browser only.
 *
 * The transport is mocked, not the hooks, so the real query hooks run and every
 * request a card would make is recorded — that is what proves a card only asks
 * for what is on the panel. The chart is mocked (its own suite covers it); the
 * contract here is *what the card hands it*.
 */
const mocks = vi.hoisted(() => ({
  paths: [] as string[],
  /* true → every request hangs, so the card stays in its loading state. */
  pending: false,
  failDiurnal: false,
  chartPoints: [] as Array<{ observed_at: string }>,
}))

const NOW = Date.now()

function point(city: string, hoursAgo: number) {
  return {
    city,
    latitude: 40.42,
    longitude: -3.7,
    temperature: 20,
    apparent_temperature: 19,
    humidity: 40,
    pressure: 1015,
    wind_speed: 10,
    wind_direction: 180,
    precipitation: 0,
    cloud_cover: 10,
    weather_code: 0,
    is_day: 1,
    observed_at: new Date(NOW - hoursAgo * 3_600_000).toISOString(),
  }
}

function payloadFor(path: string) {
  if (path === '/cities') {
    return [
      { name: 'Madrid', latitude: 40.42, longitude: -3.7 },
      { name: 'Vigo', latitude: 42.24, longitude: -8.72 },
    ]
  }
  if (path.startsWith('/analytics/diurnal')) {
    if (mocks.failDiurnal) throw new Error('502 Bad Gateway')
    return {
      days: 90,
      generated_at: null,
      cells: [
        { city: 'Madrid', hour: 0, tmean: 12, n: 30 },
        { city: 'Vigo', hour: 13, tmean: 19, n: 30 },
      ],
    }
  }
  if (path.startsWith('/analytics/wind-rose')) {
    return {
      city: 'Madrid',
      days: 90,
      generated_at: null,
      sectors: [{ sector: 0, count: 10, mean_speed: 20 }],
    }
  }
  if (path.startsWith('/weather/series')) {
    const city = new URLSearchParams(path.split('?')[1]).get('cities') ?? 'Madrid'
    const points = [point(city, 2), point(city, 1)]
    return {
      hours: 48,
      count: points.length,
      cities: [
        {
          city,
          count: points.length,
          latest_timestamp: points[points.length - 1].observed_at,
          points,
        },
      ],
    }
  }
  throw new Error(`unexpected request: ${path}`)
}

vi.mock('@/api/client', () => ({
  apiGet: vi.fn(async (path: string) => {
    mocks.paths.push(path.split('?')[0])
    if (mocks.pending) return new Promise(() => {})
    return payloadFor(path)
  }),
  apiGetWithDataAge: vi.fn(),
  apiSend: vi.fn(),
}))

/* Recharts needs a ResizeObserver and its own suite pins the drawing; here the
   assertion is which points the card passes down. */
vi.mock('@/components/charts/HistoryChart', () => ({
  HistoryChart: ({ points }: { points: Array<{ observed_at: string }> }) => {
    mocks.chartPoints = points
    return <div data-testid="history-chart" />
  },
}))

function seed(widgets: unknown[]): void {
  localStorage.setItem(DASHBOARD_KEY, JSON.stringify({ version: 1, widgets }))
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/panel']}>
        <PreferencesProvider>
          <DashboardPage />
        </PreferencesProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

/** The card that owns one heading — the `<li>` of the ordered layout. */
function card(name: string): HTMLElement {
  const heading = screen.getByRole('heading', { name })
  const item = heading.closest('li')
  if (!item) throw new Error(`no card for ${name}`)
  return item as HTMLElement
}

/** The panel's own cards — never the `<li>`s a widget renders inside itself. */
function cardItems(): HTMLElement[] {
  return Array.from(
    screen.getByRole('list', { name: 'Vistas del panel' }).children,
  ) as HTMLElement[]
}

/** Headings of the cards, in render order. */
function order(): string[] {
  return cardItems().map(
    (item) => within(item).getByRole('heading', { level: 2 }).textContent ?? '',
  )
}

async function addWidget(title: string): Promise<void> {
  await userEvent.click(screen.getByRole('button', { name: 'Añadir vista' }))
  await userEvent.click(screen.getByRole('button', { name: title }))
}

beforeEach(() => {
  localStorage.clear()
  mocks.paths.length = 0
  mocks.pending = false
  mocks.failDiurnal = false
  mocks.chartPoints = []
})

describe('Panel empty state', () => {
  it('explains that the layout lives in this browser and offers one action', () => {
    renderPage()

    expect(screen.getByRole('heading', { level: 1, name: 'Panel' })).toBeInTheDocument()
    expect(screen.getByText('Tu panel está vacío')).toBeInTheDocument()
    /* No account, no server: the one thing a user must know before building it. */
    expect(
      screen.getByText(/Se guardan solo en este navegador: no hay cuenta ni copia en el servidor/),
    ).toBeInTheDocument()

    expect(screen.getAllByRole('button', { name: 'Añadir la primera vista' })).toHaveLength(1)
    /* Nothing is on the panel, so nothing is requested. */
    expect(mocks.paths).toEqual([])
  })

  it('offers one entry per widget type, behind a labelled disclosure', async () => {
    renderPage()

    const toggle = screen.getByRole('button', { name: 'Añadir vista' })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await userEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')

    /* One entry per registered type: adding a tile later is a registry entry. */
    for (const title of ['Ciclo diurno', 'Rosa de los vientos', 'Serie de la estación']) {
      expect(screen.getByRole('button', { name: title })).toBeInTheDocument()
    }
  })

  it('keeps the empty state when no stored widget is usable', () => {
    localStorage.setItem(DASHBOARD_KEY, '{"version":1,"widgets":[{"type":"radar"}]}')
    renderPage()

    expect(screen.getByText('Tu panel está vacío')).toBeInTheDocument()
    expect(screen.queryByRole('list', { name: 'Vistas del panel' })).toBeNull()
  })
})

describe('Panel adds and removes widgets', () => {
  it('adds each of the three widget types from the chooser', async () => {
    renderPage()

    await addWidget('Ciclo diurno')
    expect(screen.getByRole('heading', { name: 'Ciclo diurno' })).toBeInTheDocument()

    await addWidget('Rosa de los vientos')
    await addWidget('Serie de la estación')

    expect(order()).toEqual(['Ciclo diurno', 'Rosa de los vientos', 'Serie de la estación'])
    /* Each card renders the existing component, not a copy of it. */
    expect(await screen.findByRole('img', { name: /Rosa de los vientos/ })).toBeInTheDocument()
    expect(await screen.findByTestId('history-chart')).toBeInTheDocument()

    /* The layout is already persisted, ids included. */
    const stored = readDashboard()
    expect(stored.map((widget) => widget.type)).toEqual(['diurnal', 'wind-rose', 'series'])
    for (const widget of stored) expect(widget.id).not.toBe('')
  })

  it('removes a widget and keeps the rest', async () => {
    seed([
      { id: 'a', type: 'diurnal', days: 90 },
      { id: 'b', type: 'wind-rose', city: 'Vigo', days: 30 },
    ])
    renderPage()

    await userEvent.click(screen.getByRole('button', { name: 'Quitar Ciclo diurno' }))

    expect(order()).toEqual(['Rosa de los vientos'])
    expect(readDashboard().map((widget) => widget.id)).toEqual(['b'])
  })

  it('survives a remount: the layout comes back from localStorage', async () => {
    const first = renderPage()
    await addWidget('Ciclo diurno')
    await addWidget('Serie de la estación')
    first.unmount()

    renderPage()

    expect(order()).toEqual(['Ciclo diurno', 'Serie de la estación'])
  })
})

describe('Panel reordering', () => {
  beforeEach(() => {
    seed([
      { id: 'a', type: 'diurnal', days: 90 },
      { id: 'b', type: 'series', city: 'Vigo', hours: 48 },
    ])
  })

  it('moves a card down and up with the buttons, persisted in the new order', async () => {
    renderPage()

    await userEvent.click(screen.getByRole('button', { name: 'Bajar Ciclo diurno' }))
    expect(order()).toEqual(['Serie de la estación', 'Ciclo diurno'])
    expect(readDashboard().map((widget) => widget.id)).toEqual(['b', 'a'])

    await userEvent.click(screen.getByRole('button', { name: 'Subir Ciclo diurno' }))
    expect(order()).toEqual(['Ciclo diurno', 'Serie de la estación'])
  })

  it('reorders from the keyboard alone', async () => {
    renderPage()

    const down = screen.getByRole('button', { name: 'Bajar Ciclo diurno' })
    down.focus()
    expect(down).toHaveFocus()
    await userEvent.keyboard('{Enter}')

    expect(order()).toEqual(['Serie de la estación', 'Ciclo diurno'])

    const up = screen.getByRole('button', { name: 'Subir Ciclo diurno' })
    up.focus()
    await userEvent.keyboard('{Enter}')

    expect(order()).toEqual(['Ciclo diurno', 'Serie de la estación'])
  })

  it('disables the move buttons at the ends of the list', () => {
    renderPage()

    expect(screen.getByRole('button', { name: 'Subir Ciclo diurno' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Bajar Ciclo diurno' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Subir Serie de la estación' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Bajar Serie de la estación' })).toBeDisabled()
  })

  it('names every icon-only button after the card it acts on', () => {
    renderPage()

    for (const label of [
      'Subir Ciclo diurno',
      'Bajar Ciclo diurno',
      'Quitar Ciclo diurno',
      'Subir Serie de la estación',
      'Bajar Serie de la estación',
      'Quitar Serie de la estación',
    ]) {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument()
    }
  })
})

describe('Panel widget states', () => {
  it('shows the loading primitive while a widget query is in flight', () => {
    mocks.pending = true
    seed([{ id: 'a', type: 'diurnal', days: 90 }])
    renderPage()

    const status = within(card('Ciclo diurno')).getByRole('status')
    expect(status).toHaveTextContent('Cargando…')
    /* A loading card is not an empty one. */
    expect(screen.queryByText('Sin observaciones en la ventana')).not.toBeInTheDocument()
  })

  it('shows ErrorState with Reintentar and does not take the page down', async () => {
    mocks.failDiurnal = true
    seed([
      { id: 'a', type: 'diurnal', days: 90 },
      { id: 'b', type: 'wind-rose', city: 'Madrid', days: 90 },
    ])
    renderPage()

    /* `useApiQuery` retries once (its own default, so the page is mounted as in
       production); the 1 s backoff is why this waits longer than `findByText`
       does by default. */
    expect(
      await screen.findByText('No se pudo cargar el ciclo diurno', {}, { timeout: 3000 }),
    ).toBeInTheDocument()
    expect(within(card('Ciclo diurno')).getByRole('button', { name: 'Reintentar' })).toBeEnabled()

    /* The page and the healthy card are untouched by the failed one. */
    expect(screen.getByRole('heading', { level: 2, name: 'Rosa de los vientos' })).toBeInTheDocument()
    expect(await screen.findByRole('img', { name: /Rosa de los vientos/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Quitar Ciclo diurno' })).toBeEnabled()
  })

  it('passes the station series, oldest first, to the chart', async () => {
    seed([{ id: 'a', type: 'series', city: 'Vigo', hours: 168 }])
    renderPage()

    await screen.findByTestId('history-chart')
    expect(mocks.chartPoints.map((entry) => entry.observed_at)).toEqual([
      point('Vigo', 2).observed_at,
      point('Vigo', 1).observed_at,
    ])
    expect(mocks.paths).toContain('/weather/series')
    /* The card asks for the window its control shows. */
    expect(screen.getByRole('radio', { name: '7 d' })).toHaveAttribute('aria-checked', 'true')
  })

  it('requests nothing for a widget type that is not on the panel', async () => {
    seed([{ id: 'a', type: 'diurnal', days: 90 }])
    renderPage()

    await screen.findByRole('rowheader', { name: 'Madrid' })

    expect(mocks.paths).toContain('/analytics/diurnal')
    expect(mocks.paths).not.toContain('/analytics/wind-rose')
    expect(mocks.paths).not.toContain('/weather/series')
  })

  it('re-queries with the control it was given', async () => {
    seed([{ id: 'a', type: 'wind-rose', city: 'Madrid', days: 90 }])
    renderPage()

    await screen.findByRole('img', { name: /Rosa de los vientos/ })
    await userEvent.click(screen.getByRole('radio', { name: '180 d' }))

    expect(readDashboard()[0].days).toBe(180)
    /* The select is the station control: choosing Vigo re-requests the rose. */
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Estación' }), 'Vigo')
    expect(readDashboard()[0].city).toBe('Vigo')
  })
})

describe('Panel reset', () => {
  it('confirms inline, then clears the stored layout', async () => {
    seed([
      { id: 'a', type: 'diurnal', days: 90 },
      { id: 'b', type: 'series', city: 'Madrid', hours: 48 },
    ])
    renderPage()

    await userEvent.click(screen.getByRole('button', { name: 'Restablecer' }))
    expect(screen.getByText(/Se quitarán las 2 vistas de este navegador/)).toBeInTheDocument()
    /* Inline, never `window.confirm`. */
    const confirmSpy = vi.spyOn(window, 'confirm')
    expect(confirmSpy).not.toHaveBeenCalled()

    await userEvent.click(screen.getByRole('button', { name: 'Sí, restablecer' }))

    expect(screen.getByText('Tu panel está vacío')).toBeInTheDocument()
    expect(readDashboard()).toEqual([])
  })

  it('can be cancelled without losing the layout', async () => {
    seed([{ id: 'a', type: 'diurnal', days: 90 }])
    renderPage()

    await userEvent.click(screen.getByRole('button', { name: 'Restablecer' }))
    await userEvent.click(screen.getByRole('button', { name: 'Cancelar' }))

    expect(order()).toEqual(['Ciclo diurno'])
    expect(readDashboard()).toHaveLength(1)
  })
})

describe('Panel cap', () => {
  it('stops at twelve widgets and says so', () => {
    seed(
      Array.from({ length: 12 }, (_, index) => ({ id: `w${index}`, type: 'diurnal', days: 90 })),
    )
    renderPage()

    expect(cardItems()).toHaveLength(12)
    expect(screen.getByRole('button', { name: 'Añadir vista' })).toBeDisabled()
    expect(screen.getByText(/Máximo de 12 vistas/)).toBeInTheDocument()
  })
})

describe('Panel surface', () => {
  it('is the fourth nav item and /panel renders it', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <PreferencesProvider>
          <MemoryRouter initialEntries={['/panel']}>
            <AppRoutes />
          </MemoryRouter>
        </PreferencesProvider>
      </QueryClientProvider>,
    )

    /* Both navs (desktop bar and the fixed phone bar) derive from the registry. */
    const navs = screen.getAllByRole('navigation', { name: 'Secciones' })
    expect(navs).toHaveLength(2)
    for (const nav of navs) {
      expect(within(nav).getAllByRole('link')).toHaveLength(4)
      expect(within(nav).getAllByRole('link')[3]).toHaveAccessibleName('Panel')
    }

    expect(await screen.findByRole('heading', { level: 1, name: 'Panel' })).toBeInTheDocument()
  })
})
