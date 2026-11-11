import { ChartLine, X } from 'lucide-react'
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from 'react'

import { useCities } from '@/api/queries'
import { Button } from '@/components/ui/Button'
import { EmptyState, ErrorState, LoadingBlock } from '@/components/ui/Feedback'
import { Segmented } from '@/components/ui/Segmented'
import { DiurnalHeatmap } from '@/features/analytics/components/DiurnalHeatmap'
import { WindRose } from '@/features/analytics/components/WindRose'
import { useAnalyticsDiurnal, useAnalyticsWindRose } from '@/features/analytics/queries'
import { aemetCoveredNames } from '@/lib/aemet-coverage'
import { formatInteger } from '@/lib/format'

const DAY_OPTIONS = [30, 90, 180] as const
type DayRange = (typeof DAY_OPTIONS)[number]

const DAY_SEGMENTS = DAY_OPTIONS.map((value) => ({ value, label: `${value} d` }))

/** Every control the Tab trap cycles, in DOM order. */
const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'

/**
 * The two climate views the deleted Analítica tab kept (audit 16): the diurnal
 * heatmap and the wind rose, for the station already selected on the page.
 *
 * Why a modal and not a panel: they will become tiles of a user-composable
 * dashboard, so they must not own a page. The views themselves stay standalone
 * (`DiurnalHeatmap`, `WindRose` take plain props and know nothing about this
 * dialog), and the queries only mount while it is open — opening a station page
 * must not fire analytics requests nobody asked for.
 *
 * Modal contract: named by its title, Esc + backdrop + close button all close,
 * focus lands on the close button and returns to the trigger, Tab stays inside,
 * the page behind does not scroll while it is open, and the enter animation is
 * opacity + a 0.96 scale over `--motion-base` (reduced motion keeps the fade
 * and drops the scale).
 */
export function ClimateViewsDialog({ city }: { city: string }) {
  const [open, setOpen] = useState(false)
  const [days, setDays] = useState<DayRange>(90)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const closeRef = useRef<HTMLButtonElement>(null)
  const titleId = useId()
  const panelId = useId()

  /* Escape closes from anywhere; the scroll lock is taken here and always given
     back (a leaked `overflow: hidden` locks the whole app, not just the modal). */
  useEffect(() => {
    if (!open) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      setOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    closeRef.current?.focus()

    return () => {
      window.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = previousOverflow
      triggerRef.current?.focus()
    }
  }, [open])

  const close = useCallback(() => setOpen(false), [])

  /* Keep Tab inside the dialog: a visible focus ring, never the browser's
     focus escaping to the page behind the overlay. */
  const onPanelKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Tab') return
    const root = panelRef.current
    if (!root) return
    const focusable = Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
      (node) => !node.hasAttribute('disabled'),
    )
    if (focusable.length < 2) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  return (
    <>
      <Button
        ref={triggerRef}
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
      >
        Clima
      </Button>

      {open ? (
        <div
          className="dialog-backdrop fixed inset-0 z-50 flex items-center justify-center bg-fg/25 p-4"
          onPointerDown={(event) => {
            if (event.target === event.currentTarget) close()
          }}
        >
          <div
            ref={panelRef}
            id={panelId}
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            onKeyDown={onPanelKeyDown}
            className="dialog-enter flex max-h-[calc(100svh-2rem)] w-full max-w-3xl flex-col rounded-panel border border-line bg-panel shadow-ui"
          >
            <header className="flex items-start justify-between gap-3 border-b border-line px-4 py-3">
              <div className="min-w-0">
                <h2 id={titleId} className="text-title font-semibold leading-tight text-fg">
                  Clima de la estación
                </h2>
                <p className="mt-1 text-body leading-tight text-fg-3">
                  {city} · últimos {days} días
                </p>
              </div>
              <button
                ref={closeRef}
                type="button"
                onClick={close}
                aria-label="Cerrar clima"
                className="-mr-2 inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-control text-fg-3 hover:text-fg"
              >
                <X aria-hidden className="h-4 w-4" />
              </button>
            </header>

            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2">
              <span className="label">Ventana</span>
              <Segmented
                value={days}
                onChange={setDays}
                options={DAY_SEGMENTS}
                label="Ventana de la climatología"
              />
            </div>

            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-3">
              <ClimateViews city={city} days={days} />
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}

/**
 * The dialog's content. Mounted only while the dialog is open, so the analytics
 * queries are lazy without teaching the hooks an `enabled` flag.
 */
function ClimateViews({ city, days }: { city: string; days: DayRange }) {
  const { data: cityList } = useCities()
  const diurnal = useAnalyticsDiurnal(days)
  const wind = useAnalyticsWindRose(city, days)

  /* AEMET only covers Spain, and `/analytics/diurnal` returns the server-side
     canonical list: filter before the matrix is built so the ramp domain and
     the row set agree. */
  const coveredNames = useMemo(() => aemetCoveredNames(cityList ?? []), [cityList])
  const cells = useMemo(
    () =>
      (diurnal.data?.cells ?? []).filter(
        (cell) => cityList === undefined || coveredNames.has(cell.city),
      ),
    [diurnal.data, cityList, coveredNames],
  )

  return (
    <>
      <section>
        <div className="flex items-baseline justify-between gap-2">
          <h3 className="label">Ciclo diurno por estación</h3>
          {diurnal.data ? (
            <span className="nums text-badge text-fg-3">{formatInteger(cells.length)} celdas</span>
          ) : null}
        </div>
        {diurnal.isLoading ? (
          <LoadingBlock />
        ) : diurnal.isError ? (
          <ErrorState
            title="No se pudo cargar el ciclo diurno"
            description={diurnal.error.message}
            action={
              <Button size="sm" onClick={() => diurnal.refetch()}>
                Reintentar
              </Button>
            }
          />
        ) : !cells.length ? (
          <EmptyState
            icon={ChartLine}
            title="Sin observaciones en la ventana"
            description={`No hay datos horarios en los últimos ${days} días. Prueba una ventana más amplia.`}
          />
        ) : (
          <div className="mt-2">
            <DiurnalHeatmap cells={cells} />
          </div>
        )}
      </section>

      <section className="border-t border-line pt-3">
        <h3 className="label">Rosa de los vientos</h3>
        <p className="mt-1 text-body leading-tight text-fg-3">
          Dirección de procedencia · últimos {days} días
        </p>
        {wind.isLoading ? (
          <LoadingBlock />
        ) : wind.isError ? (
          <ErrorState
            title="No se pudo cargar la rosa de los vientos"
            description={wind.error.message}
            action={
              <Button size="sm" onClick={() => wind.refetch()}>
                Reintentar
              </Button>
            }
          />
        ) : !wind.data?.sectors.length ? (
          <EmptyState
            title="Sin datos de viento"
            description="No hay observaciones de dirección y velocidad en la ventana seleccionada."
          />
        ) : (
          <div className="mt-2">
            <WindRose sectors={wind.data.sectors} />
          </div>
        )}
      </section>
    </>
  )
}
