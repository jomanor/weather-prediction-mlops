import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import {
  useEffect,
  useId,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react'

import type { CurrentWeather, Prediction } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { isEditableTarget } from '@/components/shell/shortcuts'
import { Button } from '@/components/ui/Button'
import { Readout } from '@/components/ui/Readout'
import {
  beginSwipe,
  isInteractiveTarget,
  resolveSwipe,
  trackSwipe,
  type SwipeStart,
} from '@/features/overview/sheet'
import {
  compassPoint,
  formatPercent,
  formatPrecipitation,
  formatPressure,
  formatTemperature,
  formatWind,
  isNum,
} from '@/lib/format'
import { describeWeather, toneClass } from '@/lib/weather'

interface StationSheetProps {
  city: string | null
  station?: CurrentWeather
  prediction?: Prediction
  position: { index: number; total: number } | null
  onPrev: () => void
  onNext: () => void
  onClose: () => void
}

/* The sheet sits directly on top of the bottom navigation: both offsets read
   the same tokens, so the safe-area inset is counted exactly once. */
const SHEET_OFFSET = 'calc(var(--nav-h, 3.25rem) + var(--safe-bottom, 0px))'
/* Never taller than the viewport minus the nav, the home indicator and a
   finger's worth of map; content beyond that scrolls inside the sheet.
   `dvh` follows the browser chrome and the soft keyboard, `vh` does not. */
const SHEET_MAX_HEIGHT =
  'calc(100dvh - var(--nav-h, 3.25rem) - var(--safe-bottom, 0px) - 1.5rem)'

/**
 * U6 mobile station detail. A non-modal bottom sheet that sits above the bottom
 * navigation (which keeps working) and swipes horizontally between stations.
 * Escape and the close button dismiss it; the prev/next buttons keep it
 * keyboard reachable.
 */
export function StationSheet({
  city,
  station,
  prediction,
  position,
  onPrev,
  onNext,
  onClose,
}: StationSheetProps) {
  const { units } = usePreferences()
  const headingId = useId()
  const sheetRef = useRef<HTMLDivElement>(null)
  const gesture = useRef<{ pointerId: number; start: SwipeStart } | null>(null)
  const open = city !== null

  /* Non-modal, so focus may legitimately sit outside the sheet: Escape is a
     window-level dismissal. It stays out of fields, whose own Escape (the
     station search) must keep closing only the field's popup. */
  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || isEditableTarget(event.target)) return
      onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [open, onClose])

  /* Focus enters the sheet on open and returns to the row/pin that opened it,
     but only while that trigger is still in the document. */
  useEffect(() => {
    if (!open) return
    const trigger = document.activeElement as HTMLElement | null
    sheetRef.current?.focus()
    return () => {
      if (!trigger || trigger === document.body || !document.contains(trigger)) return
      trigger.focus()
    }
  }, [open])

  if (!city) return null

  const weather = station ? describeWeather(station.weather_code) : null
  const WeatherIcon = weather?.icon

  const releasePointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    try {
      event.currentTarget.releasePointerCapture(event.pointerId)
    } catch {
      /* never captured (jsdom, or the pointer already went away) */
    }
  }

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    /* Primary pointer (no pinch), left button, and never a control that owns
       its own gestures. */
    if (!event.isPrimary || event.button !== 0) return
    if (isInteractiveTarget(event.target)) return
    gesture.current = {
      pointerId: event.pointerId,
      start: beginSwipe(event.clientX, event.clientY),
    }
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      /* jsdom has no pointer capture */
    }
  }

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const active = gesture.current
    if (!active || active.pointerId !== event.pointerId) return
    trackSwipe(active.start, event.clientX, event.clientY)
  }

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const active = gesture.current
    if (!active || active.pointerId !== event.pointerId) return
    gesture.current = null
    releasePointer(event)
    const direction = resolveSwipe(active.start, event.clientX, event.clientY)
    if (direction === 'next') onNext()
    else if (direction === 'prev') onPrev()
  }

  /* Touch is cancelled by the OS (notification, gesture, keyboard): drop the
     gesture instead of letting a stale start pair with a later pointer up. */
  const onPointerCancel = (event: ReactPointerEvent<HTMLDivElement>) => {
    const active = gesture.current
    if (!active || active.pointerId !== event.pointerId) return
    gesture.current = null
    releasePointer(event)
  }

  /* Scoped to the sheet, not the window: with the sheet open the map must keep
     its own arrow-key panning. */
  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (isEditableTarget(event.target)) return
    if (event.key === 'ArrowLeft') {
      event.preventDefault()
      onPrev()
    } else if (event.key === 'ArrowRight') {
      event.preventDefault()
      onNext()
    }
  }

  return (
    <div
      ref={sheetRef}
      role="dialog"
      aria-modal="false"
      aria-labelledby={headingId}
      tabIndex={-1}
      className="fixed inset-x-0 z-30 flex flex-col animate-in border-t border-line bg-panel shadow-[0_-8px_24px_rgb(0_0_0/0.12)] outline-none md:hidden"
      style={{ bottom: SHEET_OFFSET, maxHeight: SHEET_MAX_HEIGHT, touchAction: 'pan-y' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onKeyDown={onKeyDown}
    >
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-line px-3 py-1.5">
        <div className="flex min-w-0 items-center gap-2">
          <span aria-hidden className="h-1 w-5 shrink-0 rounded-full bg-line-strong" />
          <h2 id={headingId} className="truncate text-sm font-semibold text-fg">
            {city}
          </h2>
          {position ? (
            <span
              aria-live="polite"
              aria-atomic="true"
              className="nums shrink-0 text-[10px] text-fg-3"
            >
              <span className="sr-only">
                Estación {position.index} de {position.total}
              </span>
              <span aria-hidden>
                {position.index}/{position.total}
              </span>
            </span>
          ) : null}
        </div>
        {/* 44 px targets: the sheet header is the only control cluster on the
            detail surface, so it is sized for touch rather than for density. */}
        <div className="flex shrink-0 items-center gap-0.5">
          <Button
            size="icon"
            className="h-11 w-11"
            aria-label="Estación anterior"
            onClick={onPrev}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            className="h-11 w-11"
            aria-label="Estación siguiente"
            onClick={onNext}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          <Button size="icon" className="h-11 w-11" aria-label="Cerrar detalle" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="min-h-0 overflow-y-auto">
        <div className="flex items-center gap-3 px-4 py-3">
          {WeatherIcon ? (
            <WeatherIcon className={toneClass[weather!.tone]} strokeWidth={1.5} />
          ) : (
            <span
              aria-hidden
              className="h-6 w-6 rounded-full border border-dashed border-line-strong"
            />
          )}
          <div className="min-w-0">
            <div className="nums text-2xl font-medium leading-none text-fg">
              {formatTemperature(station?.temperature ?? null, units)}
            </div>
            <div className="mt-1 truncate text-[11px] text-fg-3">
              {weather?.label ?? 'esperando ingesta'}
            </div>
          </div>
          {prediction ? (
            <div className="ml-auto shrink-0 text-right">
              <div className="label">Modelo +{prediction.horizon_hours} h</div>
              <div className="nums mt-1 text-sm text-model">
                {formatTemperature(prediction.predicted_temperature, units)}
              </div>
            </div>
          ) : null}
        </div>

        <div className="grid grid-cols-3 gap-x-4 gap-y-3 border-t border-line px-4 py-3">
          <Readout
            size="sm"
            label="Sensación"
            value={formatTemperature(station?.apparent_temperature ?? null, units)}
          />
          <Readout size="sm" label="Humedad" value={formatPercent(station?.humidity ?? null)} />
          <Readout
            size="sm"
            label="Viento"
            value={formatWind(station?.wind_speed ?? null, units)}
            hint={
              isNum(station?.wind_direction)
                ? compassPoint(station!.wind_direction as number)
                : undefined
            }
          />
          <Readout
            size="sm"
            label="Presión"
            value={formatPressure(station?.pressure ?? null, units)}
          />
          <Readout
            size="sm"
            label="Precipitación"
            value={formatPrecipitation(station?.precipitation ?? null, units)}
          />
          <Readout size="sm" label="Nubosidad" value={formatPercent(station?.cloud_cover ?? null)} />
        </div>
      </div>
    </div>
  )
}
