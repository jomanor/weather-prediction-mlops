import { Layers, X } from 'lucide-react'
import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react'

import type { BasemapId, MapLayers } from '@/app/preferences'
import { cn } from '@/lib/cn'
import { RAMP_DOMAINS, RAMP_STEPS, rampToken, type RampName } from '@/lib/ramps'

export const SHEET_ID = 'map-layers-sheet'

export const LAYER_OPTIONS = [
  { key: 'hillshade', label: 'Relieve' },
  { key: 'radar', label: 'Radar de lluvia' },
] as const

export type LayerKey = (typeof LAYER_OPTIONS)[number]['key']

interface MapLayersControlProps {
  mapLayers: MapLayers
  onLayerChange: (key: LayerKey, checked: boolean) => void
  basemap: BasemapId
  onBasemapChange: (basemap: BasemapId) => void
  radarUnavailable: boolean
}

/**
 * The layer controls, once. `touch` swaps the desktop panel's 14 px rows for
 * 44 px tap rows; the content (both overlays, basemap and both legends) is the
 * same, so the phone never shows a different set of controls than the desktop.
 */
function LayerSettings({
  idPrefix,
  mapLayers,
  onLayerChange,
  basemap,
  onBasemapChange,
  radarUnavailable,
  touch = false,
}: MapLayersControlProps & { idPrefix: string; touch?: boolean }) {
  return (
    <>
      <div className="flex flex-col gap-1">
        {LAYER_OPTIONS.map((option) => (
          <label
            key={option.key}
            className={cn(
              'flex cursor-pointer items-center gap-2 text-xs text-fg-2 hover:text-fg',
              touch && 'min-h-11 text-title',
            )}
          >
            <input
              type="checkbox"
              className={cn('accent-[var(--accent)]', touch ? 'h-4 w-4' : 'h-3.5 w-3.5')}
              checked={mapLayers[option.key]}
              onChange={(event) => onLayerChange(option.key, event.target.checked)}
            />
            {option.label}
          </label>
        ))}
      </div>

      <label className={cn('label block', touch ? 'mt-3' : 'mt-2.5')} htmlFor={`${idPrefix}-basemap`}>
        Base
      </label>
      <select
        id={`${idPrefix}-basemap`}
        value={basemap}
        onChange={(event) => onBasemapChange(event.target.value as BasemapId)}
        className={cn(
          'mt-1 h-7 w-full rounded-[3px] border border-line bg-panel-2 px-1.5 text-[16px] leading-none text-fg-2 focus:border-accent focus:outline-none',
          /* 16 px on touch keeps iOS from zooming the viewport on focus. */
          !touch && 'sm:text-xs',
          /* 16 px stays: anything smaller makes iOS zoom on focus. */
          touch && 'h-11 text-[16px]',
        )}
      >
        <option value="positron">Claro (Carto)</option>
        <option value="dark">Oscuro (Carto)</option>
        <option value="topo">Relieve (OpenTopoMap)</option>
      </select>

      <div className={cn('border-t border-line', touch ? 'mt-3 pt-3' : 'mt-2.5 pt-2')}>
        <RampLegend
          name="temp"
          title="Temperatura"
          hint="icono = temp."
          format={(value) => `${value}°`}
        />
        <RampLegend
          name="wind"
          title="Viento (flecha)"
          hint="tamaño = vel."
          format={(value, index) => (index === 0 ? '0' : `${value} km/h`)}
          className="mt-2"
        />
      </div>

      {radarUnavailable ? (
        <p className="mt-1.5 text-[10px] leading-tight text-warn">Radar no disponible ahora.</p>
      ) : null}
    </>
  )
}

/** One data ramp as a four-step strip with its ends labelled. */
function RampLegend({
  name,
  title,
  hint,
  format,
  className,
}: {
  name: RampName
  title: string
  hint: string
  format: (value: number, index: number) => string
  className?: string
}) {
  const domain = RAMP_DOMAINS[name]
  const steps = RAMP_STEPS[name]
  return (
    <div className={className}>
      <div className="label mb-1">{title}</div>
      <div className="flex h-2 overflow-hidden rounded-[2px]" aria-hidden="true">
        {Array.from({ length: steps }, (_, step) => (
          <span key={step} className="flex-1" style={{ background: rampToken(name, step) }} />
        ))}
      </div>
      <div className="nums mt-0.5 flex justify-between text-[9px] text-fg-3">
        <span>{format(domain[0], 0)}</span>
        <span>{hint}</span>
        <span>{format(domain[1], 1)}</span>
      </div>
    </div>
  )
}

/** Desktop (>= `sm`): the fixed panel, unchanged. */
export function MapLayersPanel(props: MapLayersControlProps) {
  return (
    <div className="absolute right-2 top-2 z-10 hidden w-44 rounded-[3px] border border-line bg-panel p-2.5 sm:block">
      <div className="label mb-1.5">Capas</div>
      <LayerSettings idPrefix="layers-panel" {...props} />
    </div>
  )
}

/**
 * Phone (`< sm`): the panel collapses into a 44 px "Capas" button over the map
 * and a bottom sheet with the same content. The sheet stops above the bottom
 * navigation (`--nav-h` + `--safe-bottom`), scrolls if it outgrows the screen
 * and closes on backdrop tap, Escape or its close button — **not** on a layer
 * or basemap change: "radar on + dark basemap" must cost two taps, not four.
 *
 * Both surfaces are in the DOM and switched by CSS: `display: none` takes the
 * hidden one out of the tab order, so no control is ever unreachable and no
 * media query has to run before the first paint.
 */
export function MobileMapLayers({
  mapLayers,
  onLayerChange,
  basemap,
  onBasemapChange,
  radarUnavailable,
}: MapLayersControlProps) {
  const [open, setOpen] = useState(false)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const sheetRef = useRef<HTMLDivElement>(null)
  const closeRef = useRef<HTMLButtonElement>(null)

  /* Escape closes from anywhere, focus lands on the close button (a visible
     focus ring, not an invisible dialog root) and goes back to the button that
     opened the sheet, so the next Tab does not restart at the top of the page. */
  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      setOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    closeRef.current?.focus()
    const trigger = triggerRef.current
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      trigger?.focus()
    }
  }, [open])

  /* Keep Tab inside the sheet while it is open. */
  const onSheetKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Tab') return
    const root = sheetRef.current
    if (!root) return
    const focusable = Array.from(
      root.querySelectorAll<HTMLElement>('button, input, select, [href]'),
    ).filter((node) => !node.hasAttribute('disabled'))
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

  const close = () => setOpen(false)

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen(true)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-controls={open ? SHEET_ID : undefined}
        className="absolute right-2 top-2 z-10 inline-flex min-h-11 items-center gap-1.5 rounded-[3px] border border-line bg-panel px-3 text-title font-medium text-fg shadow-ui sm:hidden"
      >
        <Layers aria-hidden className="h-4 w-4 text-fg-2" />
        Capas
      </button>

      {open ? (
        <div
          className="fixed inset-0 z-50 flex items-end bg-fg/25 sm:hidden"
          onPointerDown={(event) => {
            if (event.target === event.currentTarget) close()
          }}
        >
          <div
            ref={sheetRef}
            id={SHEET_ID}
            role="dialog"
            aria-modal="true"
            aria-label="Capas del mapa"
            onKeyDown={onSheetKeyDown}
            className="animate-in w-full overflow-y-auto rounded-t-[var(--radius-panel)] border-t border-line bg-panel px-4 pt-3 pb-3 shadow-sheet"
            /* Clear of the fixed bottom nav and the iOS home indicator: the nav
               already owns `--safe-bottom`, so the sheet ends where it starts. */
            style={{
              marginBottom: 'calc(var(--nav-h) + var(--safe-bottom))',
              maxHeight: 'calc(100svh - var(--nav-h) - var(--safe-bottom) - 3.5rem)',
            }}
          >
            <div className="mb-1 flex items-center justify-between">
              <h2 className="text-title font-semibold text-fg">Capas</h2>
              <button
                ref={closeRef}
                type="button"
                onClick={close}
                aria-label="Cerrar capas"
                className="-mr-2 inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-[3px] text-fg-3 hover:text-fg"
              >
                <X aria-hidden className="h-4 w-4" />
              </button>
            </div>

            <LayerSettings
              idPrefix="layers-sheet"
              touch
              mapLayers={mapLayers}
              /* A toggle leaves the sheet open: the map is behind it, and the
                 next change is usually the other check or the basemap. */
              onLayerChange={onLayerChange}
              basemap={basemap}
              onBasemapChange={onBasemapChange}
              radarUnavailable={radarUnavailable}
            />
          </div>
        </div>
      ) : null}
    </>
  )
}
