import { ArrowDown, ArrowUp, X } from 'lucide-react'
import { useId, useMemo } from 'react'

import { useCities } from '@/api/queries'
import { Button } from '@/components/ui/Button'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Segmented } from '@/components/ui/Segmented'
import {
  DAY_OPTIONS,
  HOUR_OPTIONS,
  type DashboardWidget,
  type WidgetConfig,
  type WidgetDays,
  type WidgetHours,
} from '@/features/dashboard/types'
import {
  WIDGET_BODIES,
  resolveWidgetCity,
  widgetDays,
  widgetDefinition,
  widgetHours,
} from '@/features/dashboard/widgets'

const DAY_SEGMENTS: ReadonlyArray<{ value: WidgetDays; label: string }> = DAY_OPTIONS.map(
  (value) => ({ value, label: `${value} d` }),
)

const HOUR_SEGMENTS: ReadonlyArray<{ value: WidgetHours; label: string }> = HOUR_OPTIONS.map(
  (value) => ({ value, label: value === 168 ? '7 d' : `${value} h` }),
)

interface WidgetCardProps {
  widget: DashboardWidget
  /** Position in the layout, 0-based — the up/down buttons are disabled at the ends. */
  index: number
  total: number
  onChange: (patch: WidgetConfig) => void
  onMove: (delta: -1 | 1) => void
  onRemove: () => void
}

/**
 * One card of the panel: a title, a compact control row built from the widget
 * definition, and the body.
 *
 * Reordering is two buttons, never drag-and-drop: a pointer-only affordance
 * would lock a keyboard or screen-reader user out of their own layout. Each
 * icon button is named after the widget it acts on, so a card with two
 * neighbours of the same type still announces itself unambiguously.
 */
export function WidgetCard({ widget, index, total, onChange, onMove, onRemove }: WidgetCardProps) {
  const definition = widgetDefinition(widget.type)
  const Body = WIDGET_BODIES[widget.type]
  const citySelectId = useId()

  /* `/cities` is already in the cache on every screen (the shell's search mounts
     it), so resolving the station here costs no extra request. */
  const { data: cityList } = useCities()
  const cityNames = useMemo(() => (cityList ?? []).map((city) => city.name), [cityList])

  const city = resolveWidgetCity(widget, cityNames)
  const days = widgetDays(widget, definition)
  const hours = widgetHours(widget, definition)

  return (
    <li>
      <Panel flush>
        <PanelHeader
          title={definition.title}
          actions={
            <>
              <Button
                size="icon"
                variant="ghost"
                aria-label={`Subir ${definition.title}`}
                disabled={index === 0}
                onClick={() => onMove(-1)}
              >
                <ArrowUp aria-hidden className="h-4 w-4" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                aria-label={`Bajar ${definition.title}`}
                disabled={index === total - 1}
                onClick={() => onMove(1)}
              >
                <ArrowDown aria-hidden className="h-4 w-4" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                aria-label={`Quitar ${definition.title}`}
                onClick={onRemove}
              >
                <X aria-hidden className="h-4 w-4" />
              </Button>
            </>
          }
        />

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line px-4 py-2">
          {definition.controls.includes('city') ? (
            <div className="flex items-center gap-2">
              <label htmlFor={citySelectId} className="label">
                Estación
              </label>
              <select
                id={citySelectId}
                value={city}
                onChange={(event) => onChange({ city: event.target.value })}
                className="h-8 rounded-[3px] border border-line bg-panel px-2 text-body text-fg focus:border-accent focus:outline-none"
              >
                {cityNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
          ) : null}

          {definition.controls.includes('days') ? (
            <div className="flex items-center gap-2">
              {/* The radiogroup is named by `Segmented`; this is its visible echo. */}
              <span className="label" aria-hidden="true">
                Ventana
              </span>
              <Segmented
                value={days}
                onChange={(value) => onChange({ days: value })}
                options={DAY_SEGMENTS}
                label={`Ventana · ${definition.title}`}
              />
            </div>
          ) : null}

          {definition.controls.includes('hours') ? (
            <div className="flex items-center gap-2">
              <span className="label" aria-hidden="true">
                Rango
              </span>
              <Segmented
                value={hours}
                onChange={(value) => onChange({ hours: value })}
                options={HOUR_SEGMENTS}
                label={`Rango · ${definition.title}`}
              />
            </div>
          ) : null}
        </div>

        <div className="p-3">
          <Body city={city} days={days} hours={hours} />
        </div>
      </Panel>
    </li>
  )
}
