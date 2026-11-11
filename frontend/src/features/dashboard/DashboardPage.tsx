import { LayoutGrid } from 'lucide-react'
import { useEffect, useId, useState } from 'react'

import { Button } from '@/components/ui/Button'
import { EmptyState } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel } from '@/components/ui/Panel'
import { WidgetCard } from '@/features/dashboard/WidgetCard'
import { WIDGET_REGISTRY } from '@/features/dashboard/widgets'
import {
  MAX_WIDGETS,
  type DashboardWidget,
  type WidgetConfig,
  type WidgetType,
} from '@/features/dashboard/types'
import { createWidgetId, readDashboard, writeDashboard } from '@/features/dashboard/storage'

/**
 * Panel: the dashboard the user composes, in the order they choose, stored in
 * this browser only (`meteoml.dashboard.v1`) — no account, no server layout.
 *
 * Persistence follows `app/preferences.tsx`: a hydrate function runs in the
 * state initialiser and one write-through effect saves every change. The read
 * is defensive (`storage.ts`), so a corrupt blob lands on the empty state
 * instead of a blank page.
 *
 * Reordering is two buttons per card rather than drag-and-drop (a pointer-only
 * gesture would make the layout unusable by keyboard), and the remove/reset
 * paths confirm inline instead of through `window.confirm`, which is
 * unstyleable, blocks the tab and cannot be translated by the page.
 */
export function DashboardPage() {
  const [widgets, setWidgets] = useState<DashboardWidget[]>(readDashboard)
  const [choosing, setChoosing] = useState(false)
  const [confirmingReset, setConfirmingReset] = useState(false)
  const chooserId = useId()

  /* Write-through: every change, including the normalising read on mount, is
     the new stored layout. */
  useEffect(() => {
    writeDashboard(widgets)
  }, [widgets])

  const atCapacity = widgets.length >= MAX_WIDGETS

  const addWidget = (type: WidgetType) => {
    const definition = WIDGET_REGISTRY.find((entry) => entry.type === type)
    if (!definition) return
    setWidgets((current) =>
      current.length >= MAX_WIDGETS
        ? current
        : [...current, { id: createWidgetId(), type, ...definition.defaultConfig }],
    )
    setChoosing(false)
  }

  const removeWidget = (id: string) => {
    setWidgets((current) => current.filter((widget) => widget.id !== id))
  }

  const moveWidget = (id: string, delta: -1 | 1) => {
    setWidgets((current) => {
      const from = current.findIndex((widget) => widget.id === id)
      const to = from + delta
      if (from < 0 || to < 0 || to >= current.length) return current
      const next = [...current]
      const moved = next[from]
      next[from] = next[to]
      next[to] = moved
      return next
    })
  }

  const configureWidget = (id: string, patch: WidgetConfig) => {
    setWidgets((current) =>
      current.map((widget) => (widget.id === id ? { ...widget, ...patch } : widget)),
    )
  }

  const reset = () => {
    setWidgets([])
    setChoosing(false)
    setConfirmingReset(false)
  }

  return (
    <div className="space-y-4 pb-8">
      <PageHeader
        title="Panel"
        description="Tu panel: elige las vistas y su orden. Se guarda solo en este navegador."
        actions={
          <>
            {widgets.length > 0 ? (
              <Button onClick={() => setConfirmingReset(true)}>Restablecer</Button>
            ) : null}
            <Button
              variant="primary"
              aria-expanded={choosing}
              aria-controls={choosing ? chooserId : undefined}
              disabled={atCapacity}
              onClick={() => setChoosing((open) => !open)}
            >
              Añadir vista
            </Button>
          </>
        }
      />

      {atCapacity ? (
        <p className="px-4 text-body text-fg-3 sm:px-6">
          Máximo de {MAX_WIDGETS} vistas: quita una para añadir otra.
        </p>
      ) : null}

      {confirmingReset ? (
        <div className="px-4 sm:px-6">
          <Panel>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-body text-fg-2">
                Se quitarán las {widgets.length} vistas de este navegador. No hay copia en el
                servidor.
              </p>
              <div className="flex items-center gap-2">
                <Button size="sm" onClick={() => setConfirmingReset(false)}>
                  Cancelar
                </Button>
                <Button size="sm" variant="danger" onClick={reset}>
                  Sí, restablecer
                </Button>
              </div>
            </div>
          </Panel>
        </div>
      ) : null}

      {choosing && !atCapacity ? (
        <div className="px-4 sm:px-6">
          <Panel>
            <div id={chooserId} className="flex flex-col gap-2">
              <span className="label">Añadir al panel</span>
              <div className="flex flex-wrap items-center gap-2">
                {WIDGET_REGISTRY.map((definition) => (
                  <Button
                    key={definition.type}
                    size="sm"
                    onClick={() => addWidget(definition.type)}
                  >
                    {definition.title}
                  </Button>
                ))}
              </div>
            </div>
          </Panel>
        </div>
      ) : null}

      {widgets.length === 0 ? (
        <div className="px-4 sm:px-6">
          <Panel>
            <EmptyState
              icon={LayoutGrid}
              title="Tu panel está vacío"
              description="Añade vistas y ordénalas a tu manera. Se guardan solo en este navegador: no hay cuenta ni copia en el servidor."
              action={<Button onClick={() => setChoosing(true)}>Añadir la primera vista</Button>}
            />
          </Panel>
        </div>
      ) : (
        <div className="px-4 sm:px-6">
          {/* An ordered list, because the order *is* the layout — a screen reader
              announces "3 de 5" and the move buttons act on that position. */}
          <ol aria-label="Vistas del panel" className="grid items-start gap-4 lg:grid-cols-2">
            {widgets.map((widget, index) => (
              <WidgetCard
                key={widget.id}
                widget={widget}
                index={index}
                total={widgets.length}
                onChange={(patch) => configureWidget(widget.id, patch)}
                onMove={(delta) => moveWidget(widget.id, delta)}
                onRemove={() => removeWidget(widget.id)}
              />
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}
