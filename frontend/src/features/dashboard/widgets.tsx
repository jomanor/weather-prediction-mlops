import { ChartLine } from 'lucide-react'
import { useMemo, type ComponentType, type ReactNode } from 'react'

import { usePreferences } from '@/app/preferences'
import type { WeatherPoint } from '@/api/schemas'
import { HistoryChart } from '@/components/charts/HistoryChart'
import { Button } from '@/components/ui/Button'
import { EmptyState, ErrorState, LoadingBlock } from '@/components/ui/Feedback'
import { DiurnalHeatmap } from '@/features/analytics/components/DiurnalHeatmap'
import { WindRose } from '@/features/analytics/components/WindRose'
import { useAnalyticsDiurnal, useAnalyticsWindRose } from '@/features/analytics/queries'
import { useStationSeries } from '@/features/station/queries'
import {
  DAY_OPTIONS,
  HOUR_OPTIONS,
  type WidgetConfig,
  type WidgetControl,
  type WidgetDays,
  type WidgetHours,
  type WidgetType,
} from '@/features/dashboard/types'

/**
 * The widget catalogue: one entry per tile (`type`, `title`, `defaultConfig`,
 * `controls`) plus one body component in {@link WIDGET_BODIES}. The page and the
 * card read everything — the title, the control row, the empty/loading/error
 * copy — from here, so adding a tile is a data entry, not a page edit.
 *
 * Only components that already exist become tiles: `DiurnalHeatmap`, `WindRose`
 * and `HistoryChart` are prop-driven and know nothing about the dashboard.
 */
export interface WidgetDefinition {
  type: WidgetType
  title: string
  defaultConfig: WidgetConfig
  controls: readonly WidgetControl[]
}

export const WIDGET_REGISTRY: readonly WidgetDefinition[] = [
  {
    type: 'diurnal',
    title: 'Ciclo diurno',
    defaultConfig: { days: 90 },
    controls: ['days'],
  },
  {
    type: 'wind-rose',
    title: 'Rosa de los vientos',
    defaultConfig: { days: 90 },
    controls: ['city', 'days'],
  },
  {
    type: 'series',
    title: 'Serie de la estación',
    defaultConfig: { hours: 48 },
    controls: ['city', 'hours'],
  },
]

const BY_TYPE = new Map(WIDGET_REGISTRY.map((definition) => [definition.type, definition]))

/** Never undefined for a `WidgetType`: the type union and `WIDGET_BODIES` keep them in step. */
export function widgetDefinition(type: WidgetType): WidgetDefinition {
  const definition = BY_TYPE.get(type)
  if (!definition) throw new Error(`unknown widget type: ${type}`)
  return definition
}

/**
 * The station a card shows: the stored one while the registry still has it,
 * otherwise the first station. While the list is still loading the stored value
 * is trusted, so a reload does not flash the wrong city.
 */
export function resolveWidgetCity(config: WidgetConfig, names: readonly string[]): string {
  if (config.city && (names.length === 0 || names.includes(config.city))) return config.city
  return names[0] ?? ''
}

/** Stable identity for the "no points" case: a fresh `[]` would re-merge rows. */
const NO_POINTS: WeatherPoint[] = []

export interface WidgetBodyProps {
  /** Station resolved by the card; `''` until the registry has loaded. */
  city: string
  days: WidgetDays
  hours: WidgetHours
}

/** The four query states every tile renders the same way. */
interface QueryState {
  isLoading: boolean
  isError: boolean
  error: Error | null
  refetch: () => unknown
}

function WidgetQuery({
  query,
  errorTitle,
  isEmpty,
  emptyTitle,
  emptyDescription,
  children,
}: {
  query: QueryState
  errorTitle: string
  isEmpty: boolean
  emptyTitle: string
  emptyDescription: string
  children: ReactNode
}) {
  if (query.isLoading) return <LoadingBlock />
  if (query.isError) {
    return (
      <ErrorState
        title={errorTitle}
        description={query.error?.message}
        action={
          <Button size="sm" onClick={() => query.refetch()}>
            Reintentar
          </Button>
        }
      />
    )
  }
  if (isEmpty) {
    return <EmptyState icon={ChartLine} title={emptyTitle} description={emptyDescription} />
  }
  return <>{children}</>
}

function DiurnalBody({ days }: WidgetBodyProps) {
  /* No AEMET-coverage filter here, unlike `ClimateViewsDialog`: that filter
     matched rows against `/cities` because the endpoint returns every canonical
     station. The registry has been Spain-only since 2026-11-11, so the filter
     would drop nothing and would cost this tile a second request. */
  const query = useAnalyticsDiurnal(days)
  const cells = query.data?.cells ?? []

  return (
    <WidgetQuery
      query={query}
      errorTitle="No se pudo cargar el ciclo diurno"
      isEmpty={cells.length === 0}
      emptyTitle="Sin observaciones en la ventana"
      emptyDescription={`No hay datos horarios en los últimos ${days} días. Prueba una ventana más amplia.`}
    >
      <DiurnalHeatmap cells={cells} />
    </WidgetQuery>
  )
}

function WindRoseBody({ city, days }: WidgetBodyProps) {
  const query = useAnalyticsWindRose(city, days)
  const sectors = query.data?.sectors ?? []

  if (!city) return <LoadingBlock label="Cargando estaciones…" />

  return (
    <WidgetQuery
      query={query}
      errorTitle="No se pudo cargar la rosa de los vientos"
      isEmpty={sectors.length === 0}
      emptyTitle="Sin datos de viento"
      emptyDescription="No hay observaciones de dirección y velocidad en la ventana seleccionada."
    >
      <WindRose sectors={sectors} />
    </WidgetQuery>
  )
}

function SeriesBody({ city, hours }: WidgetBodyProps) {
  const { palette } = usePreferences()
  const query = useStationSeries(city, hours)
  const points = useMemo(
    () => query.data?.cities.find((entry) => entry.city === city)?.points ?? NO_POINTS,
    [query.data, city],
  )

  if (!city) return <LoadingBlock label="Cargando estaciones…" />

  return (
    <WidgetQuery
      query={query}
      errorTitle="No se pudo cargar la serie"
      isEmpty={points.length === 0}
      emptyTitle="Sin observaciones recientes"
      emptyDescription={`No hay observaciones de ${city} en las últimas ${hours} h.`}
    >
      <HistoryChart points={points} palette={palette} variable="temperature" />
    </WidgetQuery>
  )
}

/** Body per type. `Record<WidgetType, …>` makes a new type a compile error here. */
export const WIDGET_BODIES: Record<WidgetType, ComponentType<WidgetBodyProps>> = {
  diurnal: DiurnalBody,
  'wind-rose': WindRoseBody,
  series: SeriesBody,
}

/** The window a card renders: stored value, else the definition's default. */
export function widgetDays(config: WidgetConfig, definition: WidgetDefinition): WidgetDays {
  return config.days ?? definition.defaultConfig.days ?? DAY_OPTIONS[1]
}

export function widgetHours(config: WidgetConfig, definition: WidgetDefinition): WidgetHours {
  return config.hours ?? definition.defaultConfig.hours ?? HOUR_OPTIONS[0]
}
