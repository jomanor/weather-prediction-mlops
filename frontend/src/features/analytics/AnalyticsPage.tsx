import { ChartLine, Flame, RefreshCw } from 'lucide-react'
import { useMemo } from 'react'

import { useCities } from '@/api/queries'
import { usePreferences } from '@/app/preferences'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { EmptyState, ErrorState, LoadingBlock } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Segmented } from '@/components/ui/Segmented'
import { DailyAnomalyChart } from '@/features/analytics/components/DailyAnomalyChart'
import { DiurnalHeatmap } from '@/features/analytics/components/DiurnalHeatmap'
import { WindRose } from '@/features/analytics/components/WindRose'
import {
  useAnalyticsDaily,
  useAnalyticsDiurnal,
  useAnalyticsWindRose,
} from '@/features/analytics/queries'
import { heatwaveRuns } from '@/features/analytics/transforms'
import { resolveCity, useStationSelection } from '@/hooks/useStationSelection'
import { urlOption, useUrlState } from '@/hooks/useUrlState'
import { formatInteger } from '@/lib/format'
import { cn } from '@/lib/cn'

const DAY_OPTIONS = [30, 90, 180] as const
type DayRange = (typeof DAY_OPTIONS)[number]

/* Stable module-scope schema: `days` is the analytics window. */
const DAYS_SCHEMA = { days: urlOption(DAY_OPTIONS, 90) }

const DAY_SEGMENTS = DAY_OPTIONS.map((value) => ({ value, label: `${value} d` }))

export function AnalyticsPage() {
  const [state, setState] = useUrlState(DAYS_SCHEMA)
  const { selectedCity, selectCity } = useStationSelection()
  const { palette } = usePreferences()
  const { data: cityList } = useCities()

  const cities = useMemo(() => cityList?.map((city) => city.name) ?? [], [cityList])
  const city = resolveCity(selectedCity, cities, cityList !== undefined)
  const days: DayRange = state.days

  const diurnal = useAnalyticsDiurnal(days)
  const daily = useAnalyticsDaily(city, days)
  const wind = useAnalyticsWindRose(city, days)

  const runs = useMemo(() => heatwaveRuns(daily.data?.points ?? []), [daily.data])
  const heatwaveDays = runs.reduce((total, run) => total + (run.end - run.start + 1), 0)
  const refreshing = diurnal.isFetching || daily.isFetching || wind.isFetching

  const refetchAll = () => {
    void diurnal.refetch()
    void daily.refetch()
    void wind.refetch()
  }

  return (
    <div className="space-y-5 pb-8">
      <PageHeader
        title="Analítica climática"
        description="Ciclo diurno, anomalía y grados-día, y régimen de viento calculados sobre las observaciones almacenadas."
        actions={
          <>
            <Button onClick={refetchAll} disabled={refreshing}>
              <RefreshCw className={cn('h-3.5 w-3.5', refreshing && 'animate-spin')} />
              Actualizar
            </Button>
            <label className="sr-only" htmlFor="analytics-city">
              Estación
            </label>
            <select
              id="analytics-city"
              value={city}
              onChange={(event) => selectCity(event.target.value)}
              className="h-9 rounded-[3px] border border-line bg-panel px-2.5 text-xs text-fg focus:border-accent focus:outline-none"
            >
              {cities.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            <Segmented
              value={days}
              onChange={(value) => setState({ days: value })}
              options={DAY_SEGMENTS}
              label="Ventana"
            />
          </>
        }
      />

      <div className="px-4 sm:px-6">
        <Panel flush>
          <PanelHeader
            title="Ciclo diurno por estación"
            subtitle={`Temperatura media (°C) por hora local · últimos ${days} días`}
            actions={
              diurnal.data ? (
                <span className="nums text-[10px] text-fg-3">
                  {formatInteger(diurnal.data.cells.length)} celdas
                </span>
              ) : null
            }
          />
          {diurnal.isLoading ? (
            <LoadingBlock />
          ) : diurnal.isError ? (
            <ErrorState
              title="No se pudo cargar el ciclo diurno"
              description={diurnal.error.message}
              action={<Button onClick={() => diurnal.refetch()}>Reintentar</Button>}
            />
          ) : !diurnal.data?.cells.length ? (
            <EmptyState
              icon={ChartLine}
              title="Sin observaciones en la ventana"
              description={`No hay datos horarios en los últimos ${days} días. Amplía la ventana o comprueba la ingesta.`}
            />
          ) : (
            <div className="p-4">
              <DiurnalHeatmap cells={diurnal.data.cells} />
            </div>
          )}
        </Panel>
      </div>

      <div className="grid items-start gap-5 px-4 sm:px-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Panel flush>
          <PanelHeader
            title="Anomalía diaria y grados-día"
            subtitle={city ? `${city} · base 18 °C · últimos ${days} días` : undefined}
            actions={
              heatwaveDays > 0 ? (
                <Badge tone="bad">
                  <Flame aria-hidden className="h-3 w-3" />
                  Ola de calor · {heatwaveDays} d
                </Badge>
              ) : (
                <Badge tone="neutral">Sin ola de calor</Badge>
              )
            }
          />
          {daily.isLoading ? (
            <LoadingBlock />
          ) : daily.isError ? (
            <ErrorState
              title="No se pudo cargar la serie diaria"
              description={daily.error.message}
              action={<Button onClick={() => daily.refetch()}>Reintentar</Button>}
            />
          ) : !daily.data?.points.length ? (
            <EmptyState
              title="Sin agregados diarios"
              description="La estación no tiene observaciones suficientes para construir la serie diaria."
            />
          ) : (
            <>
              <div className="p-3">
                <DailyAnomalyChart points={daily.data.points} palette={palette} />
              </div>
              <p className="border-t border-line px-4 py-2 text-[11px] leading-relaxed text-fg-3">
                Anomalía = media diaria − climatología del mismo día del año. Ola de calor: ≥3 días
                consecutivos con máxima ≥ 35 °C (banda roja).
              </p>
            </>
          )}
        </Panel>

        <Panel flush>
          <PanelHeader
            title="Rosa de los vientos"
            subtitle={city ? `${city} · dirección de procedencia · últimos ${days} días` : undefined}
          />
          {wind.isLoading ? (
            <LoadingBlock />
          ) : wind.isError ? (
            <ErrorState
              title="No se pudo cargar la rosa de los vientos"
              description={wind.error.message}
              action={<Button onClick={() => wind.refetch()}>Reintentar</Button>}
            />
          ) : !wind.data?.sectors.length ? (
            <EmptyState
              title="Sin datos de viento"
              description="No hay observaciones de dirección y velocidad en la ventana seleccionada."
            />
          ) : (
            <div className="p-4">
              <WindRose sectors={wind.data.sectors} />
            </div>
          )}
        </Panel>
      </div>
    </div>
  )
}
