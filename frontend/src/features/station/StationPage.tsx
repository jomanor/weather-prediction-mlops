import { RefreshCw } from 'lucide-react'
import { useMemo } from 'react'

import { useCities } from '@/api/queries'
import type { WeatherPoint } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { HistoryChart, type HistoryVariable } from '@/components/charts/HistoryChart'
import { Button } from '@/components/ui/Button'
import { ChartSkeleton, EmptyState, ErrorState, LoadingBlock } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Segmented } from '@/components/ui/Segmented'
import { CurrentConditions } from '@/features/station/CurrentConditions'
import { LatestPrediction } from '@/features/station/LatestPrediction'
import {
  PREDICTION_HORIZONS,
  usePredictions,
  useStationSeries,
  type PredictionHorizon,
} from '@/features/station/queries'
import { useChartWindow } from '@/hooks/useChartSync'
import { resolveCity, useStationSelection } from '@/hooks/useStationSelection'
import { urlOption, useUrlState, type UrlCodec } from '@/hooks/useUrlState'
import { cn } from '@/lib/cn'
import { isNum } from '@/lib/format'
import { latestRun, pickHorizon } from '@/lib/predictions'

const HOURS = [24, 48, 72, 168] as const
type StationHours = (typeof HOURS)[number]

const RANGES: ReadonlyArray<{ value: StationHours; label: string }> = [
  { value: 24, label: '24 h' },
  { value: 48, label: '48 h' },
  { value: 72, label: '72 h' },
  { value: 168, label: '7 d' },
]

const VARIABLES: ReadonlyArray<{ value: HistoryVariable; label: string }> = [
  { value: 'temperature', label: 'Temp' },
  { value: 'precipitation', label: 'Lluvia' },
  { value: 'wind', label: 'Viento' },
]

/* Stable identity for the "no points" case: a fresh `[]` on every render would
   re-run the chart's row merge (and its memo) for nothing. */
const NO_POINTS: WeatherPoint[] = []

/* Contract 2: horizon selector over the M2 config; default +1 h. */
const HORIZON_OPTIONS: ReadonlyArray<{ value: PredictionHorizon; label: string }> =
  PREDICTION_HORIZONS.map((value) => ({ value, label: `+${value} h` }))

/* URL state: the station is shared app-wide; range, variable and horizon are local views. */
const STATION_SCHEMA: {
  hours: UrlCodec<StationHours>
  var: UrlCodec<HistoryVariable>
  horizon: UrlCodec<PredictionHorizon>
} = {
  hours: urlOption(HOURS, 48),
  var: urlOption(['temperature', 'precipitation', 'wind'] as const, 'temperature'),
  horizon: urlOption(PREDICTION_HORIZONS, 1),
}

export function StationPage() {
  const [state, setState] = useUrlState(STATION_SCHEMA)
  const { selectedCity, selectCity } = useStationSelection()
  const { palette } = usePreferences()
  const { data: cityList } = useCities()
  const cities = useMemo(() => cityList?.map((c) => c.name) ?? [], [cityList])

  const city = resolveCity(selectedCity, cities, cityList !== undefined)
  const hours = state.hours
  const variable = state.var
  const horizon = state.horizon

  /* ONE bulk request per station: the hero observation and the chart series are
     two views of the same payload, so they cannot disagree. */
  const series = useStationSeries(city, hours)
  const station = useMemo(
    () => series.data?.cities.find((entry) => entry.city === city),
    [series.data, city],
  )
  /* The API serves each city oldest -> newest; the chart reads left to right. */
  const points = station?.points ?? NO_POINTS
  /* The hero is the newest observation, never `points[0]`. */
  const observation = points.at(-1) ?? null
  /* One unfiltered predictions query per station; card and chart both derive
     from it so they can never disagree about which run is the latest. */
  const predictions = usePredictions(city)
  const run = useMemo(() => latestRun(predictions.data ?? []), [predictions.data])
  const selectedPrediction = useMemo(() => pickHorizon(run, horizon), [run, horizon])
  const intervalLevel =
    selectedPrediction !== undefined && isNum(selectedPrediction.interval_level)
      ? selectedPrediction.interval_level
      : null

  /* Share the observed window with the benchmark charts on the same time axis. */
  const { brushed, resetWindow } = useChartWindow(hours, observation?.observed_at)

  return (
    <div className="space-y-5 pb-8">
      <PageHeader
        title={city || 'Selecciona una estación'}
        description="Condiciones observadas, evolución reciente y última predicción del modelo para la estación seleccionada."
        actions={
          <>
            <label className="sr-only" htmlFor="station-select">
              Estación
            </label>
            <select
              id="station-select"
              value={city}
              onChange={(event) => selectCity(event.target.value)}
              className="h-9 rounded-[3px] border border-line bg-panel px-2.5 text-xs text-fg focus:border-accent focus:outline-none"
            >
              {(cities ?? []).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            <Segmented
              value={hours}
              onChange={(value) => setState({ hours: value })}
              options={RANGES}
              label="Rango"
            />
            <Segmented
              value={horizon}
              onChange={(value) => setState({ horizon: value })}
              options={HORIZON_OPTIONS}
              label="Horizonte de predicción"
            />
            <Button onClick={() => series.refetch()} disabled={series.isFetching}>
              <RefreshCw className={cn('h-3.5 w-3.5', series.isFetching && 'animate-spin')} />
              Actualizar
            </Button>
          </>
        }
      />

      {!city ? (
        <div className="px-4 sm:px-6">
          <Panel>
            <LoadingBlock label="Cargando estaciones…" />
          </Panel>
        </div>
      ) : (
        <>
          <div className="px-4 sm:px-6">
            {series.isError ? (
              <Panel>
                <ErrorState
                  title="No se pudo cargar la observación"
                  description={series.error.message}
                  action={<Button onClick={() => series.refetch()}>Reintentar</Button>}
                />
              </Panel>
            ) : series.isLoading ? (
              <Panel>
                <LoadingBlock />
              </Panel>
            ) : observation ? (
              <CurrentConditions station={observation} />
            ) : (
              /* `/series` answers 200 with an empty city instead of the old
                 `/current` 404, so "no observation in this window" is an empty
                 state, not an error — and the window can be widened from here. */
              <Panel>
                <EmptyState
                  title="Sin observaciones recientes"
                  description={`No hay observaciones de ${city} en las últimas ${hours} h. Amplía el rango o revisa la ingesta.`}
                  action={<Button onClick={() => series.refetch()}>Reintentar</Button>}
                />
              </Panel>
            )}
          </div>

          <div className="grid items-start gap-5 px-4 sm:px-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
            <Panel flush className="order-2 xl:order-1">
              <PanelHeader
                title="Evolución observada"
                subtitle={`Últimas ${hours} horas · modelo a +${horizon} h`}
                actions={
                  <>
                    {brushed ? (
                      <Button size="sm" onClick={resetWindow}>
                        Restablecer zoom
                      </Button>
                    ) : null}
                    <Segmented
                      value={variable}
                      onChange={(value) => setState({ var: value }, { replace: false })}
                      options={VARIABLES}
                      label="Variable"
                    />
                  </>
                }
              />
              <div className="p-3">
                {series.isLoading ? (
                  <ChartSkeleton />
                ) : series.isError ? (
                  <ErrorState title="Sin histórico" description={series.error.message} />
                ) : (
                  <HistoryChart
                    points={points}
                    predictions={run}
                    palette={palette}
                    variable={variable}
                    intervalLevel={intervalLevel}
                  />
                )}
              </div>
            </Panel>

            <div className="order-1 xl:order-2">
              {predictions.isLoading ? (
                <Panel>
                  <LoadingBlock />
                </Panel>
              ) : (
                <LatestPrediction predictions={predictions.data ?? []} horizon={horizon} />
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
