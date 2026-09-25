import { CloudOff, RefreshCw } from 'lucide-react'
import { lazy, Suspense, useMemo } from 'react'

import { useBenchmarkSummary, useCities } from '@/api/queries'
import type { Prediction } from '@/api/schemas'
import { Button } from '@/components/ui/Button'
import { ErrorState, LoadingBlock } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel, PanelBody, PanelHeader } from '@/components/ui/Panel'
import { Segmented } from '@/components/ui/Segmented'
import { CityManager } from '@/features/overview/CityManager'
import { useLatestPredictions, useStations } from '@/features/overview/queries'
import { StationTable } from '@/features/overview/StationTable'
import { useStationSelection } from '@/hooks/useStationSelection'
import { urlOption, useUrlState, type UrlCodec } from '@/hooks/useUrlState'
import { formatNumber, formatPercent, formatTemperature, formatWind, isNum } from '@/lib/format'

const StationMap = lazy(() =>
  import('@/components/map/StationMap').then((module) => ({ default: module.StationMap })),
)

const TABS = [
  { value: 'red' as const, label: 'Red' },
  { value: 'estaciones' as const, label: 'Estaciones' },
]

const TAB_SCHEMA: { tab: UrlCodec<'red' | 'estaciones'> } = {
  tab: urlOption(['red', 'estaciones'] as const, 'red'),
}

export function OverviewPage() {
  const [tab, setTab] = useUrlState(TAB_SCHEMA)
  const { selectedCity, selectCity } = useStationSelection()
  const managing = tab.tab === 'estaciones'

  const citiesQuery = useCities()
  const stationsQuery = useStations()
  const predictionsQuery = useLatestPredictions()
  const summaryQuery = useBenchmarkSummary()

  const cities = citiesQuery.data ?? []
  const stations = stationsQuery.data?.stations ?? []
  const stationIndex = useMemo(
    () => new Map(stations.map((station) => [station.city, station])),
    [stations],
  )

  const predictionIndex = useMemo(() => {
    const index = new Map<string, Prediction>()
    for (const prediction of predictionsQuery.data?.predictions ?? []) {
      if (!index.has(prediction.city)) index.set(prediction.city, prediction)
    }
    return index
  }, [predictionsQuery.data])

  const stats = useMemo(() => {
    const temperatures = stations.map((s) => s.temperature).filter(isNum)
    const humidities = stations.map((s) => s.humidity).filter(isNum)
    const winds = stations.map((s) => s.wind_speed).filter(isNum)
    const mean = (values: number[]) =>
      values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null

    const modelErrors = (summaryQuery.data?.cities ?? [])
      .map((city) => city.model.mae)
      .filter(isNum)

    return {
      monitored: cities.length,
      observed: stations.length,
      avgTemperature: mean(temperatures),
      avgHumidity: mean(humidities),
      avgWind: mean(winds),
      modelMae: mean(modelErrors),
    }
  }, [cities, stations, summaryQuery.data])

  const refreshedAt = predictionsQuery.data?.generated_at ?? stationsQuery.dataUpdatedAt

  if (stationsQuery.isError || citiesQuery.isError) {
    return (
      <div className="p-4 sm:p-6">
        <ErrorState
          title="No se pudo cargar la red de estaciones"
          description={(stationsQuery.error ?? citiesQuery.error)?.message}
          action={
            <Button onClick={() => stationsQuery.refetch()}>
              <RefreshCw className="h-3.5 w-3.5" />
              Reintentar
            </Button>
          }
        />
      </div>
    )
  }

  return (
    <div className="space-y-4 pb-8">
      <PageHeader
        title="Red nacional"
        description="Observación en vivo de las estaciones registradas y última predicción del modelo."
        actions={
          <Button onClick={() => stationsQuery.refetch()} disabled={stationsQuery.isFetching}>
            <RefreshCw className={stationsQuery.isFetching ? 'h-3.5 w-3.5 animate-spin' : 'h-3.5 w-3.5'} />
            Actualizar
          </Button>
        }
      />

      <div className="px-4 sm:px-6">
        <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2 border-y border-line py-2.5">
          <Stat value={stats.monitored} label="estaciones monitorizadas" />
          <Stat
            value={formatTemperature(stats.avgTemperature, 'metric', 1)}
            label="temperatura media"
          />
          <Stat value={formatPercent(stats.avgHumidity, 0)} label="humedad media" />
          <Stat value={formatWind(stats.avgWind, 'metric')} label="viento medio" />
          <Stat
            value={isNum(stats.modelMae) ? formatNumber(stats.modelMae, 2) : undefined}
            unit="°C"
            label="MAE modelo 24 h"
            hint={
              summaryQuery.isError ? 'sin benchmark' : isNum(stats.modelMae) ? undefined : 'sin datos'
            }
          />
        </div>
      </div>

      <div className="space-y-4 px-4 sm:px-6">
        <Panel flush className="overflow-hidden">
          <PanelHeader
            title="Mapa de estaciones"
            subtitle="Temperatura observada, relieve y radar de precipitación"
            actions={
              <>
                {refreshedAt ? (
                  <span className="nums mr-2 text-[10px] text-fg-3">
                    Act. {new Date(refreshedAt).toLocaleTimeString('es-ES')}
                  </span>
                ) : null}
                <Segmented
                  value={tab.tab}
                  onChange={(value) => setTab({ tab: value }, { replace: false })}
                  options={TABS}
                  label="Vista"
                />
              </>
            }
          />
          <div className="h-[400px] sm:h-[480px]">
            {stationsQuery.isLoading ? (
              <LoadingBlock label="Cargando estaciones…" />
            ) : (
              <Suspense fallback={<LoadingBlock label="Cargando mapa…" />}>
                <StationMap
                  stations={stations}
                  selectedCity={selectedCity}
                  onSelect={selectCity}
                />
              </Suspense>
            )}
          </div>
          {managing ? (
            <PanelBody className="border-t border-line">
              <CityManager />
            </PanelBody>
          ) : null}
        </Panel>

        <Panel flush className="overflow-hidden">
          <PanelHeader
            title="Detalle por estación"
            subtitle="Observación actual frente a la predicción del modelo"
          />
          <PanelBody className="p-0">
            {stationsQuery.isLoading ? (
              <LoadingBlock />
            ) : stations.length === 0 ? (
              <div className="flex flex-col items-center py-12 text-fg-3">
                <CloudOff className="h-5 w-5" />
                <p className="mt-3 text-xs">Sin observaciones disponibles</p>
              </div>
            ) : (
              <StationTable
                rows={cities.map((city) => ({
                  name: city.name,
                  station: stationIndex.get(city.name),
                }))}
                predictions={predictionIndex}
              />
            )}
          </PanelBody>
        </Panel>
      </div>
    </div>
  )
}

function Stat({
  value,
  label,
  unit,
  hint,
}: {
  value: string | number | undefined
  label: string
  unit?: string
  hint?: string
}) {
  return (
    <div className="flex items-baseline gap-2 whitespace-nowrap pr-4">
      {value === undefined ? (
        <span className="nums text-sm text-fg-3">—</span>
      ) : (
        <span className="nums text-sm font-medium text-fg">{value}</span>
      )}
      {unit ? <span className="nums text-[11px] text-fg-3">{unit}</span> : null}
      <span className="text-[11px] text-fg-3">{hint ?? label}</span>
    </div>
  )
}


