import { CloudOff, RefreshCw } from 'lucide-react'
import { lazy, Suspense, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'

import { useBenchmarkSummary, useLatestPredictions, useStations } from '@/api/queries'
import type { Prediction } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { Button } from '@/components/ui/Button'
import { ErrorState, LoadingBlock, Skeleton } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel, PanelBody, PanelHeader } from '@/components/ui/Panel'
import { Readout } from '@/components/ui/Readout'
import { StationTable } from '@/features/overview/StationTable'
import { formatNumber, formatPercent, formatTemperature, formatWind, isNum } from '@/lib/format'

// MapLibre is by far the heaviest dependency; keep it out of the initial bundle.
const StationMap = lazy(() =>
  import('@/components/map/StationMap').then((module) => ({ default: module.StationMap })),
)

export function OverviewPage() {
  const navigate = useNavigate()
  const { units } = usePreferences()
  const stationsQuery = useStations()
  const predictionsQuery = useLatestPredictions()
  const summaryQuery = useBenchmarkSummary()

  const stations = stationsQuery.data?.stations ?? []

  const predictionIndex = useMemo(() => {
    const index = new Map<string, Prediction>()
    for (const prediction of predictionsQuery.data?.predictions ?? []) {
      if (!index.has(prediction.city)) index.set(prediction.city, prediction)
    }
    return index
  }, [predictionsQuery.data])

  const kpis = useMemo(() => {
    const temperatures = stations.map((s) => s.temperature).filter(isNum)
    const humidities = stations.map((s) => s.humidity).filter(isNum)
    const winds = stations.map((s) => s.wind_speed).filter(isNum)
    const mean = (values: number[]) =>
      values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null

    const modelErrors = (summaryQuery.data?.cities ?? [])
      .map((city) => city.model.mae)
      .filter(isNum)

    return {
      online: stations.length,
      avgTemperature: mean(temperatures),
      avgHumidity: mean(humidities),
      avgWind: mean(winds),
      modelMae: mean(modelErrors),
      covered: predictionIndex.size,
    }
  }, [stations, summaryQuery.data, predictionIndex])

  const refreshedAt = predictionsQuery.data?.generated_at ?? stationsQuery.dataUpdatedAt

  if (stationsQuery.isError) {
    return (
      <div className="p-4 sm:p-6">
        <ErrorState
          title="No se pudo cargar la red de estaciones"
          description={stationsQuery.error.message}
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
    <div className="space-y-5 pb-8">
      <PageHeader
        eyebrow="Red nacional"
        title="Resumen operativo"
        description="Estado en vivo de las estaciones, última predicción del modelo Spark GBT y cobertura de la inferencia."
        actions={
          <Button onClick={() => stationsQuery.refetch()} disabled={stationsQuery.isFetching}>
            <RefreshCw className={stationsQuery.isFetching ? 'h-3.5 w-3.5 animate-spin' : 'h-3.5 w-3.5'} />
            Actualizar
          </Button>
        }
      />

      <div className="px-4 sm:px-6">
        <Panel>
          <div className="grid grid-cols-2 divide-line md:grid-cols-5 md:divide-x">
            {stationsQuery.isLoading ? (
              Array.from({ length: 5 }).map((_, index) => (
                <div key={index} className="p-4">
                  <Skeleton className="h-2.5 w-16" />
                  <Skeleton className="mt-2.5 h-6 w-20" />
                </div>
              ))
            ) : (
              <>
                <div className="p-4">
                  <Readout
                    label="Estaciones"
                    value={kpis.online}
                    hint={`${kpis.covered} con predicción`}
                  />
                </div>
                <div className="p-4">
                  <Readout
                    label="Temp. media"
                    value={formatTemperature(kpis.avgTemperature, units, 1)}
                  />
                </div>
                <div className="p-4">
                  <Readout label="Humedad media" value={formatPercent(kpis.avgHumidity, 0)} />
                </div>
                <div className="p-4">
                  <Readout label="Viento medio" value={formatWind(kpis.avgWind, units)} />
                </div>
                <div className="col-span-2 p-4 md:col-span-1">
                  <Readout
                    label="MAE modelo 24 h"
                    value={isNum(kpis.modelMae) ? formatNumber(kpis.modelMae, 2) : '—'}
                    unit={isNum(kpis.modelMae) ? '°C' : undefined}
                    hint={
                      summaryQuery.isError
                        ? 'Benchmark no disponible'
                        : isNum(kpis.modelMae)
                          ? 'Media nacional'
                          : 'Sin predicciones registradas'
                    }
                  />
                </div>
              </>
            )}
          </div>
        </Panel>
      </div>

      <div className="space-y-5 px-4 sm:px-6">
        <Panel flush className="overflow-hidden">
          <PanelHeader
            title="Mapa de estaciones"
            subtitle="Temperatura observada y última predicción por ciudad"
            actions={
              refreshedAt ? (
                <span className="nums text-[10px] text-fg-3">
                  Act. {new Date(refreshedAt).toLocaleTimeString('es-ES')}
                </span>
              ) : null
            }
          />
          <div className="h-[400px] sm:h-[480px]">
            {stationsQuery.isLoading ? (
              <LoadingBlock label="Cargando estaciones…" />
            ) : (
              <Suspense fallback={<LoadingBlock label="Cargando mapa…" />}>
                <StationMap
                  stations={stations}
                  selectedCity={null}
                  onSelect={(city) => navigate(`/stations?city=${encodeURIComponent(city)}`)}
                />
              </Suspense>
            )}
          </div>
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
              <StationTable stations={stations} predictions={predictionIndex} />
            )}
          </PanelBody>
        </Panel>
      </div>
    </div>
  )
}
