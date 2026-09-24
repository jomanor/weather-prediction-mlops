import { AlertTriangle, Info } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { useBenchmark, useBenchmarkSummary, useCities } from '@/api/queries'
import { usePreferences } from '@/app/preferences'
import { BenchmarkChart } from '@/components/charts/BenchmarkChart'
import { ResidualChart } from '@/components/charts/ResidualChart'
import { SeriesLegend } from '@/components/charts/SeriesLegend'
import { EmptyState, ErrorState, LoadingBlock } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Segmented } from '@/components/ui/Segmented'
import { MetricsCompare } from '@/features/benchmark/MetricsCompare'
import { cn } from '@/lib/cn'
import { formatNumber, isNum } from '@/lib/format'

const RANGES = [
  { value: 24, label: '24 h' },
  { value: 48, label: '48 h' },
  { value: 72, label: '72 h' },
]

export function BenchmarkPage() {
  const [params, setParams] = useSearchParams()
  const { palette } = usePreferences()
  const { data: cityList } = useCities()
  const cities = useMemo(() => cityList?.map((c) => c.name) ?? [], [cityList])
  const [hours, setHours] = useState(24)

  const city = params.get('city') ?? cities?.[0] ?? ''
  const benchmark = useBenchmark(city, hours)
  const summary = useBenchmarkSummary()

  const verdict = useMemo(() => {
    const modelMae = benchmark.data?.metrics.model.mae
    const aemetMae = benchmark.data?.metrics.aemet.mae
    if (!isNum(modelMae) || !isNum(aemetMae)) return null
    const difference = Math.abs(modelMae - aemetMae)
    if (difference < 0.05) return 'Ambos modelos tienen un error equivalente en este periodo.'
    return modelMae < aemetMae
      ? `El modelo Spark GBT reduce el MAE en ${formatNumber(difference, 2)} °C frente a AEMET.`
      : `AEMET reduce el MAE en ${formatNumber(difference, 2)} °C frente al modelo Spark GBT.`
  }, [benchmark.data])

  const selectCity = (next: string) => {
    const updated = new URLSearchParams(params)
    updated.set('city', next)
    setParams(updated, { replace: true })
  }

  const hasSeries = Boolean(benchmark.data?.series.length)

  return (
    <div className="space-y-5 pb-8">
      <PageHeader
        title="Benchmark de modelos"
        description="Contraste entre la observación, la predicción del modelo Spark GBT y la predicción oficial de AEMET OpenData sobre la misma ventana temporal."
        actions={
          <>
            <label className="sr-only" htmlFor="benchmark-city">
              Estación
            </label>
            <select
              id="benchmark-city"
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
            <Segmented value={hours} onChange={setHours} options={RANGES} label="Ventana" />
          </>
        }
      />

      {/* The whole view depends on the benchmark call, so a failure is reported once. */}
      {benchmark.isError ? (
        <div className="px-4 sm:px-6">
          <Panel>
            <ErrorState
              title="No se pudo calcular el benchmark"
              description={
                <>
                  {benchmark.error.message}
                  <br />
                  Se necesitan observaciones y predicciones solapadas en las últimas {hours} horas
                  para esta estación.
                </>
              }
            />
          </Panel>
        </div>
      ) : null}

      {benchmark.data && !benchmark.data.aemet.available ? (
        <div className="px-4 sm:px-6">
          <div className="flex items-start gap-2.5 rounded-[var(--radius-panel)] border border-warn/35 bg-warn/5 px-4 py-3">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" />
            <div className="text-xs leading-relaxed">
              <p className="font-medium text-fg">Benchmark AEMET no disponible</p>
              <p className="mt-0.5 text-fg-2">
                {benchmark.data.aemet.error ??
                  'La API de AEMET OpenData no ha devuelto datos para esta estación.'}{' '}
                Solo se muestra la comparación frente a la observación.
              </p>
            </div>
          </div>
        </div>
      ) : null}

      {!benchmark.isError ? (
        <>
          <div className="grid items-start gap-5 px-4 sm:px-6 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
            <Panel flush>
              <PanelHeader
                title="Temperatura: observado vs modelos"
                subtitle={city ? `Estación ${city} · ventana ${hours} h` : undefined}
                actions={
                  <SeriesLegend
                    className="hidden sm:flex"
                    items={[
                      { label: 'Observado', color: palette.observed },
                      { label: 'Spark GBT', color: palette.model, dash: '6 3' },
                      { label: 'AEMET', color: palette.aemet, dash: '2 3' },
                    ]}
                  />
                }
              />
              <div className="p-3">
                {benchmark.isLoading ? (
                  <LoadingBlock />
                ) : hasSeries ? (
                  <BenchmarkChart points={benchmark.data!.series} palette={palette} />
                ) : (
                  <EmptyState
                    title="Sin datos comparables"
                    description="No hay observaciones y predicciones solapadas en esta ventana temporal."
                  />
                )}
              </div>
            </Panel>

            <Panel flush>
              <PanelHeader
                title="Métricas de error"
                subtitle={verdict ? undefined : 'Sin muestras suficientes en esta ventana'}
              />
              {benchmark.isLoading ? (
                <LoadingBlock />
              ) : benchmark.data ? (
                <>
                  <MetricsCompare
                    model={benchmark.data.metrics.model}
                    aemet={benchmark.data.metrics.aemet}
                  />
                  {verdict ? (
                    <p className="flex items-start gap-2 border-t border-line px-4 py-3 text-[11px] leading-relaxed text-fg-2">
                      <Info className="mt-0.5 h-3 w-3 shrink-0 text-accent" />
                      {verdict}
                    </p>
                  ) : null}
                </>
              ) : (
                <EmptyState title="Sin métricas" />
              )}
            </Panel>
          </div>

          <div className="px-4 sm:px-6">
            <Panel flush>
              <PanelHeader
                title="Residuo del modelo"
                subtitle="Predicción − observado, por hora. Rojo: el modelo se pasa; azul: se queda corto."
              />
              <div className="p-3">
                {benchmark.isLoading ? (
                  <LoadingBlock />
                ) : hasSeries ? (
                  <ResidualChart points={benchmark.data!.series} palette={palette} />
                ) : (
                  <EmptyState title="Sin residuos que mostrar" />
                )}
              </div>
            </Panel>
          </div>
        </>
      ) : null}

      <div className="px-4 sm:px-6">
        <Panel flush>
          <PanelHeader
            title="Comparativa nacional"
            subtitle="MAE por estación en la ventana seleccionada"
          />
          {summary.isLoading ? (
            <LoadingBlock />
          ) : summary.isError ? (
            <ErrorState title="Resumen no disponible" description={summary.error.message} />
          ) : !summary.data?.cities.length ? (
            <EmptyState
              title="Sin datos de benchmark"
              description="Ninguna estación tiene todavía observaciones y predicciones solapadas."
            />
          ) : (
            <div className="max-w-3xl overflow-x-auto">
              <table className="w-full border-collapse text-xs">
                <thead>
                  <tr className="border-b border-line">
                    <th className="label px-4 py-2 text-left font-normal">Estación</th>
                    <th className="label px-4 py-2 text-right font-normal text-model">
                      Spark GBT MAE
                    </th>
                    <th className="label px-4 py-2 text-right font-normal text-aemet">AEMET MAE</th>
                    <th className="label px-4 py-2 text-right font-normal">Muestras</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.data.cities.map((row) => {
                    const modelBetter =
                      isNum(row.model.mae) && isNum(row.aemet.mae) && row.model.mae < row.aemet.mae
                    return (
                      <tr
                        key={row.city}
                        onClick={() => selectCity(row.city)}
                        className={cn(
                          'cursor-pointer border-b border-line/60 transition-colors last:border-0 hover:bg-panel-2',
                          row.city === city && 'bg-panel-2',
                        )}
                      >
                        <td className="whitespace-nowrap px-4 py-2 font-medium text-fg">{row.city}</td>
                        <td
                          className={cn(
                            'nums whitespace-nowrap px-4 py-2 text-right',
                            modelBetter ? 'font-semibold text-fg' : 'text-fg-2',
                          )}
                        >
                          {formatNumber(row.model.mae, 2)}
                        </td>
                        <td
                          className={cn(
                            'nums whitespace-nowrap px-4 py-2 text-right',
                            !modelBetter && isNum(row.aemet.mae)
                              ? 'font-semibold text-fg'
                              : 'text-fg-2',
                          )}
                        >
                          {formatNumber(row.aemet.mae, 2)}
                        </td>
                        <td className="nums whitespace-nowrap px-4 py-2 text-right text-fg-3">
                          {formatNumber(row.model.n, 0)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </div>
  )
}
