import { Boxes, RefreshCw } from 'lucide-react'

import { useModels } from '@/api/queries'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { EmptyState, ErrorState, LoadingBlock } from '@/components/ui/Feedback'
import { PageHeader } from '@/components/ui/PageHeader'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { cn } from '@/lib/cn'
import { formatDateTime, formatNumber } from '@/lib/format'

export function ModelsPage() {
  const models = useModels()

  return (
    <div className="space-y-5 pb-8">
      <PageHeader
        eyebrow="MLOps"
        title="Registro de modelos"
        description="Artefactos entrenados por el job Spark GBT, con sus métricas de validación y la versión que sirve la inferencia."
        actions={
          <Button onClick={() => models.refetch()} disabled={models.isFetching}>
            <RefreshCw className={cn('h-3.5 w-3.5', models.isFetching && 'animate-spin')} />
            Actualizar
          </Button>
        }
      />

      <div className="px-4 sm:px-6">
        <Panel flush>
          <PanelHeader
            title="Modelos registrados"
            subtitle="Ordenados por fecha de entrenamiento, más reciente primero"
            actions={
              models.data ? (
                <span className="nums text-[10px] text-fg-3">{models.data.count} artefactos</span>
              ) : null
            }
          />
          {models.isLoading ? (
            <LoadingBlock />
          ) : models.isError ? (
            <ErrorState
              title="No se pudo consultar el registro"
              description={models.error.message}
              action={<Button onClick={() => models.refetch()}>Reintentar</Button>}
            />
          ) : !models.data?.models.length ? (
            <EmptyState
              icon={Boxes}
              title="Sin modelos entrenados"
              description="Ejecuta el job de entrenamiento (ml_training.py) para registrar el primer modelo en model_registry."
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs">
                <thead>
                  <tr className="border-b border-line">
                    <th className="label px-4 py-2 text-left font-normal">Modelo</th>
                    <th className="label px-4 py-2 text-left font-normal">Objetivo</th>
                    <th className="label px-4 py-2 text-right font-normal">Horizonte</th>
                    <th className="label px-4 py-2 text-right font-normal">RMSE</th>
                    <th className="label px-4 py-2 text-right font-normal">MAE</th>
                    <th className="label px-4 py-2 text-right font-normal">R²</th>
                    <th className="label px-4 py-2 text-left font-normal">Entrenado</th>
                    <th className="label px-4 py-2 text-left font-normal">Estado</th>
                  </tr>
                </thead>
                <tbody>
                  {models.data.models.map((model) => (
                    <tr
                      key={`${model.name}-${model.version}`}
                      className="border-b border-line/60 last:border-0"
                    >
                      <td className="px-4 py-2.5">
                        <div className="font-medium text-fg">{model.name}</div>
                        <div className="nums text-[10px] text-fg-3">{model.version}</div>
                      </td>
                      <td className="px-4 py-2.5 text-fg-2">{model.target}</td>
                      <td className="nums px-4 py-2.5 text-right text-fg-2">
                        +{model.horizon_hours} h
                      </td>
                      <td className="nums px-4 py-2.5 text-right text-fg-2">
                        {formatNumber(model.metrics.rmse ?? null, 3)}
                      </td>
                      <td className="nums px-4 py-2.5 text-right text-fg-2">
                        {formatNumber(model.metrics.mae ?? null, 3)}
                      </td>
                      <td className="nums px-4 py-2.5 text-right text-fg-2">
                        {formatNumber(model.metrics.r2 ?? null, 3)}
                      </td>
                      <td className="nums px-4 py-2.5 text-fg-3">{formatDateTime(model.created_at)}</td>
                      <td className="px-4 py-2.5">
                        <Badge tone={model.stage === 'production' ? 'ok' : 'neutral'}>
                          {model.stage}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </div>
  )
}
