import { Cpu } from 'lucide-react'

import type { Prediction } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { Badge } from '@/components/ui/Badge'
import { EmptyState } from '@/components/ui/Feedback'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Readout } from '@/components/ui/Readout'
import { formatDateTime, formatNumber, formatTemperature, isNum } from '@/lib/format'

export function LatestPrediction({ predictions }: { predictions: Prediction[] }) {
  const { units } = usePreferences()
  const latest = predictions[0]

  return (
    <Panel flush>
      <PanelHeader
        title="Predicción del modelo"
        subtitle="Salida real del job de inferencia Spark GBT"
        actions={
          latest ? (
            <Badge tone="model">
              <Cpu className="h-2.5 w-2.5" />
              +{latest.horizon_hours} h
            </Badge>
          ) : null
        }
      />

      {!latest ? (
        <EmptyState
          title="Sin predicciones para esta estación"
          description="El job de inferencia todavía no ha escrito resultados para esta ciudad en weather_predictions."
        />
      ) : (
        <>
          <div className="grid grid-cols-3 divide-line border-b border-line sm:divide-x">
            <div className="p-4">
              <Readout
                label="Temp. prevista"
                value={formatTemperature(latest.predicted_temperature, units)}
              />
            </div>
            <div className="p-4">
              <Readout
                label="Observado origen"
                value={formatTemperature(latest.observed_temperature, units)}
              />
            </div>
            <div className="p-4">
              <Readout
                label="Prob. lluvia"
                value={isNum(latest.predicted_rain) ? formatNumber(latest.predicted_rain, 2) : '—'}
              />
            </div>
          </div>

          <dl className="grid gap-x-6 gap-y-2 px-4 py-3 text-[11px] sm:grid-cols-2">
            <div className="flex justify-between gap-3">
              <dt className="text-fg-3">Instante origen</dt>
              <dd className="nums text-fg-2">{formatDateTime(latest.source_timestamp)}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-3">Ejecución</dt>
              <dd className="nums text-fg-2">{formatDateTime(latest.prediction_timestamp)}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-3">Modelo</dt>
              <dd className="truncate text-fg-2">{latest.temp_model_name ?? '—'}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-fg-3">Versión</dt>
              <dd className="nums text-fg-2">{latest.temp_model_version ?? '—'}</dd>
            </div>
          </dl>

          {predictions.length > 1 ? (
            <div className="border-t border-line px-4 py-3">
              <div className="label mb-2">Últimas ejecuciones</div>
              <ul className="space-y-1">
                {predictions.slice(0, 6).map((prediction) => (
                  <li
                    key={prediction.prediction_timestamp + prediction.source_timestamp}
                    className="flex items-center justify-between gap-3 text-[11px]"
                  >
                    <span className="nums text-fg-3">{formatDateTime(prediction.source_timestamp)}</span>
                    <span className="nums text-model">
                      {formatTemperature(prediction.predicted_temperature, units)}
                    </span>
                    <span className="nums text-fg-3">
                      obs {formatTemperature(prediction.observed_temperature, units)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </>
      )}
    </Panel>
  )
}
