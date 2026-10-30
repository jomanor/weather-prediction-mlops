import { Cpu } from 'lucide-react'

import type { Prediction } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { Badge } from '@/components/ui/Badge'
import { EmptyState } from '@/components/ui/Feedback'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Readout } from '@/components/ui/Readout'
import {
  ageInHours,
  formatAgeHours,
  formatDateTime,
  formatNumber,
  formatPercent,
  formatTemperature,
  isNum,
} from '@/lib/format'

/** A prediction whose source observation is older than this is flagged, not hidden. */
export const PREDICTION_STALE_HOURS = 24

export function LatestPrediction({ predictions }: { predictions: Prediction[] }) {
  const { units } = usePreferences()
  const latest = predictions[0]
  const hasInterval =
    latest !== undefined && isNum(latest.temp_lower) && isNum(latest.temp_upper)
  /* TODO(stale-api): the prediction API will expose `source_age_hours` and a
     `stale` flag. Until then the age is derived locally from the origin
     observation so the badge stays honest without depending on new fields. */
  const ageHours = latest ? ageInHours(latest.source_timestamp) : null
  const isStale = ageHours !== null && ageHours > PREDICTION_STALE_HOURS

  return (
    <Panel flush>
      <PanelHeader
        title="Predicción del modelo"
        subtitle="Salida real del job de inferencia Spark GBT"
        actions={
          latest ? (
            <>
              <span
                className="nums text-[10px] text-fg-3"
                title="Antigüedad de la observación de origen"
              >
                {formatAgeHours(latest.source_timestamp)}
              </span>
              {isStale ? (
                <Badge tone="warn" title={`Más de ${PREDICTION_STALE_HOURS} h desde el origen`}>
                  Antigua
                </Badge>
              ) : null}
              <Badge tone="model">
                <Cpu className="h-2.5 w-2.5" />
                +{latest.horizon_hours} h
              </Badge>
            </>
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
            {hasInterval ? (
              <div className="flex justify-between gap-3">
                <dt className="text-fg-3">Intervalo</dt>
                <dd className="nums text-fg-2">
                  {formatTemperature(latest.temp_lower, units, 1)} –{' '}
                  {formatTemperature(latest.temp_upper, units, 1)}
                </dd>
              </div>
            ) : null}
            {isNum(latest.interval_level) ? (
              <div className="flex justify-between gap-3">
                <dt className="text-fg-3">Nivel</dt>
                <dd className="nums text-fg-2">
                  {formatPercent(latest.interval_level * 100, 0)}
                </dd>
              </div>
            ) : null}
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
