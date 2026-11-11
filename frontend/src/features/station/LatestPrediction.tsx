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
import { latestRun, pickHorizon } from '@/lib/predictions'

/** A prediction whose source observation is older than this is flagged, not hidden.
 * Must match the API's ``PREDICTION_STALE_AFTER_HOURS`` (12 h, one authority in
 * ``backend/app/core/thresholds.py``): this is only the fallback for a legacy
 * payload with no ``stale`` flag, so it must not disagree with the server. */
export const PREDICTION_STALE_HOURS = 12

interface LatestPredictionProps {
  /** Prediction history as served (oldest → newest); the newest run is picked here. */
  predictions: Prediction[]
  /** Selected M2 horizon; falls back to the newest row when it is missing. */
  horizon: number
}

export function LatestPrediction({ predictions, horizon }: LatestPredictionProps) {
  const { units } = usePreferences()
  /* The API is chronological, so the latest forecast is the newest
     `source_timestamp` run, then the row for the selected horizon. */
  const latest = pickHorizon(latestRun(predictions), horizon)
  const hasInterval =
    latest !== undefined && isNum(latest.temp_lower) && isNum(latest.temp_upper)
  /* The API resolves age/stale from the same timestamp, so both agree; the
     local fallback keeps a legacy payload honest too. */
  const ageHours = latest?.source_age_hours ?? (latest ? ageInHours(latest.source_timestamp) : null)
  /* OR, not `??`: a legacy payload without the field parses to `stale=false`,
     and the local age must still flag it. */
  const isStale =
    latest?.stale === true || (ageHours !== null && ageHours > PREDICTION_STALE_HOURS)
  /* One string, not a label plus a number: the stale state must read as a
     sentence in a screen reader as well as on screen. */
  const staleNote = `origen de más de ${PREDICTION_STALE_HOURS} h`

  return (
    <Panel flush>
      <PanelHeader
        title="Predicción"
        subtitle="Última salida del job de inferencia"
        actions={
          latest ? (
            <>
              {/* S10: the age is the fact and the words below carry the stale
                  state — the "Antigua" badge said the same thing twice. */}
              <span
                className="nums text-[10px] text-fg-3"
                title="Antigüedad de la observación de origen"
              >
                {formatAgeHours(latest.source_timestamp)}
              </span>
              {isStale ? (
                <span
                  className="text-[11px] text-warn"
                  title={`Más de ${PREDICTION_STALE_HOURS} h desde el origen`}
                >
                  {staleNote}
                </span>
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
        </>
      )}
    </Panel>
  )
}
