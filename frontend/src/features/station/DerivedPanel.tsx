import { useMemo } from 'react'

import type { WeatherPoint } from '@/api/schemas'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Readout } from '@/components/ui/Readout'
import { EmptyState } from '@/components/ui/Feedback'
import { dewPoint, specificHumidity, windComponents } from '@/lib/derived'
import { formatNumber, isNum } from '@/lib/format'

/**
 * Quantities the Spark feature pipeline derives server-side, recomputed here
 * from the raw observations so they can be inspected per hour.
 */
export function DerivedPanel({ points }: { points: WeatherPoint[] }) {
  const derived = useMemo(() => {
    return points
      .map((point) => {
        if (!isNum(point.temperature) || !isNum(point.humidity) || !isNum(point.pressure)) {
          return null
        }
        const q = specificHumidity(point.temperature, point.humidity, point.pressure)
        const dp = dewPoint(point.temperature, point.humidity)
        const wind =
          isNum(point.wind_speed) && isNum(point.wind_direction)
            ? windComponents(point.wind_speed, point.wind_direction)
            : null
        return { observed_at: point.observed_at, q, dp, wind }
      })
      .filter((row): row is NonNullable<typeof row> => row !== null)
  }, [points])

  const latest = derived.at(-1) ?? null
  const meanQ = derived.length
    ? derived.reduce((sum, row) => sum + (row.q ?? 0), 0) / derived.length
    : null

  return (
    <Panel flush>
      <PanelHeader
        title="Variables derivadas"
        subtitle="Calculadas en el cliente a partir de la observación (Bolton 1980)"
      />
      {!latest ? (
        <EmptyState
          title="Sin datos suficientes"
          description="Se necesitan temperatura, humedad y presión simultáneas para derivar estas variables."
        />
      ) : (
        <div className="grid grid-cols-2 divide-line sm:grid-cols-4 sm:divide-x">
          <div className="p-4">
            <Readout
              label="Humedad específica q"
              value={formatNumber(latest.q, 2)}
              unit="g/kg"
              hint={`media ${formatNumber(meanQ, 2)} g/kg`}
            />
          </div>
          <div className="p-4">
            <Readout label="Punto de rocío" value={formatNumber(latest.dp, 1)} unit="°C" />
          </div>
          <div className="p-4">
            <Readout label="Viento u (zonal)" value={formatNumber(latest.wind?.u, 2)} unit="m/s" />
          </div>
          <div className="p-4">
            <Readout
              label="Viento v (meridional)"
              value={formatNumber(latest.wind?.v, 2)}
              unit="m/s"
            />
          </div>
        </div>
      )}
    </Panel>
  )
}
