import { Activity, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

import { useWeatherQuality } from '@/api/queries'
import type { CityQuality } from '@/api/schemas'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { EmptyState, ErrorState, Skeleton } from '@/components/ui/Feedback'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { Readout } from '@/components/ui/Readout'
import { formatNumber, formatPercent, formatRelative, isNum } from '@/lib/format'

const STATUS: Record<
  CityQuality['status'],
  { tone: 'ok' | 'warn' | 'bad'; label: string; icon: LucideIcon }
> = {
  ok: { tone: 'ok', label: 'Correcto', icon: CheckCircle2 },
  warn: { tone: 'warn', label: 'Aviso', icon: AlertTriangle },
  bad: { tone: 'bad', label: 'Crítico', icon: XCircle },
}

const SEVERITY: Record<CityQuality['status'], number> = { bad: 0, warn: 1, ok: 2 }

/**
 * Data-quality meter (Contract 1). Self-fetching; every row pairs the colour
 * with a label and an icon, so status never depends on colour alone.
 */
export function DataQualityMeter({ className }: { className?: string }) {
  const quality = useWeatherQuality(7)
  const data = quality.data

  const cities = data?.cities ?? []
  const worst = cities.reduce<CityQuality | null>(
    (current, city) =>
      !current || SEVERITY[city.status] < SEVERITY[current.status] ? city : current,
    null,
  )
  const totalExpected = cities.reduce((sum, city) => sum + city.expected_hours, 0)
  const totalObserved = cities.reduce((sum, city) => sum + city.observed_hours, 0)
  const completeness = totalExpected > 0 ? totalObserved / totalExpected : null
  const flagged = cities.filter((city) => city.status !== 'ok').length
  const WorstIcon = worst ? STATUS[worst.status].icon : null

  return (
    <Panel flush className={className}>
      <PanelHeader
        title="Calidad de datos"
        subtitle="Cobertura de weather_features en los últimos 7 días, por estación"
        actions={
          <>
            {data?.generated_at ? (
              <span className="nums mr-1 text-[10px] text-fg-3">
                {formatRelative(data.generated_at)}
              </span>
            ) : null}
            {worst && WorstIcon ? (
              <Badge tone={STATUS[worst.status].tone}>
                <WorstIcon className="h-2.5 w-2.5" />
                {STATUS[worst.status].label}
              </Badge>
            ) : null}
          </>
        }
      />

      {quality.isLoading ? (
        <div className="space-y-2 p-4">
          {[0, 1, 2, 3].map((row) => (
            <Skeleton key={row} className="h-5 w-full" />
          ))}
        </div>
      ) : quality.isError ? (
        <ErrorState
          title="No se pudo consultar la calidad"
          description={quality.error.message}
          action={<Button onClick={() => quality.refetch()}>Reintentar</Button>}
        />
      ) : cities.length === 0 ? (
        <EmptyState
          icon={Activity}
          title="Sin datos de calidad"
          description="El endpoint no devolvió estaciones en la ventana solicitada."
        />
      ) : (
        <>
          <div className="grid grid-cols-3 divide-line border-b border-line sm:divide-x">
            <div className="p-4">
              <Readout
                label="Cobertura nacional"
                value={formatPercent(isNum(completeness) ? completeness * 100 : null, 1)}
              />
            </div>
            <div className="p-4">
              <Readout label="Estaciones" value={formatNumber(cities.length, 0)} />
            </div>
            <div className="p-4">
              <Readout
                label="Con aviso"
                value={formatNumber(flagged, 0)}
                hint={flagged > 0 ? 'completeness o antigüedad' : 'todo en rango'}
              />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-line">
                  <th className="label px-4 py-2 text-left font-normal">Estado</th>
                  <th className="label px-4 py-2 text-left font-normal">Estación</th>
                  <th className="label px-4 py-2 text-right font-normal">Cobertura</th>
                  <th className="label hidden px-4 py-2 text-right font-normal sm:table-cell">
                    Hueco máx.
                  </th>
                  <th className="label px-4 py-2 text-right font-normal">Antigüedad</th>
                </tr>
              </thead>
              <tbody>
                {[...cities]
                  .sort(
                    (a, b) =>
                      SEVERITY[a.status] - SEVERITY[b.status] ||
                      a.completeness - b.completeness,
                  )
                  .map((city) => {
                    const status = STATUS[city.status]
                    const Icon = status.icon
                    return (
                      <tr
                        key={city.city}
                        className="border-b border-line/60 last:border-0"
                      >
                        <td className="px-4 py-2">
                          <Badge tone={status.tone}>
                            <Icon className="h-2.5 w-2.5" />
                            {status.label}
                          </Badge>
                        </td>
                        <td className="px-4 py-2 text-fg">{city.city}</td>
                        <td className="nums px-4 py-2 text-right text-fg-2">
                          {formatPercent(city.completeness * 100, 1)}
                        </td>
                        <td className="nums hidden px-4 py-2 text-right text-fg-2 sm:table-cell">
                          {formatNumber(city.max_gap_hours, 1)} h
                        </td>
                        <td className="nums px-4 py-2 text-right text-fg-3">
                          {formatRelative(city.last_observed_at)}
                        </td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  )
}
