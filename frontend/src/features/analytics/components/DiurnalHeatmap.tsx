import { useMemo } from 'react'

import { HOURS, buildDiurnalMatrix } from '@/features/analytics/transforms'
import type { DiurnalCell } from '@/features/analytics/schemas'
import { cn } from '@/lib/cn'
import { formatNumber, isNum } from '@/lib/format'
import { RAMP_STEPS, rampColor, rampDomain, rampToken } from '@/lib/ramps'

/**
 * Diurnal heatmap: one row per canonical city, one column per local hour
 * (Contract 4, `/analytics/diurnal`). Magnitude uses the shared temperature
 * ramp so the grid agrees with the map; hours with no sample stay a neutral
 * cell and are announced as "sin datos" rather than drawn as a value.
 */
export function DiurnalHeatmap({
  cells,
  className,
}: {
  cells: readonly DiurnalCell[]
  className?: string
}) {
  const matrix = useMemo(() => buildDiurnalMatrix(cells), [cells])
  const domain = rampDomain('temp')

  if (!matrix.rows.length) return null

  return (
    <div className={cn('overflow-x-auto', className)}>
      <table className="border-separate border-spacing-[3px] text-[10px]">
        <caption className="sr-only">
          Temperatura media por estación y hora local, en grados centígrados
        </caption>
        <thead>
          <tr>
            <th
              scope="col"
              className="label sticky left-0 z-10 bg-panel py-1 pr-3 text-left font-normal"
            >
              Estación
            </th>
            {HOURS.map((hour) => (
              <th
                key={hour}
                scope="col"
                aria-label={`${hour}:00`}
                className="label w-5 py-1 text-center font-normal"
              >
                {hour % 3 === 0 ? hour : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => (
            <tr key={row.city}>
              <th
                scope="row"
                className="sticky left-0 z-10 max-w-[7.5rem] truncate bg-panel py-0.5 pr-3 text-left font-medium text-fg-2"
              >
                {row.city}
              </th>
              {row.values.map((value, hour) => (
                <td key={hour} className="p-0 align-middle">
                  <span
                    aria-hidden
                    className="block h-4 w-5 rounded-[2px]"
                    style={{
                      background: isNum(value) ? rampColor('temp', value, domain) : 'var(--panel-2)',
                    }}
                    title={
                      isNum(value)
                        ? `${row.city} · ${hour}:00 · ${formatNumber(value, 1)} °C`
                        : `${row.city} · ${hour}:00 · sin datos`
                    }
                  />
                  <span className="sr-only">
                    {`${row.city} ${hour}:00, ${
                      isNum(value) ? `${formatNumber(value, 1)} grados` : 'sin datos'
                    }`}
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mt-3 flex items-center gap-2 pl-1 text-[10px] text-fg-3">
        <span className="nums">{formatNumber(domain[0], 0)}°</span>
        <span className="flex" aria-hidden>
          {Array.from({ length: RAMP_STEPS.temp }, (_, step) => (
            <span
              key={step}
              className="h-2.5 w-4"
              style={{ background: rampToken('temp', step) }}
            />
          ))}
        </span>
        <span className="nums">{formatNumber(domain[1], 0)}°</span>
        <span className="ml-2">
          Temperatura media por hora local
          {isNum(matrix.min) && isNum(matrix.max)
            ? ` · rango ${formatNumber(matrix.min, 1)}–${formatNumber(matrix.max, 1)} °C`
            : ''}
        </span>
      </div>
    </div>
  )
}
