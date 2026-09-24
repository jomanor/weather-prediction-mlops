import type { Metrics } from '@/api/schemas'
import { cn } from '@/lib/cn'
import { formatInteger, formatNumber, formatSigned } from '@/lib/format'

const ROWS: { key: keyof Omit<Metrics, 'n'>; label: string; hint: string }[] = [
  { key: 'mae', label: 'MAE', hint: 'Error absoluto medio' },
  { key: 'rmse', label: 'RMSE', hint: 'Raíz del error cuadrático medio' },
  { key: 'bias', label: 'Sesgo', hint: 'Error medio con signo' },
]

function bestIndex(model: number | null, aemet: number | null): 'model' | 'aemet' | null {
  if (model === null || aemet === null) return null
  return Math.abs(model) <= Math.abs(aemet) ? 'model' : 'aemet'
}

export function MetricsCompare({ model, aemet }: { model: Metrics; aemet: Metrics }) {
  return (
    <table className="w-full border-collapse text-xs">
      <thead>
        <tr className="border-b border-line">
          <th className="label px-4 py-2 text-left font-normal">Métrica</th>
          <th className="label px-4 py-2 text-right font-normal text-model">Spark GBT</th>
          <th className="label px-4 py-2 text-right font-normal text-aemet">AEMET</th>
        </tr>
      </thead>
      <tbody>
        {ROWS.map((row) => {
          const modelValue = model[row.key]
          const aemetValue = aemet[row.key]
          const winner = bestIndex(modelValue, aemetValue)
          return (
            <tr key={row.key} className="border-b border-line/60 last:border-0">
              <td className="px-4 py-2.5">
                <span className="font-medium text-fg-2">{row.label}</span>
                <span className="ml-2 hidden text-[10px] text-fg-3 sm:inline">{row.hint}</span>
              </td>
              <td
                className={cn(
                  'nums px-4 py-2.5 text-right',
                  winner === 'model' ? 'font-semibold text-fg' : 'text-fg-2',
                )}
              >
                {row.key === 'bias' ? formatSigned(modelValue, 2) : formatNumber(modelValue, 2)}
              </td>
              <td
                className={cn(
                  'nums px-4 py-2.5 text-right',
                  winner === 'aemet' ? 'font-semibold text-fg' : 'text-fg-2',
                )}
              >
                {row.key === 'bias' ? formatSigned(aemetValue, 2) : formatNumber(aemetValue, 2)}
              </td>
            </tr>
          )
        })}
        <tr>
          <td className="px-4 py-2.5 text-fg-3">Muestras</td>
          <td className="nums px-4 py-2.5 text-right text-fg-2">{formatInteger(model.n)}</td>
          <td className="nums px-4 py-2.5 text-right text-fg-2">{formatInteger(aemet.n)}</td>
        </tr>
      </tbody>
    </table>
  )
}
