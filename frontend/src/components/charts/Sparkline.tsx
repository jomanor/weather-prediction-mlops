import { Line, LineChart, ResponsiveContainer } from 'recharts'

interface SparklineProps {
  values: (number | null)[]
  color: string
  height?: number
}

/**
 * Axis-free trend line for embedding inside a readout. Too few points draw a
 * quiet baseline instead of a misleading empty box (same height, no shift).
 */
export function Sparkline({ values, color, height = 28 }: SparklineProps) {
  const data = values.map((value, index) => ({ index, value }))
  const hasTrend = data.filter((row) => row.value !== null).length >= 2

  if (!hasTrend) {
    return (
      <div
        aria-hidden
        className="flex items-center"
        style={{ height }}
        data-empty="true"
      >
        <span className="h-px w-full bg-line" />
      </div>
    )
  }

  return (
    <div aria-hidden style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 0, bottom: 2, left: 0 }}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
