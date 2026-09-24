import { Line, LineChart, ResponsiveContainer } from 'recharts'

interface SparklineProps {
  values: (number | null)[]
  color: string
  height?: number
}

/** Axis-free trend line for embedding inside a readout. */
export function Sparkline({ values, color, height = 28 }: SparklineProps) {
  const data = values.map((value, index) => ({ index, value }))
  if (data.length < 2) return <div style={{ height }} />

  return (
    <ResponsiveContainer width="100%" height={height}>
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
  )
}
