import { clamp } from '@/lib/format'
import type { TimeDomain } from '@/hooks/useChartSync'

/** Recharts `Brush` reports a start/end index; charts need the epoch-ms range. */
export interface BrushRange {
  startIndex?: number
  endIndex?: number
}

/**
 * Map a brush index range onto the shared time domain. Rows are assumed to be
 * ascending by `ts` (the API contract). Degenerate ranges (fewer than two rows,
 * or an inverted/empty selection) return `null`, i.e. "no shared domain".
 */
export function brushDomain(
  rows: ReadonlyArray<{ ts: number }>,
  range: BrushRange,
): TimeDomain | null {
  const last = rows.length - 1
  if (last < 1) return null
  const start = clamp(range.startIndex ?? 0, 0, last)
  const end = clamp(range.endIndex ?? last, 0, last)
  if (start >= end) return null
  return [rows[start].ts, rows[end].ts]
}
