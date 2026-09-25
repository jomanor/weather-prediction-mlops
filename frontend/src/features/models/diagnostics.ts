import type { ModelDiagnostics } from '@/api/schemas'
import { isNum } from '@/lib/format'

/**
 * Contract 3: drift badge thresholds. `maxDriftPsi` is the largest finite
 * `drift_psi` across features; a legacy/absent diagnostics block yields
 * `null` so the UI can render an absent state instead of a fabricated grade.
 */
export type DriftTone = 'ok' | 'warn' | 'bad' | 'neutral'

export function maxDriftPsi(diagnostics: ModelDiagnostics | null | undefined): number | null {
  const values = Object.values(diagnostics?.drift_psi ?? {}).filter(isNum)
  return values.length ? Math.max(...values) : null
}

export function driftTone(psi: number | null): DriftTone {
  if (!isNum(psi)) return 'neutral'
  if (psi < 0.1) return 'ok'
  if (psi <= 0.25) return 'warn'
  return 'bad'
}

const DRIFT_LABEL: Record<Exclude<DriftTone, 'neutral'>, string> = {
  ok: 'Estable',
  warn: 'Aviso',
  bad: 'Deriva',
}

export function driftLabel(psi: number | null): string | null {
  const tone = driftTone(psi)
  return tone === 'neutral' ? null : DRIFT_LABEL[tone]
}
