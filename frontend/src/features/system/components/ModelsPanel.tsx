import { Boxes, ChevronDown, ChevronRight, RefreshCw } from 'lucide-react'
import { useState, type ReactNode } from 'react'

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { EmptyState, ErrorState, TableSkeleton } from '@/components/ui/Feedback'
import { Panel, PanelHeader } from '@/components/ui/Panel'
import { useModels } from '@/features/models/queries'
import { driftLabel, driftTone, maxDriftPsi } from '@/features/models/diagnostics'
import { cn } from '@/lib/cn'
import {
  formatDateTime,
  formatNumber,
  formatPercent,
  formatSigned,
  isNum,
  PRECISION,
} from '@/lib/format'
import type { ModelDiagnostics } from '@/api/schemas'

type SkillTone = 'ok' | 'warn' | 'bad' | 'neutral'

/**
 * Registry stages are the backend's English lifecycle values; the table is
 * Spanish. Anything the registry adds later falls through unchanged rather than
 * printing nothing.
 */
const STAGE_LABELS: Record<string, string> = {
  production: 'Producción',
  staging: 'Staging',
  superseded: 'Sustituido',
  none: 'Sin estado',
}

export function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage
}

/** Skill thresholds: ≥ 0.3 is a solid win, 0–0.3 marginal, < 0 worse than the baseline. */
export function skillTone(skill: number | null | undefined): SkillTone {
  if (!isNum(skill)) return 'neutral'
  if (skill >= 0.3) return 'ok'
  if (skill >= 0) return 'warn'
  return 'bad'
}

/** Contract 3 drift badge: an absent state (`—`) when the registry has no PSI. */
function DriftBadge({ diagnostics }: { diagnostics: ModelDiagnostics }) {
  const psi = maxDriftPsi(diagnostics)
  const label = driftLabel(psi)
  if (psi === null || label === null) {
    return (
      <span className="nums text-fg-3" title="Sin diagnóstico de deriva en el registro">
        —
      </span>
    )
  }
  return (
    <Badge tone={driftTone(psi)} title={`PSI máx. ${formatNumber(psi, PRECISION.skill)}`}>
      {label}
    </Badge>
  )
}

/**
 * Registry tab: one row per artefact, honest metrics folded into an expander.
 *
 * The expander carries only facts the row does not — the persistence and
 * climatology baselines, Brier/AUC, the split and the interval offsets. The
 * skill score it used to repeat is the row's own column, and it printed at a
 * different precision one click away (finding 17).
 */
export function ModelsPanel() {
  const models = useModels()
  const [expanded, setExpanded] = useState<string | null>(null)

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 sm:px-6">
        <p className="text-xs text-fg-3">
          Partición temporal, baselines de persistencia/climatología y skill score.
        </p>
        <Button onClick={() => models.refetch()} disabled={models.isFetching}>
          <RefreshCw className={cn('h-3.5 w-3.5', models.isFetching && 'animate-spin')} />
          Actualizar
        </Button>
      </div>

      <div className="px-4 sm:px-6">
        <Panel flush>
          <PanelHeader
            title="Modelos registrados"
            subtitle="Ordenados por fecha de entrenamiento, más reciente primero"
            actions={
              models.data ? (
                <span className="nums text-[10px] text-fg-3">{models.data.count} artefactos</span>
              ) : null
            }
          />
          {models.isLoading ? (
            <TableSkeleton rows={5} />
          ) : models.isError ? (
            <ErrorState
              title="No se pudo consultar el registro"
              description={models.error.message}
              action={<Button onClick={() => models.refetch()}>Reintentar</Button>}
            />
          ) : !models.data?.models.length ? (
            <EmptyState
              icon={Boxes}
              title="Sin modelos entrenados"
              description="Ejecuta el job de entrenamiento (ml_training.py) para registrar el primer modelo en model_registry."
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs">
                <thead>
                  <tr className="border-b border-line">
                    <th className="w-8 px-2 py-2" aria-label="Detalle" />
                    <th className="label px-4 py-2 text-left font-normal">Modelo</th>
                    <th className="label px-4 py-2 text-left font-normal">Objetivo</th>
                    <th className="label px-4 py-2 text-right font-normal">Horizonte</th>
                    <th className="label px-4 py-2 text-right font-normal">RMSE</th>
                    <th className="label px-4 py-2 text-right font-normal">MAE</th>
                    <th className="label px-4 py-2 text-right font-normal">R²</th>
                    <th className="label px-4 py-2 text-right font-normal" title="1 − rmse / persistencia; en lluvia 1 − brier / persistencia">
                      Skill
                    </th>
                    <th className="label px-4 py-2 text-right font-normal" title="Fracción de residuos de test dentro del intervalo">
                      Cobertura
                    </th>
                    <th
                      className="label px-4 py-2 text-left font-normal"
                      title="PSI de deriva train→test por feature (máximo); <0,1 estable, ≤0,25 aviso, >0,25 deriva"
                    >
                      Deriva
                    </th>
                    <th className="label px-4 py-2 text-left font-normal">Entrenado</th>
                    <th className="label px-4 py-2 text-left font-normal">Estado</th>
                  </tr>
                </thead>
                <tbody>
                  {models.data.models.map((model) => {
                    const key = `${model.name}-${model.version}`
                    const open = expanded === key
                    const { metrics } = model
                    return (
                      <FragmentRow
                        key={key}
                        open={open}
                        onToggle={() => setExpanded(open ? null : key)}
                        summary={
                          <>
                            <td className="px-4 py-2.5">
                              <div className="font-medium text-fg">{model.name}</div>
                              <div className="nums text-[10px] text-fg-3">{model.version}</div>
                            </td>
                            <td className="px-4 py-2.5 text-fg-2">{model.target}</td>
                            <td className="nums px-4 py-2.5 text-right text-fg-2">
                              +{model.horizon_hours} h
                            </td>
                            <td className="nums px-4 py-2.5 text-right text-fg-2">
                              {formatNumber(metrics.rmse ?? null, PRECISION.error)}
                            </td>
                            <td className="nums px-4 py-2.5 text-right text-fg-2">
                              {formatNumber(metrics.mae ?? null, PRECISION.error)}
                            </td>
                            <td className="nums px-4 py-2.5 text-right text-fg-2">
                              {formatNumber(metrics.r2 ?? null, PRECISION.skill)}
                            </td>
                            <td className="px-4 py-2.5 text-right">
                              <Badge tone={skillTone(metrics.skill_score)}>
                                {formatNumber(metrics.skill_score ?? null, PRECISION.skill)}
                              </Badge>
                            </td>
                            <td className="nums px-4 py-2.5 text-right text-fg-2">
                              {formatPercent(
                                isNum(metrics.coverage) ? metrics.coverage * 100 : null,
                                PRECISION.percent,
                              )}
                              {isNum(model.interval?.level) ? (
                                <span className="ml-1 text-[10px] text-fg-3">
                                  {`n${formatPercent(
                                    (model.interval?.level ?? 0) * 100,
                                    PRECISION.percent,
                                  )}`}
                                </span>
                              ) : null}
                            </td>
                            <td className="px-4 py-2.5">
                              <DriftBadge diagnostics={model.diagnostics} />
                            </td>
                            <td className="nums px-4 py-2.5 text-fg-3">
                              {formatDateTime(model.created_at)}
                            </td>
                            <td className="px-4 py-2.5">
                              <Badge tone={model.stage === 'production' ? 'ok' : 'neutral'}>
                                {stageLabel(model.stage)}
                              </Badge>
                            </td>
                          </>
                        }
                        detail={
                          <DetailGrid
                            split={model.split}
                            interval={model.interval}
                            commit={model.commit}
                            rows={[
                              [
                                'RMSE persistencia',
                                formatNumber(metrics.persistence_rmse ?? null, PRECISION.error),
                              ],
                              [
                                'RMSE climatología',
                                formatNumber(metrics.climatology_rmse ?? null, PRECISION.error),
                              ],
                              ['Brier', formatNumber(metrics.brier ?? null, PRECISION.error)],
                              [
                                'Brier persistencia',
                                formatNumber(metrics.persistence_brier ?? null, PRECISION.error),
                              ],
                              [
                                'Prevalencia lluvia',
                                formatPercent(
                                  isNum(metrics.prevalence) ? metrics.prevalence * 100 : null,
                                  PRECISION.probability,
                                ),
                              ],
                              ['AUC ROC', formatNumber(metrics.auc_roc ?? null, PRECISION.skill)],
                              ['AUC PR', formatNumber(metrics.auc_pr ?? null, PRECISION.skill)],
                            ]}
                          />
                        }
                      />
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </div>
  )
}

/** One model row plus its optional honest-metrics detail row. */
function FragmentRow({
  open,
  onToggle,
  summary,
  detail,
}: {
  open: boolean
  onToggle: () => void
  summary: ReactNode
  detail: ReactNode
}) {
  return (
    <>
      <tr className="border-b border-line/60 last:border-0">
        <td className="px-2 py-2.5 align-top">
          <button
            type="button"
            onClick={onToggle}
            aria-expanded={open}
            aria-label={open ? 'Ocultar detalles' : 'Mostrar detalles'}
            className="flex h-5 w-5 items-center justify-center rounded border border-line text-fg-3 hover:border-line-strong hover:text-fg"
          >
            {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          </button>
        </td>
        {summary}
      </tr>
      {open ? (
        <tr className="border-b border-line/60 bg-panel-2/40 last:border-0">
          <td />
          <td colSpan={11} className="px-4 py-3">
            {detail}
          </td>
        </tr>
      ) : null}
    </>
  )
}

function DetailGrid({
  split,
  interval,
  commit,
  rows,
}: {
  split: { kind?: string; train_end?: string | null; val_end?: string | null; test_start?: string | null } | null
  interval: { level?: number | null; lower_offset?: number | null; upper_offset?: number | null } | null
  commit: string | null
  rows: Array<[string, ReactNode]>
}) {
  const items: Array<[string, ReactNode]> = rows.filter(([, value]) => value !== '—')
  if (split) {
    items.push(['Partición', split.kind ?? '—'])
    if (split.test_start) items.push(['Test desde', formatDateTime(split.test_start)])
  }
  if (interval) {
    if (isNum(interval.lower_offset) && isNum(interval.upper_offset)) {
      items.push([
        'Offsets intervalo',
        `${formatSigned(interval.lower_offset, PRECISION.error)} / ${
          formatSigned(interval.upper_offset, PRECISION.error)
        } °C`,
      ])
    }
  }
  if (commit) items.push(['Commit', commit.slice(0, 10)])

  return (
    <dl className="grid grid-cols-2 gap-x-8 gap-y-1.5 sm:grid-cols-4">
      {items.map(([label, value]) => (
        <div key={label} className="flex justify-between gap-3">
          <dt className="text-fg-3">{label}</dt>
          <dd className="nums text-fg-2">{value}</dd>
        </div>
      ))}
    </dl>
  )
}
