import { PageHeader } from '@/components/ui/PageHeader'
import { Segmented } from '@/components/ui/Segmented'
import { BenchmarkPanel } from '@/features/system/components/BenchmarkPanel'
import { ModelsPanel } from '@/features/system/components/ModelsPanel'
import { DataQualityMeter } from '@/features/system/components/DataQualityMeter'
import { urlOption, useUrlState, type UrlCodec } from '@/hooks/useUrlState'

export const SYSTEM_TABS = ['calidad', 'modelos', 'benchmark'] as const
export type SystemTab = (typeof SYSTEM_TABS)[number]

const TABS: ReadonlyArray<{ value: SystemTab; label: string }> = [
  { value: 'calidad', label: 'Calidad' },
  { value: 'modelos', label: 'Modelos' },
  { value: 'benchmark', label: 'Benchmark' },
]

/* Stable module-scope schema: `tab` is the only Sistema state. */
const TAB_SCHEMA: { tab: UrlCodec<SystemTab> } = { tab: urlOption(SYSTEM_TABS, 'calidad') }

const DESCRIPTION: Record<SystemTab, string> = {
  calidad: 'Cobertura y antigüedad de los datos que alimentan al modelo.',
  modelos: 'Artefactos entrenados: skill frente al baseline, cobertura y deriva.',
  benchmark: 'El modelo frente a la observación y frente a AEMET sobre la misma ventana.',
}

/**
 * Sistema: the MLOps trust surface. Three tabs on one URL (`?tab=`), each one
 * the content of a former top-level page — Calidad (data quality), Modelos
 * (registry) and Benchmark (model vs AEMET). The old `/benchmark` and `/models`
 * paths redirect here.
 *
 * The fourth tab, Analítica (audit 16), is gone: the diurnal heatmap and the
 * wind rose are the only views it carried that anyone opened, and they now live
 * behind the station page header (`ClimateViewsDialog`), so the legacy
 * `/analytics` path redirects to `/stations`. A stale `?tab=analitica` link is
 * not a dead end either: `urlOption` resolves it to Calidad, like any other
 * unknown tab.
 */
export function SystemPage() {
  const [state, setState] = useUrlState(TAB_SCHEMA)
  const tab = state.tab

  return (
    <div className="space-y-4 pb-8">
      <PageHeader title="Sistema" description={DESCRIPTION[tab]} />

      <div className="px-4 sm:px-6">
        <Segmented
          value={tab}
          onChange={(value) => setState({ tab: value }, { replace: false })}
          options={TABS}
          label="Sección de Sistema"
          /* Wraps instead of overflowing if a translation or a larger font
             makes the labels wider than a 320 px phone. */
          className="flex-wrap"
        />
      </div>

      {tab === 'calidad' ? (
        <div className="px-4 sm:px-6">
          <DataQualityMeter />
        </div>
      ) : tab === 'modelos' ? (
        <ModelsPanel />
      ) : (
        <BenchmarkPanel />
      )}
    </div>
  )
}
