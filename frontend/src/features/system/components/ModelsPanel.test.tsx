import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useModels } from '@/features/models/queries'
import { ModelsPanel, stageLabel } from '@/features/system/components/ModelsPanel'

/**
 * Finding 17: one precision authority. The row and the expander printed the same
 * metric at different precisions one click apart (`skill 0,53` vs `0,532`, RMSE
 * 3 dp here and 2 dp in Benchmark). Every number now comes from `PRECISION`.
 *
 * S2: the registry stage is an English lifecycle value rendered in a Spanish
 * table; it is mapped, with the raw value as the fallback.
 */
vi.mock('@/features/models/queries', () => ({ useModels: vi.fn() }))

const mocked = vi.mocked(useModels)

const MODEL = {
  name: 'gbt_temperature_24h',
  version: 'v42',
  target: 'temperature',
  horizon_hours: 24,
  created_at: '2026-09-25T12:00:00Z',
  metrics: {
    rmse: 1.2345,
    mae: 1.2345,
    r2: 0.8765,
    skill_score: 0.5321,
    persistence_rmse: 2.3456,
    climatology_rmse: 3.4567,
    brier: 0.1234,
    persistence_brier: 0.2345,
    prevalence: 0.1834,
    coverage: 0.9123,
    auc_roc: 0.8452,
    auc_pr: 0.6123,
  },
  split: {
    kind: 'temporal',
    train_end: '2026-08-01T00:00:00Z',
    val_end: null,
    test_start: '2026-09-01T00:00:00Z',
  },
  interval: { level: 0.9, lower_offset: -1.2345, upper_offset: 1.4345 },
  diagnostics: {
    by_city: [],
    by_hour_of_day: [],
    by_rain_bucket: [],
    drift_psi: { temperature: 0.1234 },
  },
  commit: 'abc123def4567890',
  stage: 'production',
}

function mockModels(models: unknown[] = [MODEL]) {
  mocked.mockReturnValue({
    data: { count: models.length, models },
    isLoading: false,
    isError: false,
    isFetching: false,
    error: null,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof useModels>)
}

describe('ModelsPanel stage labels', () => {
  beforeEach(() => {
    mocked.mockReset()
  })

  it('names the lifecycle stages in Spanish', () => {
    expect(stageLabel('production')).toBe('Producción')
    expect(stageLabel('staging')).toBe('Staging')
    expect(stageLabel('superseded')).toBe('Sustituido')
    expect(stageLabel('none')).toBe('Sin estado')
    /* A stage the registry adds later is shown, not swallowed. */
    expect(stageLabel('quarantined')).toBe('quarantined')
  })

  it('prints the mapped stage instead of the raw registry value', () => {
    mockModels([
      { ...MODEL, name: 'produccion', stage: 'production' },
      { ...MODEL, name: 'candidato', stage: 'staging' },
    ])
    render(<ModelsPanel />)

    expect(screen.getByText('Producción')).toBeInTheDocument()
    expect(screen.getByText('Staging')).toBeInTheDocument()
    expect(screen.queryByText('production')).not.toBeInTheDocument()
    expect(screen.queryByText('staging')).not.toBeInTheDocument()
  })
})

describe('ModelsPanel precision', () => {
  beforeEach(() => {
    mocked.mockReset()
  })

  it('prints each metric at its canonical precision in the row', () => {
    mockModels()
    render(<ModelsPanel />)

    /* RMSE and MAE share PRECISION.error (2 dp), not 3 dp. */
    expect(screen.getAllByText('1,23')).toHaveLength(2)
    expect(screen.queryByText('1,235')).not.toBeInTheDocument()
    /* Skill and R² share PRECISION.skill (2 dp). */
    expect(screen.getByText('0,53')).toBeInTheDocument()
    expect(screen.getByText('0,88')).toBeInTheDocument()
    /* Coverage and the interval level are whole percent. */
    expect(screen.getByText('91%')).toBeInTheDocument()
    expect(screen.getByText('n90%')).toBeInTheDocument()
  })

  it('does not drift to another precision once the row is expanded', async () => {
    mockModels()
    render(<ModelsPanel />)

    await userEvent.click(screen.getByRole('button', { name: 'Mostrar detalles' }))

    /* The expansion carries only facts the row does not: the persistence and
       climatology baselines, Brier, AUC and the split. The duplicated
       "Skill score" row (the reported `0,532`) is gone. */
    expect(screen.queryByText('Skill score')).not.toBeInTheDocument()
    expect(screen.queryByText('0,532')).not.toBeInTheDocument()
    expect(screen.getByText('RMSE persistencia')).toBeInTheDocument()
    expect(screen.getByText('2,35')).toBeInTheDocument()
    expect(screen.getByText('Brier')).toBeInTheDocument()
    expect(screen.getByText('AUC ROC')).toBeInTheDocument()
    expect(screen.getByText('Offsets intervalo')).toBeInTheDocument()

    /* One skill number, one precision, on both sides of the click. */
    expect(within(screen.getByRole('table')).getAllByText('0,53')).toHaveLength(1)
  })
})

/**
 * Finding 15: Calidad and Benchmark never had a refresh button, Modelos did —
 * the same incoherence in the other direction. The registry query is cached and
 * refetched on focus (`src/test/query-defaults.test.ts`), so the button went and
 * `Reintentar` stayed in the error state, where a human knows something the
 * poll does not.
 */
describe('ModelsPanel has no manual refresh control', () => {
  beforeEach(() => {
    mocked.mockReset()
  })

  it('renders no Actualizar button next to the panel blurb', () => {
    mockModels()
    render(<ModelsPanel />)
    expect(screen.queryByRole('button', { name: /Actualizar/ })).not.toBeInTheDocument()
  })

  it('keeps the Reintentar button in the error state', () => {
    mocked.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      isFetching: false,
      error: new Error('registro caído'),
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useModels>)
    render(<ModelsPanel />)
    expect(screen.getByRole('button', { name: 'Reintentar' })).toBeInTheDocument()
  })
})
