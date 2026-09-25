import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

/** Shared x-domain for the history, benchmark and residual charts (epoch ms). */
export type TimeDomain = [number, number]

interface ChartSyncValue {
  domain: TimeDomain | null
  /** Anchor the shared window: `hours` long, ending at `anchor` (default: now). */
  setWindow: (hours: number, anchor?: number) => void
}

const HOUR_MS = 3_600_000

/* Charts can render outside the provider; then each uses its own data extent. */
const DETACHED: ChartSyncValue = { domain: null, setWindow: () => {} }

const ChartSyncContext = createContext<ChartSyncValue | null>(null)

export function ChartSyncProvider({ children }: { children: ReactNode }) {
  const [domain, setDomain] = useState<TimeDomain | null>(null)

  const setWindow = useCallback((hours: number, anchor?: number) => {
    const end = anchor ?? Date.now()
    setDomain([end - hours * HOUR_MS, end])
  }, [])

  const value = useMemo(() => ({ domain, setWindow }), [domain, setWindow])
  return <ChartSyncContext.Provider value={value}>{children}</ChartSyncContext.Provider>
}

export function useChartSync(): ChartSyncValue {
  return useContext(ChartSyncContext) ?? DETACHED
}
