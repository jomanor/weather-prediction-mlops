import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

/** Shared x-domain for the history, benchmark and residual charts (epoch ms). */
export type TimeDomain = [number, number]

interface ChartSyncValue {
  domain: TimeDomain | null
  /** True while a brush selection (not the page window) owns the domain. */
  brushed: boolean
  /** Bumped when the domain is cleared, so a chart can remount its brush. */
  resetToken: number
  /** Anchor the shared window: `hours` long, ending at `anchor` (default: now). */
  setWindow: (hours: number, anchor?: number) => void
  /** Publish a brushed selection; an empty/inverted range clears it. */
  setDomain: (domain: TimeDomain | null) => void
  /** Clear the shared domain and reset the brushes. */
  reset: () => void
}

const HOUR_MS = 3_600_000

/* Charts can render outside the provider; then each uses its own data extent. */
const DETACHED: ChartSyncValue = {
  domain: null,
  brushed: false,
  resetToken: 0,
  setWindow: () => {},
  setDomain: () => {},
  reset: () => {},
}

const ChartSyncContext = createContext<ChartSyncValue | null>(null)

export function ChartSyncProvider({ children }: { children: ReactNode }) {
  const [domain, setDomainState] = useState<TimeDomain | null>(null)
  const [brushed, setBrushed] = useState(false)
  const [resetToken, setResetToken] = useState(0)

  const setWindow = useCallback((hours: number, anchor?: number) => {
    const end = anchor ?? Date.now()
    setDomainState([end - hours * HOUR_MS, end])
    setBrushed(false)
    setResetToken((token) => token + 1)
  }, [])

  const setDomain = useCallback((next: TimeDomain | null) => {
    if (next && next[1] > next[0]) {
      setDomainState(next)
      setBrushed(true)
    } else {
      setDomainState(null)
      setBrushed(false)
    }
  }, [])

  const reset = useCallback(() => {
    setDomainState(null)
    setBrushed(false)
    setResetToken((token) => token + 1)
  }, [])

  const value = useMemo(
    () => ({ domain, brushed, resetToken, setWindow, setDomain, reset }),
    [domain, brushed, resetToken, setWindow, setDomain, reset],
  )
  return <ChartSyncContext.Provider value={value}>{children}</ChartSyncContext.Provider>
}

export function useChartSync(): ChartSyncValue {
  return useContext(ChartSyncContext) ?? DETACHED
}

/**
 * Anchors the shared window to the page's data extent and clears it on unmount.
 * The provider lives above the router, so without the cleanup a brushed or
 * windowed domain would leak into the next section.
 */
export function useChartWindow(hours: number, anchor: string | null | undefined) {
  const { setWindow, reset, ...rest } = useChartSync()

  const anchorMs = useMemo(() => {
    if (!anchor) return null
    const parsed = new Date(anchor).getTime()
    return Number.isFinite(parsed) ? parsed : null
  }, [anchor])

  useEffect(() => {
    if (anchorMs === null) return
    setWindow(hours, anchorMs)
  }, [anchorMs, hours, setWindow])

  useEffect(() => reset, [reset])

  const resetWindow = useCallback(() => {
    if (anchorMs === null) reset()
    else setWindow(hours, anchorMs)
  }, [anchorMs, hours, setWindow, reset])

  return { ...rest, resetWindow }
}
