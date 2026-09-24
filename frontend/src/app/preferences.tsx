import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { readChartPalette, type ChartPalette, type ThemeChoice } from '@/lib/chart-theme'
import type { UnitSystem } from '@/lib/format'

const THEME_KEY = 'meteoml.theme'
const UNITS_KEY = 'meteoml.units'

interface PreferencesValue {
  theme: ThemeChoice
  isDark: boolean
  setTheme: (theme: ThemeChoice) => void
  units: UnitSystem
  setUnits: (units: UnitSystem) => void
  locale: string
  palette: ChartPalette
}

const PreferencesContext = createContext<PreferencesValue | null>(null)

function readStored<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const value = localStorage.getItem(key)
    return allowed.includes(value as T) ? (value as T) : fallback
  } catch {
    return fallback
  }
}

function prefersDark(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches
}

export function PreferencesProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeChoice>(() =>
    readStored(THEME_KEY, ['light', 'dark', 'system'] as const, 'system'),
  )
  const [units, setUnitsState] = useState<UnitSystem>(() =>
    readStored(UNITS_KEY, ['metric', 'imperial'] as const, 'metric'),
  )
  const [systemDark, setSystemDark] = useState(prefersDark)

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = (event: MediaQueryListEvent) => setSystemDark(event.matches)
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  const isDark = theme === 'dark' || (theme === 'system' && systemDark)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark)
  }, [isDark])

  const setTheme = useCallback((next: ThemeChoice) => {
    setThemeState(next)
    try {
      localStorage.setItem(THEME_KEY, next)
    } catch {
      /* storage unavailable */
    }
  }, [])

  const setUnits = useCallback((next: UnitSystem) => {
    setUnitsState(next)
    try {
      localStorage.setItem(UNITS_KEY, next)
    } catch {
      /* storage unavailable */
    }
  }, [])

  // Recomputed after paint so the palette reads the freshly applied tokens.
  const palette = useMemo(() => readChartPalette(isDark), [isDark])

  const value = useMemo<PreferencesValue>(
    () => ({ theme, isDark, setTheme, units, setUnits, locale: 'es-ES', palette }),
    [theme, isDark, setTheme, units, setUnits, palette],
  )

  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>
}

export function usePreferences(): PreferencesValue {
  const context = useContext(PreferencesContext)
  if (!context) throw new Error('usePreferences must be used within PreferencesProvider')
  return context
}
