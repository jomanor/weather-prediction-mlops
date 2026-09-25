import { createContext, useCallback, useContext, useMemo, type ReactNode } from 'react'

import { urlString, useUrlState } from '@/hooks/useUrlState'

/** URL-backed selection shared by the map, the station table and the charts. */
interface StationSelectionValue {
  selectedCity: string | null
  selectCity: (city: string | null) => void
}

/* Stable schema: `useUrlState` relies on schema identity for callback stability. */
const CITY_SCHEMA = { city: urlString('') }

const StationSelectionContext = createContext<StationSelectionValue | null>(null)

export function StationSelectionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useUrlState(CITY_SCHEMA)

  const selectCity = useCallback(
    (next: string | null) => setState({ city: next ?? '' }),
    [setState],
  )

  const value = useMemo<StationSelectionValue>(
    () => ({ selectedCity: state.city || null, selectCity }),
    [state.city, selectCity],
  )

  return (
    <StationSelectionContext.Provider value={value}>{children}</StationSelectionContext.Provider>
  )
}

export function useStationSelection(): StationSelectionValue {
  const value = useContext(StationSelectionContext)
  if (!value) throw new Error('useStationSelection must be used within StationSelectionProvider')
  return value
}

/**
 * Resolve the URL-backed city against the loaded registry. An unknown value is
 * treated as no selection (the caller falls back to the first city), matching
 * the overview and the map, which only ever highlight a known city. While the
 * registry is still loading (`loaded` false) the URL value is trusted so the
 * station queries can start without waiting for the list.
 */
export function resolveCity(
  selectedCity: string | null,
  cities: readonly string[],
  loaded: boolean,
): string {
  if (selectedCity && (!loaded || cities.includes(selectedCity))) return selectedCity
  return cities[0] ?? ''
}
