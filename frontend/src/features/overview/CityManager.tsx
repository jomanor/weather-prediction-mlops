import { MapPinPlus, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { useCities } from '@/api/queries'
import { Button } from '@/components/ui/Button'
import { useAddCity, useGeoSearch, useRemoveCity } from '@/features/overview/queries'
import { cn } from '@/lib/cn'
import { formatLatitude, formatLongitude } from '@/lib/format'

/**
 * Station registry editor: pick a location through the geocoding proxy and
 * remove stations from monitoring. Historical observations are never deleted
 * upstream; removal only stops future ingestion.
 *
 * Removal is destructive (it drops the city from the registry ingestion reads),
 * so it goes through an inline confirmation that names the city and the
 * consequence. Cancel is focused by default and Escape also cancels.
 */
export function CityManager({ className }: { className?: string }) {
  const citiesQuery = useCities()
  const addCity = useAddCity()
  const removeCity = useRemoveCity()

  const [term, setTerm] = useState('')
  const [debounced, setDebounced] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [pendingRemoval, setPendingRemoval] = useState<string | null>(null)
  const [restoreFocusTo, setRestoreFocusTo] = useState<string | null>(null)
  const removeButtonRefs = useRef(new Map<string, HTMLButtonElement>())

  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(term.trim()), 350)
    return () => window.clearTimeout(timer)
  }, [term])

  /* Return focus to the row's remove button once the confirmation closes. */
  useEffect(() => {
    if (pendingRemoval !== null || !restoreFocusTo) return
    removeButtonRefs.current.get(restoreFocusTo)?.focus()
    setRestoreFocusTo(null)
  }, [pendingRemoval, restoreFocusTo])

  const geoQuery = useGeoSearch(debounced)

  const cities = citiesQuery.data ?? []
  const results = geoQuery.data?.results ?? []
  const known = new Set(cities.map((city) => city.name.trim().toLowerCase()))

  const add = (result: (typeof results)[number]) => {
    setError(null)
    addCity.mutate(
      { name: result.name, latitude: result.latitude, longitude: result.longitude },
      { onError: (cause) => setError(cause.message) },
    )
  }

  const remove = (name: string) => {
    setError(null)
    removeCity.mutate(name, { onError: (cause) => setError(cause.message) })
  }

  /** Opens the confirmation only; no request is issued until it is confirmed. */
  const requestRemoval = (name: string) => {
    setError(null)
    setPendingRemoval(name)
  }

  const cancelRemoval = (name: string) => {
    setPendingRemoval(null)
    setRestoreFocusTo(name)
  }

  const confirmRemoval = (name: string) => {
    remove(name)
    setPendingRemoval(null)
  }

  return (
    <div className={cn('space-y-4', className)}>
      <div>
        <label htmlFor="city-search" className="label">
          Añadir estación
        </label>
        <div className="relative mt-1.5">
          <MapPinPlus className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-3" />
          <input
            id="city-search"
            type="text"
            value={term}
            onChange={(event) => {
              setTerm(event.target.value)
              setError(null)
            }}
            placeholder="Buscar localidad… (mín. 2 letras)"
            className="h-9 w-full max-w-md rounded-[3px] border border-line bg-panel pl-8 pr-3 text-xs text-fg focus:border-accent focus:outline-none"
          />
        </div>

        {debounced.length >= 2 ? (
          <ul className="mt-1.5 max-w-md divide-y divide-line/60 rounded-[3px] border border-line bg-panel">
            {geoQuery.isLoading ? (
              <li className="px-3 py-2 text-xs text-fg-3">Buscando…</li>
            ) : results.length === 0 ? (
              <li className="px-3 py-2 text-xs text-fg-3">
                {geoQuery.isError ? 'No se pudo contactar con el geocoder.' : 'Sin resultados.'}
              </li>
            ) : (
              results.map((result) => {
                const already = known.has(result.name.trim().toLowerCase())
                return (
                  <li
                    key={`${result.name}-${result.latitude}-${result.longitude}`}
                    className="flex items-center justify-between gap-3 px-3 py-1.5"
                  >
                    <span className="min-w-0 text-xs">
                      <span className="font-medium text-fg">{result.name}</span>
                      <span className="ml-1.5 text-fg-3">
                        {[result.admin1, result.country].filter(Boolean).join(', ')}
                      </span>
                    </span>
                    {already ? (
                      <span className="nums shrink-0 text-[10px] text-fg-3">añadida</span>
                    ) : (
                      <button
                        type="button"
                        onClick={() => add(result)}
                        disabled={addCity.isPending}
                        className="nums shrink-0 rounded-[3px] border border-line px-2 py-0.5 text-[11px] text-fg-2 hover:bg-panel-2 hover:text-fg"
                      >
                        Añadir
                      </button>
                    )}
                  </li>
                )
              })
            )}
          </ul>
        ) : null}
      </div>

      {error ? <p className="text-xs text-bad">{error}</p> : null}

      <div>
        <div className="label">Estaciones monitorizadas</div>
        {citiesQuery.isLoading ? (
          <p className="mt-1.5 text-xs text-fg-3">Cargando…</p>
        ) : (
          <ul className="mt-1.5 max-w-2xl columns-1 gap-x-8 gap-y-0.5 sm:columns-2">
            {cities.map((city) => {
              const confirming = pendingRemoval === city.name
              return (
                <li key={city.name} className="break-inside-avoid px-1 py-1">
                  <div className="group flex items-center justify-between gap-2">
                    <span className="nums min-w-0 truncate text-xs text-fg-2">
                      <span className="font-medium text-fg">{city.name}</span>{' '}
                      {formatLatitude(city.latitude)} {formatLongitude(city.longitude)}
                    </span>
                    <button
                      type="button"
                      aria-label={`Quitar ${city.name}`}
                      onClick={() => requestRemoval(city.name)}
                      disabled={removeCity.isPending}
                      ref={(node) => {
                        if (node) removeButtonRefs.current.set(city.name, node)
                        else removeButtonRefs.current.delete(city.name)
                      }}
                      className={cn(
                        'shrink-0 rounded-[3px] p-0.5 text-fg-3 opacity-0 transition-opacity hover:text-bad focus-visible:opacity-100 group-hover:opacity-100',
                        confirming && 'hidden',
                      )}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>

                  {confirming ? (
                    <div
                      role="group"
                      aria-label={`Confirmar quitar ${city.name}`}
                      onKeyDown={(event) => {
                        if (event.key === 'Escape') {
                          event.preventDefault()
                          cancelRemoval(city.name)
                        }
                      }}
                      className="animate-in mt-1.5 rounded-[3px] border border-bad/40 bg-bad/5 p-2"
                    >
                      <p className="text-[11px] leading-snug text-fg-2">
                        Ingesta y datos dejarán de actualizarse para{' '}
                        <span className="font-medium text-fg">{city.name}</span>.
                      </p>
                      <div className="mt-2 flex flex-wrap items-center gap-2">
                        <Button autoFocus size="sm" onClick={() => cancelRemoval(city.name)}>
                          Cancelar
                        </Button>
                        <Button
                          variant="danger"
                          size="sm"
                          disabled={removeCity.isPending}
                          onClick={() => confirmRemoval(city.name)}
                        >
                          Quitar estación
                        </Button>
                      </div>
                    </div>
                  ) : null}
                </li>
              )
            })}
          </ul>
        )}
        <p className="mt-2 max-w-md text-[11px] leading-snug text-fg-3">
          Quitar una estación la excluye de la ingesta, pero conserva su histórico en la base de
          datos. Una estación nueva aparece con valores cuando el productor vuelva a ingerir datos.
        </p>
      </div>
    </div>
  )
}
