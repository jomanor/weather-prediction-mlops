import { Search, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { useCities } from '@/api/queries'
import { cn } from '@/lib/cn'

/** Command-palette style station lookup over the cities the backend knows. */
export function StationSearch() {
  const navigate = useNavigate()
  const { data: cities, isLoading } = useCities()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [highlight, setHighlight] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const results = useMemo(() => {
    const term = query.trim().toLowerCase()
    const list = cities ?? []
    if (!term) return list.slice(0, 8)
    return list.filter((city) => city.toLowerCase().includes(term)).slice(0, 8)
  }, [cities, query])

  useEffect(() => setHighlight(0), [query])

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === '/' && document.activeElement?.tagName !== 'INPUT') {
        event.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  const choose = (city: string) => {
    setQuery('')
    setOpen(false)
    navigate(`/stations?city=${encodeURIComponent(city)}`)
  }

  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Escape') {
      setOpen(false)
      return
    }
    if (!results.length) return
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setHighlight((index) => (index + 1) % results.length)
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setHighlight((index) => (index - 1 + results.length) % results.length)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      choose(results[highlight])
    }
  }

  return (
    <div ref={containerRef} className="relative w-full max-w-xs">
      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-3" />
      <input
        ref={inputRef}
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-controls="station-search-results"
        aria-label="Buscar estación"
        placeholder={isLoading ? 'Cargando estaciones…' : 'Buscar estación  /'}
        value={query}
        onChange={(event) => {
          setQuery(event.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
        className={cn(
          'h-8 w-full rounded-md border border-line bg-panel-2 pl-8 pr-7 text-xs text-fg',
          'placeholder:text-fg-3 focus:border-accent focus:outline-none',
        )}
      />
      {query ? (
        <button
          type="button"
          aria-label="Limpiar búsqueda"
          onClick={() => {
            setQuery('')
            inputRef.current?.focus()
          }}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-fg-3 hover:text-fg"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      ) : null}

      {open && results.length > 0 ? (
        <ul
          id="station-search-results"
          role="listbox"
          className="panel-shadow absolute left-0 right-0 top-full z-50 mt-1 max-h-64 overflow-y-auto rounded-md border border-line bg-panel py-1"
        >
          {results.map((city, index) => (
            <li key={city} role="option" aria-selected={index === highlight}>
              <button
                type="button"
                onPointerEnter={() => setHighlight(index)}
                onClick={() => choose(city)}
                className={cn(
                  'flex w-full items-center px-3 py-1.5 text-left text-xs',
                  index === highlight ? 'bg-panel-2 text-fg' : 'text-fg-2',
                )}
              >
                {city}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}
