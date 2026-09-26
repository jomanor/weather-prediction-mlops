import { Search, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'

import { useCities } from '@/api/queries'
import { NAV_ITEMS } from '@/app/registry'
import {
  filterPaletteItems,
  type PaletteItem,
} from '@/components/shell/command-palette'
import { SECTION_SHORTCUTS } from '@/components/shell/shortcuts'
import { cn } from '@/lib/cn'

const LIST_ID = 'command-palette-list'

/**
 * U6: hand-rolled command palette (no new dependency). Opens on Cmd/Ctrl-K,
 * lists sections and stations, closes on Escape/backdrop click, traps focus and
 * follows the ARIA combobox pattern (`aria-activedescendant`, not DOM focus).
 */
export function CommandPalette() {
  const navigate = useNavigate()
  const { data: cities } = useCities()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [highlight, setHighlight] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const restoreRef = useRef<HTMLElement | null>(null)

  const items = useMemo<PaletteItem[]>(
    () => [
      ...NAV_ITEMS.map((item) => ({
        id: `route:${item.to}`,
        label: item.label,
        group: 'Secciones' as const,
        to: item.to,
      })),
      ...(cities ?? []).map((city) => ({
        id: `station:${city.name}`,
        label: city.name,
        group: 'Estaciones' as const,
        to: `/stations?city=${encodeURIComponent(city.name)}`,
        keywords: 'estación estacion',
      })),
    ],
    [cities],
  )

  const results = useMemo(() => filterPaletteItems(items, query), [items, query])

  useEffect(() => setHighlight(0), [query, open])

  /* Cmd/Ctrl-K toggles the palette from anywhere in the shell. */
  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== 'k') return
      event.preventDefault()
      setOpen((value) => !value)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  /* Move focus into the dialog and restore it on close. */
  useEffect(() => {
    if (!open) return
    restoreRef.current = document.activeElement as HTMLElement | null
    inputRef.current?.focus()
    return () => {
      restoreRef.current?.focus?.()
      restoreRef.current = null
    }
  }, [open])

  const close = useCallback(() => {
    setOpen(false)
    setQuery('')
  }, [])

  const choose = useCallback(
    (item: PaletteItem) => {
      close()
      navigate(item.to)
    },
    [close, navigate],
  )

  const onInputKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (!results.length) return
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setHighlight((index) => (index + 1) % results.length)
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setHighlight((index) => (index - 1 + results.length) % results.length)
    } else if (event.key === 'Home') {
      event.preventDefault()
      setHighlight(0)
    } else if (event.key === 'End') {
      event.preventDefault()
      setHighlight(results.length - 1)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      choose(results[highlight])
    }
  }

  /* Keep Tab inside the dialog while it is open. */
  const onPanelKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      close()
      return
    }
    if (event.key !== 'Tab') return
    const root = panelRef.current
    if (!root) return
    const focusable = Array.from(
      root.querySelectorAll<HTMLElement>('input, button:not([disabled])'),
    )
    if (focusable.length < 2) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-fg/25 p-4 pt-[12vh] backdrop-blur-[2px] motion-reduce:backdrop-blur-none"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) close()
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label="Paleta de comandos"
        onKeyDown={onPanelKeyDown}
        className="w-full max-w-lg animate-in overflow-hidden rounded-[var(--radius-panel)] border border-line bg-panel shadow-lg"
      >
        <div className="flex items-center gap-2 border-b border-line px-3">
          <Search aria-hidden className="h-3.5 w-3.5 shrink-0 text-fg-3" />
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded="true"
            aria-controls={LIST_ID}
            aria-autocomplete="list"
            aria-activedescendant={results.length ? `palette-option-${highlight}` : undefined}
            aria-label="Buscar secciones y estaciones"
            placeholder="Buscar sección o estación…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={onInputKeyDown}
            className="h-11 w-full bg-transparent text-sm text-fg placeholder:text-fg-3 focus:outline-none"
          />
          <button
            type="button"
            aria-label="Cerrar la paleta"
            onClick={close}
            className="shrink-0 rounded p-1 text-fg-3 transition-colors hover:text-fg"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {results.length ? (
          <ul
            id={LIST_ID}
            role="listbox"
            aria-label="Resultados"
            className="max-h-80 overflow-y-auto py-1"
          >
            {results.map((item, index) => (
              <li
                key={item.id}
                id={`palette-option-${index}`}
                role="option"
                aria-selected={index === highlight}
                onPointerEnter={() => setHighlight(index)}
                onClick={() => choose(item)}
                className={cn(
                  'flex cursor-pointer items-center justify-between gap-3 px-3 py-2 text-xs transition-colors',
                  index === highlight ? 'bg-panel-2 text-fg' : 'text-fg-2',
                )}
              >
                <span>{item.label}</span>
                <span className="text-[10px] uppercase tracking-wider text-fg-3">{item.group}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="px-3 py-6 text-center text-xs text-fg-3">Sin resultados</p>
        )}

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-line px-3 py-2.5">
          <span className="label">Atajos</span>
          {SECTION_SHORTCUTS.map((shortcut) => (
            <span key={shortcut.key} className="flex items-center gap-1 text-[11px] text-fg-3">
              <Kbd>g</Kbd>
              <Kbd>{shortcut.key}</Kbd>
              {shortcut.label}
            </span>
          ))}
          <span className="flex items-center gap-1 text-[11px] text-fg-3">
            <Kbd>/</Kbd> buscar estación
          </span>
          <span className="flex items-center gap-1 text-[11px] text-fg-3">
            <Kbd>⌘K</Kbd> paleta
          </span>
        </div>
      </div>
    </div>
  )
}

function Kbd({ children }: { children: string }) {
  return (
    <kbd className="rounded border border-line bg-panel-2 px-1 font-mono text-[10px] leading-4 text-fg-2">
      {children}
    </kbd>
  )
}
