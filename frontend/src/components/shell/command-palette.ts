/** Pure filtering for the hand-rolled command palette (U6). */

export interface PaletteItem {
  id: string
  label: string
  group: 'Secciones' | 'Estaciones'
  /** Router target; a station target carries its `city` query param. */
  to: string
  keywords?: string
}

/** Case- and accent-insensitive: "Estacion" must find "Estación". */
export function normalizeTerm(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
}

/**
 * Filter sections and stations by a query. An empty query shows the sections
 * plus the first few stations; a query ranks prefix matches first, then
 * alphabetical, capped at `limit`.
 */
export function filterPaletteItems(
  items: readonly PaletteItem[],
  query: string,
  limit = 10,
): PaletteItem[] {
  const term = normalizeTerm(query.trim())
  if (!term) {
    return [
      ...items.filter((item) => item.group === 'Secciones').slice(0, 5),
      ...items.filter((item) => item.group === 'Estaciones').slice(0, 5),
    ]
  }

  return items
    .map((item) => {
      const label = normalizeTerm(item.label)
      const haystack = normalizeTerm(`${item.label} ${item.keywords ?? ''}`)
      const score = haystack.includes(term) ? (label.startsWith(term) ? 0 : 1) : -1
      return { item, score }
    })
    .filter((entry) => entry.score >= 0)
    .sort((a, b) => a.score - b.score || a.item.label.localeCompare(b.item.label, 'es'))
    .slice(0, limit)
    .map((entry) => entry.item)
}
