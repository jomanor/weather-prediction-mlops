/**
 * U6 keyboard shortcuts. Sections are reachable with `g` followed by a key;
 * the palette lists this table so the bindings are discoverable.
 */

export const SECTION_SHORTCUTS = [
  { key: 'o', to: '/', label: 'Red' },
  { key: 's', to: '/stations', label: 'Estación' },
  { key: 'b', to: '/benchmark', label: 'Benchmark' },
  { key: 'm', to: '/models', label: 'Modelos' },
  { key: 'a', to: '/analytics', label: 'Analítica' },
] as const

export type SectionShortcut = (typeof SECTION_SHORTCUTS)[number]

export function resolveSectionShortcut(key: string): SectionShortcut | null {
  const normalised = key.toLowerCase()
  return SECTION_SHORTCUTS.find((shortcut) => shortcut.key === normalised) ?? null
}

/** Typing in a field (or a contenteditable) must never trigger a shortcut. */
export function isEditableTarget(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null
  if (!element) return false
  return (
    element.tagName === 'INPUT' ||
    element.tagName === 'TEXTAREA' ||
    element.tagName === 'SELECT' ||
    element.isContentEditable === true
  )
}
