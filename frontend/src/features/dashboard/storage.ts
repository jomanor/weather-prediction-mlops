import {
  DAY_OPTIONS,
  HOUR_OPTIONS,
  MAX_WIDGETS,
  WIDGET_TYPES,
  type DashboardWidget,
  type WidgetConfig,
  type WidgetType,
} from '@/features/dashboard/types'

/**
 * The dashboard layout lives in the browser and nowhere else: one versioned
 * key, no account, no server copy. Read and write can never throw — a broken
 * blob must degrade to the empty state, not to a blank page.
 *
 * Format (`meteoml.dashboard.v1`):
 *
 *   { "version": 1, "widgets": [{ "id", "type", "city"?, "days"?, "hours"? }] }
 *
 * `id` is persisted so React keys (and the up/down buttons' focus) survive a
 * reload; it is generated only when the blob has no usable one.
 */
export const DASHBOARD_KEY = 'meteoml.dashboard.v1'
export const DASHBOARD_VERSION = 1

let counter = 0

/** A fresh widget id. `crypto.randomUUID` when the context has it (https/localhost). */
export function createWidgetId(): string {
  counter += 1
  const uuid =
    typeof globalThis.crypto?.randomUUID === 'function' ? globalThis.crypto.randomUUID() : null
  return `w-${uuid ?? `${Date.now().toString(36)}-${counter}`}`
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function isWidgetType(value: unknown): value is WidgetType {
  return typeof value === 'string' && (WIDGET_TYPES as readonly string[]).includes(value)
}

/** A stored `days`/`hours` is trusted only when it is one of the offered options. */
function oneOf<T extends number>(options: readonly T[], value: unknown): T | undefined {
  return typeof value === 'number' && (options as readonly number[]).includes(value)
    ? (value as T)
    : undefined
}

/** A stored city must be a non-empty string; anything else falls back to the registry. */
function readCity(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  return trimmed === '' ? undefined : trimmed
}

/** Only the keys this format knows are copied; no unknown key survives the next write. */
function withConfig(target: DashboardWidget, source: WidgetConfig): DashboardWidget {
  const city = readCity(source.city)
  const days = oneOf(DAY_OPTIONS, source.days)
  const hours = oneOf(HOUR_OPTIONS, source.hours)
  if (city !== undefined) target.city = city
  if (days !== undefined) target.days = days
  if (hours !== undefined) target.hours = hours
  return target
}

/**
 * Parse one stored blob into a layout that is safe to render.
 *
 * Defensive by contract: malformed JSON, a non-object root, a different
 * `version`, a non-array `widgets`, an unknown `type`, an invalid option, a
 * missing/duplicate id and a list past {@link MAX_WIDGETS} all resolve to
 * "render what is valid, drop the rest" — never to a throw.
 *
 * `makeId` is injectable so the ids a test sees are deterministic; production
 * always uses {@link createWidgetId}.
 */
export function parseDashboard(
  raw: string | null,
  makeId: () => string = createWidgetId,
): DashboardWidget[] {
  if (!raw) return []

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return []
  }

  const root = asRecord(parsed)
  /* A future format is not guessed at: an unknown version reads as "no layout"
     and the empty state explains how to rebuild it. */
  if (!root || root.version !== DASHBOARD_VERSION) return []

  const entries = Array.isArray(root.widgets) ? root.widgets : []
  const seen = new Set<string>()
  const widgets: DashboardWidget[] = []

  for (const entry of entries) {
    if (widgets.length >= MAX_WIDGETS) break
    const record = asRecord(entry)
    if (!record || !isWidgetType(record.type)) continue

    const storedId = typeof record.id === 'string' ? record.id.trim() : ''
    /* A duplicate id would be a duplicate React key *and* two cards with the
       same move/remove handlers, so the second one gets a fresh id. */
    const id = storedId !== '' && !seen.has(storedId) ? storedId : makeId()
    seen.add(id)

    widgets.push(
      withConfig(
        { id, type: record.type },
        {
          city: readCity(record.city),
          days: oneOf(DAY_OPTIONS, record.days),
          hours: oneOf(HOUR_OPTIONS, record.hours),
        },
      ),
    )
  }

  return widgets
}

/** Current layout, or `[]` when storage is unavailable or holds nothing usable. */
export function readDashboard(): DashboardWidget[] {
  try {
    return parseDashboard(localStorage.getItem(DASHBOARD_KEY))
  } catch {
    /* Storage blocked (private mode, a hostile embed): an in-memory panel is
       still a usable panel. */
    return []
  }
}

/** Write-through save. A full or blocked storage keeps the layout in memory. */
export function writeDashboard(widgets: readonly DashboardWidget[]): void {
  try {
    const payload = {
      version: DASHBOARD_VERSION,
      widgets: widgets.map((widget) => withConfig({ id: widget.id, type: widget.type }, widget)),
    }
    localStorage.setItem(DASHBOARD_KEY, JSON.stringify(payload))
  } catch {
    /* storage unavailable or full */
  }
}
