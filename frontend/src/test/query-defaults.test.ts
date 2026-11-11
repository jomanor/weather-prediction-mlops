import { readFileSync } from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * Finding 15: a poll already runs (300 s on `/weather/current`, `/map/stations`
 * and `/weather/series`), so the four manual `Actualizar` buttons only moved the
 * clock. They are gone and window focus is the human's refetch gesture instead.
 *
 * This reads `main.tsx` rather than importing it: the module mounts the app at
 * import time. There is no request storm to worry about in the suite — every
 * test builds its own `QueryClient`, and jsdom never dispatches a real window
 * focus event, so the default only fires in a browser. The defaults are pinned
 * here so a future edit cannot silently go back to `false` (or move them out of
 * this file) without someone reading this comment first.
 */
const MAIN = readFileSync(path.join(process.cwd(), 'src/main.tsx'), 'utf8')

describe('React Query defaults', () => {
  it('refetches stale queries when the window regains focus', () => {
    expect(MAIN).toMatch(/refetchOnWindowFocus:\s*true/)
    expect(MAIN).not.toMatch(/refetchOnWindowFocus:\s*false/)
  })

  it('keeps the 60 s stale window and a single retry', () => {
    expect(MAIN).toMatch(/staleTime:\s*60_000/)
    expect(MAIN).toMatch(/retry:\s*1\b/)
  })
})
