import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * Guards the anti-vibe-code contract in `docs/app-rethink.md` §3.
 *
 * The design system is only real if it cannot silently regress, so this test
 * reads the token sheet and the primitive sources and fails on the generated
 * look: gradients, glows, oversized type, stacked shadows, one-off spacing.
 *
 * Scope: `styles/index.css`, `components/ui`, `components/charts`. The feature
 * pages are being restructured by another owner and join the guard afterwards.
 */
const ROOT = process.cwd()
const CSS_PATH = path.join(ROOT, 'src/styles/index.css')

/** The only allowed type steps, in px. */
const TYPE_STEPS = [10, 11, 12, 13, 14, 16, 18, 22]
/** Nothing above this may exist in the scale, whatever a consumer writes. */
const TYPE_CEILING = 22
const RADIUS_CEILING = 6

const BANNED: { pattern: RegExp; why: string }[] = [
  { pattern: /\bbg-gradient-to-/, why: 'gradients' },
  { pattern: /\bbackdrop-blur\b/, why: 'blurred overlays' },
  { pattern: /\bdrop-shadow\b/, why: 'glows/drop shadows' },
  { pattern: /\bshadow-(sm|md|lg|xl|2xl)\b/, why: 'one shadow token (`shadow-ui`)' },
  { pattern: /\brounded-(sm|md|lg|xl|2xl|3xl)\b/, why: 'use rounded-chip/-control/-panel' },
  { pattern: /\btext-(xl|2xl|3xl|4xl|5xl|6xl|7xl|8xl|9xl)\b/, why: 'display type is banned' },
  { pattern: /text-\[[^\]]*px\]/, why: 'use a step of the type scale' },
  { pattern: /\btransition-all\b/, why: 'animate specific properties only' },
  { pattern: /\banimate-pulse\b/, why: 'use the reduced-motion-gated shimmer' },
]

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = path.join(dir, entry)
    if (statSync(full).isDirectory()) return sourceFiles(full)
    return /\.tsx?$/.test(entry) && !entry.includes('.test.') ? [full] : []
  })
}

/** Token declarations of one `@theme` namespace, in px. */
function tokensInPx(css: string, prefix: string): { name: string; px: number }[] {
  const pattern = new RegExp(`--${prefix}-([a-z0-9-]+):\\s*([\\d.]+)(rem|px);`, 'g')
  return [...css.matchAll(pattern)].map(([, name, raw, unit]) => ({
    name,
    px: unit === 'rem' ? Number(raw) * 16 : Number(raw),
  }))
}

describe('design tokens', () => {
  const css = readFileSync(CSS_PATH, 'utf8')

  it('declares the eight-step type scale and nothing above 22 px', () => {
    const sizes = tokensInPx(css, 'text')
    expect(sizes.length).toBeGreaterThanOrEqual(8)
    for (const token of sizes) {
      expect(token.px, `--text-${token.name} is above the 22 px ceiling`).toBeLessThanOrEqual(
        TYPE_CEILING,
      )
      expect(TYPE_STEPS, `--text-${token.name} is off the type scale`).toContain(token.px)
    }
  })

  it('keeps every radius at 6 px or below', () => {
    for (const token of tokensInPx(css, 'radius')) {
      expect(token.px, `--radius-${token.name} is above 6 px`).toBeLessThanOrEqual(RADIUS_CEILING)
    }
  })

  it('defines the single shadow scale, motion ladder and reduced-motion gate', () => {
    for (const required of [
      '--shadow:',
      '--shadow-sheet:',
      '--motion-fast:',
      '--motion-slow:',
      '--ease-ui:',
      '--nav-h:',
      '--safe-bottom:',
      'prefers-reduced-motion: reduce',
    ]) {
      expect(css, `${required} missing from index.css`).toContain(required)
    }
    /* The sheet/elevation pair is the whole scale: no third shadow token. */
    const shadowTokens = new Set([...css.matchAll(/(--shadow[a-z-]*):/g)].map(([, name]) => name))
    expect([...shadowTokens].sort()).toEqual(['--shadow', '--shadow-sheet'])
  })

  it('keeps light and dark tokens in parity', () => {
    const declarations = (selector: string) => {
      const block = css.split(selector)[1] ?? ''
      return new Set(
        [...block.slice(0, block.indexOf('}')).matchAll(/(--[a-z0-9-]+):/g)].map(([, name]) => name),
      )
    }
    const light = declarations(':root {')
    const dark = declarations('.dark {')
    const themeOnly = new Set(['--radius-panel'])
    for (const name of dark) {
      expect(light.has(name) || themeOnly.has(name), `${name} has no light-theme value`).toBe(true)
    }
  })
})

describe('modal motion (audit 16)', () => {
  /* The station page's climate dialog is the app's first centred modal, so its
     enter animation is pinned here: opacity + a 0.96 scale inside the motion
     ladder, never `transition: all`, never `scale(0)`, and reduced motion keeps
     the fade while dropping the movement. */
  const css = readFileSync(CSS_PATH, 'utf8')
  const keyframes = (name: string) =>
    css.match(new RegExp(`@keyframes ${name} \\{([\\s\\S]*?)\\n\\}`))?.[1] ?? ''

  it('uses the token duration and a real scale, never zero', () => {
    expect(css).toContain('animation: dialog-in var(--motion-base) var(--ease-ui) both')
    const base = Number(css.match(/--motion-base: (\d+)ms/)?.[1])
    expect(base).toBeGreaterThanOrEqual(120)
    expect(base).toBeLessThanOrEqual(240)

    const dialogIn = keyframes('dialog-in')
    expect(dialogIn).toContain('opacity: 0')
    expect(dialogIn).toContain('scale(0.96)')
    expect(dialogIn).not.toMatch(/scale\(\s*0\s*[,)]/)
    expect(dialogIn).not.toContain('translate')
  })

  it('drops the scale but keeps the fade under reduced motion', () => {
    const fade = keyframes('dialog-fade-in')
    expect(fade).toContain('opacity')
    expect(fade).not.toContain('transform')
    /* The global 0.001 ms gate would collapse the fade together with the scale;
       the override restores the opacity-only animation. */
    expect(css).toMatch(/\.dialog-enter \{\s*animation-name: dialog-fade-in !important;/)
  })
})

describe('one precision per metric', () => {
  /* Finding 17: the row and its expander printed skill as `0,53` and `0,532`,
     and RMSE as 3 dp here and 2 dp in Benchmark. The guard scanned only
     `components/ui` and `components/charts`, so it could not see it. */
  const PANELS = [
    'src/features/system/components/ModelsPanel.tsx',
    'src/features/system/components/BenchmarkPanel.tsx',
    /* The bench tab renders this one through BenchmarkPanel; leaving it out is
       the same blind spot that hid finding 17 (it printed MAE at 2 dp by hand). */
    'src/features/benchmark/MetricsCompare.tsx',
  ]

  it('takes every displayed precision from PRECISION in lib/format', () => {
    for (const relative of PANELS) {
      const code = readFileSync(path.join(ROOT, relative), 'utf8')
      expect(code, `${relative}: must import PRECISION`).toContain('PRECISION')
      /* A digit after the comma is a local precision authority; the same metric
         would then print differently one click (or one tab) away. */
      /* `,?` before the paren: prettier puts a trailing comma in every
         multi-line call, which is exactly how these panels are written —
         without it the guard missed `formatNumber(\n  value,\n  3,\n)`. */
      /* `,?` before the paren: prettier puts a trailing comma in every
         multi-line call, which is exactly how these panels are written —
         without it the guard missed `formatNumber(\n  value,\n  3,\n)`.
         `[^;()]`: stop at the first paren, otherwise a later call's argument
         (`commit.slice(0, 10)`) is read as this call's precision. */
      const local = [...code.matchAll(/format(?:Number|Percent|Signed)\([^;()]{0,120}?,\s*(\d+)\s*,?\s*\)/g)]
      expect(
        local.map(([, digits]) => digits),
        `${relative}: local precision literal, use PRECISION`,
      ).toEqual([])
    }
  })
})

describe('primitives and charts', () => {
  const files = [
    ...sourceFiles(path.join(ROOT, 'src/components/ui')),
    ...sourceFiles(path.join(ROOT, 'src/components/charts')),
  ]

  it('finds the primitive sources', () => {
    expect(files.length).toBeGreaterThan(8)
  })

  it('contains no banned generated-look utility', () => {
    for (const file of files) {
      const code = readFileSync(file, 'utf8')
      for (const { pattern, why } of BANNED) {
        expect(pattern.test(code), `${path.relative(ROOT, file)}: ${why}`).toBe(false)
      }
    }
  })

  it('formats numbers through lib/format instead of ad-hoc literals', () => {
    for (const file of files) {
      const code = readFileSync(file, 'utf8')
      /* `toFixed`/`Intl` in a chart or primitive means a second precision. */
      expect(/\.toFixed\(/.test(code), `${path.relative(ROOT, file)}: .toFixed()`).toBe(false)
      expect(/new Intl\./.test(code), `${path.relative(ROOT, file)}: inline Intl`).toBe(false)
    }
  })
})
