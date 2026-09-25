import { describe, expect, it } from 'vitest'

import { skillTone } from '@/features/models/ModelsPage'

describe('skillTone thresholds', () => {
  it('treats ≥ 0.3 as a solid win', () => {
    expect(skillTone(0.3)).toBe('ok')
    expect(skillTone(0.9)).toBe('ok')
  })

  it('treats 0–0.3 as marginal', () => {
    expect(skillTone(0.299)).toBe('warn')
    expect(skillTone(0)).toBe('warn')
  })

  it('treats a negative skill as worse than the baseline', () => {
    expect(skillTone(-0.001)).toBe('bad')
  })

  it('is neutral when there is no value', () => {
    expect(skillTone(null)).toBe('neutral')
    expect(skillTone(undefined)).toBe('neutral')
    expect(skillTone(Number.NaN)).toBe('neutral')
  })
})
