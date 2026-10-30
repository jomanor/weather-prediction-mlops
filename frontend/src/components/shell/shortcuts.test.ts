import { describe, expect, it } from 'vitest'

import { NAV_ITEMS } from '@/app/registry'
import {
  SECTION_SHORTCUTS,
  isEditableTarget,
  resolveSectionShortcut,
} from '@/components/shell/shortcuts'

describe('resolveSectionShortcut', () => {
  it('resolves the documented section keys', () => {
    expect(resolveSectionShortcut('r')?.to).toBe('/')
    expect(resolveSectionShortcut('E')?.to).toBe('/stations')
    expect(resolveSectionShortcut('s')?.to).toBe('/sistema')
  })

  it('returns null for an unbound key', () => {
    expect(resolveSectionShortcut('z')).toBeNull()
  })

  it('keeps keys unique', () => {
    expect(new Set(SECTION_SHORTCUTS.map((shortcut) => shortcut.key)).size).toBe(
      SECTION_SHORTCUTS.length,
    )
  })

  it('targets exactly the three registered surfaces', () => {
    expect(SECTION_SHORTCUTS.map((shortcut) => shortcut.to)).toEqual(
      NAV_ITEMS.map((item) => item.to),
    )
  })
})

describe('isEditableTarget', () => {
  it('detects fields and contenteditable', () => {
    expect(isEditableTarget(document.createElement('input'))).toBe(true)
    expect(isEditableTarget(document.createElement('textarea'))).toBe(true)
    expect(isEditableTarget(document.createElement('select'))).toBe(true)
    expect(isEditableTarget(document.createElement('div'))).toBe(false)
    expect(isEditableTarget(null)).toBe(false)
  })
})
