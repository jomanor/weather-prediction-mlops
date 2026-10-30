import { describe, expect, it } from 'vitest'

import {
  beginSwipe,
  isInteractiveTarget,
  resolveSwipe,
  stepStationIndex,
  trackSwipe,
} from '@/features/overview/sheet'

describe('resolveSwipe', () => {
  it('ignores a move shorter than the threshold', () => {
    const start = beginSwipe(100, 100)
    expect(resolveSwipe(start, 100 + 20, 100)).toBeNull()
    expect(resolveSwipe(start, 100 - 59, 100)).toBeNull()
  })

  it('ignores a gesture locked onto the vertical axis', () => {
    const start = beginSwipe(100, 100)
    expect(resolveSwipe(start, 100 - 70, 100 + 80)).toBeNull()
    expect(start.axis).toBe('vertical')
  })

  it('ignores a diagonal gesture that is too close to call', () => {
    const start = beginSwipe(100, 100)
    expect(resolveSwipe(start, 100 - 70, 100 + 60)).toBeNull()
    expect(resolveSwipe(start, 100 + 70, 100 - 60)).toBeNull()
  })

  it('advances on a left swipe and rewinds on a right swipe', () => {
    const left = beginSwipe(200, 100)
    expect(resolveSwipe(left, 200 - 90, 100 + 4)).toBe('next')

    const right = beginSwipe(100, 100)
    expect(resolveSwipe(right, 100 + 90, 100 - 4)).toBe('prev')
  })
})

describe('trackSwipe', () => {
  it('stays undecided inside the slop and then locks once', () => {
    const start = beginSwipe(0, 0)
    expect(trackSwipe(start, 4, 4)).toBeNull()
    expect(trackSwipe(start, 40, 2)).toBe('horizontal')
    // A later vertical move must not re-lock the axis.
    expect(trackSwipe(start, 41, 400)).toBe('horizontal')
  })
})

describe('stepStationIndex', () => {
  it('wraps forwards past the end and backwards before the start', () => {
    expect(stepStationIndex(0, 3, 1)).toBe(1)
    expect(stepStationIndex(2, 3, 1)).toBe(0)
    expect(stepStationIndex(0, 3, -1)).toBe(2)
    expect(stepStationIndex(1, 3, -1)).toBe(0)
  })

  it('returns null when there is nowhere to step', () => {
    expect(stepStationIndex(0, 1, 1)).toBeNull()
    expect(stepStationIndex(0, 0, 1)).toBeNull()
    expect(stepStationIndex(1, 5, 0)).toBeNull()
  })

  it('returns null for an unknown or out-of-range selection', () => {
    expect(stepStationIndex(-1, 5, 1)).toBeNull()
    expect(stepStationIndex(5, 5, 1)).toBeNull()
    expect(stepStationIndex(9, 5, -1)).toBeNull()
  })
})

describe('isInteractiveTarget', () => {
  const matches = (html: string) => {
    const host = document.createElement('div')
    host.innerHTML = html
    const target = host.querySelector('[data-test]') ?? host.firstElementChild
    return isInteractiveTarget(target)
  }

  it('treats controls as owning their own gestures', () => {
    expect(matches('<button data-test type="button">x</button>')).toBe(true)
    expect(matches('<input data-test />')).toBe(true)
    expect(matches('<div data-test role="button" tabindex="0">x</div>')).toBe(true)
    expect(matches('<div data-test contenteditable="true">x</div>')).toBe(true)
  })

  it('looks through descendants of a control', () => {
    expect(isInteractiveTarget(document.querySelector('body'))).toBe(false)
    const button = document.createElement('button')
    const icon = document.createElement('span')
    button.append(icon)
    expect(isInteractiveTarget(icon)).toBe(true)
  })

  it('lets plain elements and non-elements start a swipe', () => {
    expect(matches('<div data-test><p>readout</p></div>')).toBe(false)
    expect(matches('<h2 data-test>x</h2>')).toBe(false)
    expect(isInteractiveTarget(null)).toBe(false)
  })
})
