/**
 * Interaction rules for the mobile station sheet: the horizontal swipe between
 * stations and the wrap-around prev/next stepping. Kept out of the component so
 * both are unit-testable without a pointer, a map or a router.
 */

/** Horizontal travel that turns a press into a swipe. */
export const SWIPE_THRESHOLD_PX = 60

/** Movement below the slop stays undecided, so a tap never scores as a swipe. */
const SWIPE_SLOP_PX = 12

/** Horizontal travel must beat the vertical one by this factor (vertical = page scroll). */
const SWIPE_AXIS_RATIO = 1.5

export type SwipeAxis = 'horizontal' | 'vertical' | null

export interface SwipeStart {
  x: number
  y: number
  /** Locked on the first move past the slop; `null` while the press is undecided. */
  axis: SwipeAxis
}

export function beginSwipe(x: number, y: number): SwipeStart {
  return { x, y, axis: null }
}

/**
 * Locks the gesture axis once the pointer has travelled past the slop and
 * returns it. Mutates `start`, so the lock survives the remaining moves.
 */
export function trackSwipe(start: SwipeStart, x: number, y: number): SwipeAxis {
  if (start.axis === null && Math.hypot(x - start.x, y - start.y) > SWIPE_SLOP_PX) {
    start.axis = Math.abs(x - start.x) > Math.abs(y - start.y) ? 'horizontal' : 'vertical'
  }
  return start.axis
}

/**
 * `next`/`prev` for a deliberate horizontal swipe, otherwise `null`: too short,
 * locked onto the vertical axis, or diagonal enough to be an ambiguous gesture.
 */
export function resolveSwipe(start: SwipeStart, x: number, y: number): 'next' | 'prev' | null {
  const dx = x - start.x
  const dy = y - start.y
  if (Math.abs(dx) < SWIPE_THRESHOLD_PX) return null
  if (trackSwipe(start, x, y) !== 'horizontal') return null
  if (Math.abs(dx) < Math.abs(dy) * SWIPE_AXIS_RATIO) return null
  return dx < 0 ? 'next' : 'prev'
}

/** Controls own their gestures: a swipe starting on one of these is ignored. */
const INTERACTIVE_SELECTOR = [
  'button',
  'a',
  'input',
  'select',
  'textarea',
  'label',
  'summary',
  '[role="button"]',
  '[role="link"]',
  '[role="slider"]',
  '[role="radio"]',
  '[contenteditable="true"]',
].join(',')

export function isInteractiveTarget(target: EventTarget | null): boolean {
  const element = target as Element | null
  if (!element || typeof element.closest !== 'function') return false
  return element.closest(INTERACTIVE_SELECTOR) !== null
}

/**
 * Wrap-around station stepping within the registry. Returns the target index,
 * or `null` when there is nowhere to go (unknown selection, single station or
 * no movement) so the caller can stop predictably instead of jumping.
 */
export function stepStationIndex(index: number, total: number, delta: number): number | null {
  if (total < 2 || delta === 0 || index < 0 || index >= total) return null
  return (index + delta + total) % total
}
