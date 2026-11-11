/**
 * The user-composable dashboard's domain: which widget types exist, which
 * options their controls offer, and the shape one stored widget takes.
 *
 * React-free on purpose — `storage.ts` parses with these values and is unit
 * tested without a renderer.
 */

/** Every tile type the panel can hold. `widgets.tsx` maps each one to a body. */
export const WIDGET_TYPES = ['diurnal', 'wind-rose', 'series'] as const
export type WidgetType = (typeof WIDGET_TYPES)[number]

/** Climatology windows — the same three `ClimateViewsDialog` offers. */
export const DAY_OPTIONS = [30, 90, 180] as const
export type WidgetDays = (typeof DAY_OPTIONS)[number]

/** Observation windows — the four the station page offers. */
export const HOUR_OPTIONS = [24, 48, 72, 168] as const
export type WidgetHours = (typeof HOUR_OPTIONS)[number]

/**
 * Twelve cards is a dashboard; more is a scrolling list. The cap is enforced on
 * read (`storage.ts`) *and* on add (the page), so neither side can grow it.
 */
export const MAX_WIDGETS = 12

/** Per-widget settings. Absent means "use the widget's `defaultConfig`". */
export interface WidgetConfig {
  city?: string
  days?: WidgetDays
  hours?: WidgetHours
}

/** One card of the layout, exactly as persisted. */
export interface DashboardWidget extends WidgetConfig {
  id: string
  type: WidgetType
}

/** The controls a card renders, in the order it renders them. */
export type WidgetControl = 'city' | 'days' | 'hours'
