import type { LucideIcon } from 'lucide-react'
import type { ComponentType, LazyExoticComponent } from 'react'

/**
 * Navigation + feature-module contract.
 *
 * A feature is a self-describing folder: its `index.ts` exports a
 * `FeatureManifest`, and `app/registry.ts` is the only place that lists it.
 */

export interface NavItem {
  /** Absolute route path, e.g. `/` or `/stations`. */
  to: string
  label: string
  /** Optional lucide icon, rendered by the compact (mobile) navigation. */
  icon?: LucideIcon
  /** Lower sorts first; ties break on `to`. */
  order?: number
}

export interface FeatureRoute {
  /** Path relative to the app shell; `''` is the index route. */
  path: string
  component: LazyExoticComponent<ComponentType>
}

export interface FeatureManifest {
  /** Unique across the registry. */
  id: string
  navItem?: NavItem
  routes: FeatureRoute[]
}

export function orderNavItems(items: NavItem[]): NavItem[] {
  return [...items].sort((a, b) => (a.order ?? 100) - (b.order ?? 100) || a.to.localeCompare(b.to))
}
