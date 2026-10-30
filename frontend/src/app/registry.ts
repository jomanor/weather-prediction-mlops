import { orderNavItems, type FeatureManifest, type NavItem } from '@/app/nav'
import { overviewFeature } from '@/features/overview'
import { stationFeature } from '@/features/station'
import { systemFeature } from '@/features/system'

/**
 * The feature registry. Adding a feature means adding its folder and one entry
 * here — the route table and the navigation both derive from this list.
 *
 * Explicit imports (not `import.meta.glob`) keep the graph statically analysable
 * so each feature still code-splits into its own lazy chunk.
 *
 * Three surfaces only: Resumen, Estación, Sistema. Quality, models, benchmark
 * and analytics live inside Sistema, behind `?tab=`.
 */
export const FEATURES: FeatureManifest[] = [overviewFeature, stationFeature, systemFeature]

/** Absolute path of a feature route inside the shell. */
export function routePath(path: string): string {
  return path === '' ? '/' : `/${path}`
}

/** Nav items in render order. */
export const NAV_ITEMS: NavItem[] = orderNavItems(
  FEATURES.flatMap((feature) => (feature.navItem ? [feature.navItem] : [])),
)

/** Every registered route with the id of the feature that owns it. */
export const ROUTES = FEATURES.flatMap((feature) =>
  feature.routes.map((route) => ({ id: feature.id, path: route.path, to: routePath(route.path) })),
)

/** Permanent redirects from surfaces this registry absorbed. */
export const REDIRECTS = FEATURES.flatMap((feature) =>
  (feature.redirects ?? []).map((redirect) => ({
    id: feature.id,
    from: routePath(redirect.path),
    to: redirect.to,
  })),
)
