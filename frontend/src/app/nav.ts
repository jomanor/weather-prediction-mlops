export interface NavItem {
  to: string
  label: string
}

export const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Resumen' },
  { to: '/stations', label: 'Estación' },
  { to: '/benchmark', label: 'Benchmark' },
  { to: '/models', label: 'Modelos' },
]
