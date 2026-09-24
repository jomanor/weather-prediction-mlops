import { Boxes, Gauge, Scale, RadioTower, type LucideIcon } from 'lucide-react'

export interface NavItem {
  to: string
  label: string
  description: string
  icon: LucideIcon
}

export const NAV_ITEMS: NavItem[] = [
  {
    to: '/',
    label: 'Resumen',
    description: 'Red nacional en vivo',
    icon: Gauge,
  },
  {
    to: '/stations',
    label: 'Estación',
    description: 'Condiciones y evolución',
    icon: RadioTower,
  },
  {
    to: '/benchmark',
    label: 'Benchmark',
    description: 'Modelo vs AEMET vs observado',
    icon: Scale,
  },
  {
    to: '/models',
    label: 'Modelos',
    description: 'Registro de experimentos',
    icon: Boxes,
  },
]
