import {
  Cloud,
  CloudDrizzle,
  CloudFog,
  CloudLightning,
  CloudRain,
  CloudSnow,
  CloudSun,
  Sun,
  type LucideIcon,
} from 'lucide-react'

export type WeatherTone = 'clear' | 'cloud' | 'rain' | 'snow' | 'storm' | 'fog' | 'unknown'

export interface WeatherDescriptor {
  label: string
  icon: LucideIcon
  tone: WeatherTone
}

const UNKNOWN: WeatherDescriptor = { label: 'Sin dato', icon: Cloud, tone: 'unknown' }

const DESCRIPTORS: Record<number, WeatherDescriptor> = {
  0: { label: 'Despejado', icon: Sun, tone: 'clear' },
  1: { label: 'Mayormente despejado', icon: CloudSun, tone: 'clear' },
  2: { label: 'Parcialmente nuboso', icon: CloudSun, tone: 'cloud' },
  3: { label: 'Cubierto', icon: Cloud, tone: 'cloud' },
  45: { label: 'Niebla', icon: CloudFog, tone: 'fog' },
  48: { label: 'Niebla helada', icon: CloudFog, tone: 'fog' },
  51: { label: 'Llovizna débil', icon: CloudDrizzle, tone: 'rain' },
  53: { label: 'Llovizna moderada', icon: CloudDrizzle, tone: 'rain' },
  55: { label: 'Llovizna intensa', icon: CloudDrizzle, tone: 'rain' },
  56: { label: 'Llovizna helada', icon: CloudDrizzle, tone: 'rain' },
  57: { label: 'Llovizna helada intensa', icon: CloudDrizzle, tone: 'rain' },
  61: { label: 'Lluvia débil', icon: CloudRain, tone: 'rain' },
  63: { label: 'Lluvia moderada', icon: CloudRain, tone: 'rain' },
  65: { label: 'Lluvia intensa', icon: CloudRain, tone: 'rain' },
  66: { label: 'Lluvia helada', icon: CloudRain, tone: 'rain' },
  67: { label: 'Lluvia helada intensa', icon: CloudRain, tone: 'rain' },
  71: { label: 'Nieve débil', icon: CloudSnow, tone: 'snow' },
  73: { label: 'Nieve moderada', icon: CloudSnow, tone: 'snow' },
  75: { label: 'Nieve intensa', icon: CloudSnow, tone: 'snow' },
  77: { label: 'Granos de nieve', icon: CloudSnow, tone: 'snow' },
  80: { label: 'Chubascos débiles', icon: CloudRain, tone: 'rain' },
  81: { label: 'Chubascos moderados', icon: CloudRain, tone: 'rain' },
  82: { label: 'Chubascos torrenciales', icon: CloudRain, tone: 'rain' },
  85: { label: 'Chubascos de nieve', icon: CloudSnow, tone: 'snow' },
  86: { label: 'Chubascos de nieve intensos', icon: CloudSnow, tone: 'snow' },
  95: { label: 'Tormenta', icon: CloudLightning, tone: 'storm' },
  96: { label: 'Tormenta con granizo', icon: CloudLightning, tone: 'storm' },
  99: { label: 'Tormenta con granizo fuerte', icon: CloudLightning, tone: 'storm' },
}

export function describeWeather(code: number | null | undefined): WeatherDescriptor {
  if (typeof code !== 'number') return UNKNOWN
  return DESCRIPTORS[code] ?? UNKNOWN
}

export const toneClass: Record<WeatherTone, string> = {
  clear: 'text-aemet',
  cloud: 'text-fg-2',
  rain: 'text-accent',
  snow: 'text-accent',
  storm: 'text-model',
  fog: 'text-fg-3',
  unknown: 'text-fg-3',
}
