import {
  Cloud,
  CloudDrizzle,
  CloudFog,
  CloudLightning,
  CloudMoon,
  CloudRain,
  CloudSnow,
  CloudSun,
  Moon,
  MoonStar,
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

/*
 * Night variants for the four clear/cloud codes. Only these four change with the
 * sun; rain, snow, fog and storms keep their glyph and label at night. Tones are
 * deliberately identical to the daytime entry so ramps/colours do not shift.
 */
const NIGHT_DESCRIPTORS: Record<number, WeatherDescriptor> = {
  0: { label: 'Cielo despejado (noche)', icon: Moon, tone: 'clear' },
  1: { label: 'Mayormente despejado (noche)', icon: MoonStar, tone: 'clear' },
  2: { label: 'Parcialmente nuboso (noche)', icon: CloudMoon, tone: 'cloud' },
  3: { label: 'Cubierto (noche)', icon: CloudMoon, tone: 'cloud' },
}

/**
 * `is_day` is the API's per-station flag: `1` daytime, `0` night, `null`/absent
 * unknown (payloads served before the backend deploy omit the field).
 *
 * The local-hour fallback exists only for the unknown case so the UI keeps
 * looking right until `is_day` ships. It is a viewer-clock guess — never use it
 * for a station whose flag is known — and it becomes dead code once every
 * payload carries `is_day`. Day is 07:00–21:59 local, matching the handoff.
 */
function isNight(isDay: number | null | undefined): boolean {
  if (isDay === 0) return true
  if (isDay === 1) return false
  const hour = new Date().getHours()
  return hour < 7 || hour > 21
}

export function describeWeather(
  code: number | null | undefined,
  isDay: number | null = null,
): WeatherDescriptor {
  if (typeof code !== 'number') return UNKNOWN
  if (isNight(isDay)) {
    const night = NIGHT_DESCRIPTORS[code]
    if (night) return night
  }
  return DESCRIPTORS[code] ?? UNKNOWN
}

/**
 * Stable image key for a weather-code glyph. The map rasterises one SDF image
 * per key, so the same glyph is reused across intensity variants.
 */
export type WeatherIconName =
  | 'sun'
  | 'cloud-sun'
  | 'cloud'
  | 'fog'
  | 'drizzle'
  | 'rain'
  | 'snow'
  | 'storm'
  | 'moon'
  | 'moon-star'
  | 'cloud-moon'

export const WEATHER_ICONS: Record<WeatherIconName, LucideIcon> = {
  sun: Sun,
  'cloud-sun': CloudSun,
  cloud: Cloud,
  fog: CloudFog,
  drizzle: CloudDrizzle,
  rain: CloudRain,
  snow: CloudSnow,
  storm: CloudLightning,
  moon: Moon,
  'moon-star': MoonStar,
  'cloud-moon': CloudMoon,
}

const ICON_NAMES = new Map<LucideIcon, WeatherIconName>([
  [Sun, 'sun'],
  [CloudSun, 'cloud-sun'],
  [Cloud, 'cloud'],
  [CloudFog, 'fog'],
  [CloudDrizzle, 'drizzle'],
  [CloudRain, 'rain'],
  [CloudSnow, 'snow'],
  [CloudLightning, 'storm'],
  [Moon, 'moon'],
  [MoonStar, 'moon-star'],
  [CloudMoon, 'cloud-moon'],
])

export function weatherIconName(
  code: number | null | undefined,
  isDay: number | null = null,
): WeatherIconName {
  return ICON_NAMES.get(describeWeather(code, isDay).icon) ?? 'cloud'
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
