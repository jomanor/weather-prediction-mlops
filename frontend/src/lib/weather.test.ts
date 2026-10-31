import { Cloud, CloudMoon, CloudRain, CloudSun, Moon, MoonStar, Sun } from 'lucide-react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { describeWeather, weatherIconName, WEATHER_ICONS } from '@/lib/weather'

/* Local-time construction (not an ISO string) so the viewer-clock fallback is
 * asserted independently of the machine timezone. */
function atLocalHour(hour: number, minute = 0) {
  vi.setSystemTime(new Date(2026, 9, 31, hour, minute, 0))
}

afterEach(() => {
  vi.useRealTimers()
})

describe('describeWeather day/night matrix', () => {
  const matrix = [
    {
      code: 0,
      day: { label: 'Despejado', icon: Sun, tone: 'clear', glyph: 'sun' },
      night: { label: 'Cielo despejado (noche)', icon: Moon, tone: 'clear', glyph: 'moon' },
    },
    {
      code: 1,
      day: { label: 'Mayormente despejado', icon: CloudSun, tone: 'clear', glyph: 'cloud-sun' },
      night: {
        label: 'Mayormente despejado (noche)',
        icon: MoonStar,
        tone: 'clear',
        glyph: 'moon-star',
      },
    },
    {
      code: 2,
      day: { label: 'Parcialmente nuboso', icon: CloudSun, tone: 'cloud', glyph: 'cloud-sun' },
      night: {
        label: 'Parcialmente nuboso (noche)',
        icon: CloudMoon,
        tone: 'cloud',
        glyph: 'cloud-moon',
      },
    },
    {
      code: 3,
      day: { label: 'Cubierto', icon: Cloud, tone: 'cloud', glyph: 'cloud' },
      night: { label: 'Cubierto (noche)', icon: CloudMoon, tone: 'cloud', glyph: 'cloud-moon' },
    },
  ] as const

  for (const entry of matrix) {
    it(`code ${entry.code} is daytime for is_day 1`, () => {
      const weather = describeWeather(entry.code, 1)
      expect(weather.label).toBe(entry.day.label)
      expect(weather.icon).toBe(entry.day.icon)
      expect(weather.tone).toBe(entry.day.tone)
      expect(weatherIconName(entry.code, 1)).toBe(entry.day.glyph)
    })

    it(`code ${entry.code} is night-time for is_day 0`, () => {
      const weather = describeWeather(entry.code, 0)
      expect(weather.label).toBe(entry.night.label)
      expect(weather.icon).toBe(entry.night.icon)
      /* Tones are unchanged by day/night, so colour ramps do not shift. */
      expect(weather.tone).toBe(entry.night.tone)
      expect(weatherIconName(entry.code, 0)).toBe(entry.night.glyph)
    })

    it(`code ${entry.code} falls back to the local hour for a null is_day`, () => {
      vi.useFakeTimers()
      atLocalHour(12)
      expect(describeWeather(entry.code, null).label).toBe(entry.day.label)
      atLocalHour(23)
      expect(describeWeather(entry.code, null).label).toBe(entry.night.label)
    })
  }

  it('keeps the rain glyph and label at night for a code with no moon variant', () => {
    const weather = describeWeather(61, 0)
    expect(weather.label).toBe('Lluvia débil')
    expect(weather.icon).toBe(CloudRain)
    expect(weather.tone).toBe('rain')
    expect(weatherIconName(61, 0)).toBe('rain')
    /* Identical to the daytime descriptor: only the four clear/cloud codes vary. */
    expect(weather).toEqual(describeWeather(61, 1))
  })

  it('keeps the unknown fallback regardless of the flag', () => {
    for (const isDay of [1, 0, null, undefined]) {
      expect(describeWeather(999, isDay).label).toBe('Sin dato')
      expect(describeWeather(null, isDay).tone).toBe('unknown')
      expect(weatherIconName(undefined, isDay)).toBe('cloud')
    }
  })
})

describe('describeWeather viewer-clock fallback (pre-is_day payloads)', () => {
  it('treats 07:00 through 21:59 local as daytime', () => {
    vi.useFakeTimers()
    atLocalHour(7)
    expect(describeWeather(0, null).label).toBe('Despejado')
    atLocalHour(21, 59)
    expect(describeWeather(0, null).label).toBe('Despejado')
  })

  it('treats before 07:00 and after 21:59 local as night', () => {
    vi.useFakeTimers()
    atLocalHour(6, 59)
    expect(describeWeather(0, null).label).toBe('Cielo despejado (noche)')
    atLocalHour(22)
    expect(describeWeather(0, null).label).toBe('Cielo despejado (noche)')
  })

  it('defaults the flag to null when it is omitted', () => {
    vi.useFakeTimers()
    atLocalHour(12)
    expect(describeWeather(1)).toEqual(describeWeather(1, null))
  })
})

describe('map glyph set', () => {
  it('ships an SDF glyph for every icon a descriptor can produce', () => {
    const names = [0, 1, 2, 3, 45, 61, 71, 95].flatMap((code) => [
      weatherIconName(code, 1),
      weatherIconName(code, 0),
    ])
    for (const name of names) expect(WEATHER_ICONS[name]).toBeDefined()
    expect(Object.keys(WEATHER_ICONS)).toEqual(
      expect.arrayContaining(['moon', 'moon-star', 'cloud-moon']),
    )
  })
})
