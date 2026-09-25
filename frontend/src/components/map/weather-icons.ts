import { ArrowUp, type LucideIcon } from 'lucide-react'
import type maplibregl from 'maplibre-gl'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { WEATHER_ICONS, type WeatherIconName } from '@/lib/weather'

/**
 * Rasterises the lucide glyphs into MapLibre SDF images so a single image can
 * be tinted by the temperature/wind ramps (`icon-color` only applies to SDF).
 * No second icon set: the paths come from the same lucide components the rest
 * of the UI uses.
 */

export const WEATHER_IMAGE_PREFIX = 'meteoml-wc-'
export const WIND_ARROW_IMAGE = 'meteoml-wind-arrow'

const ICON_PX = 64
const PIXEL_RATIO = 2

function renderIconDataUrl(icon: LucideIcon): string {
  const markup = renderToStaticMarkup(
    createElement(icon, { color: '#ffffff', strokeWidth: 2, absoluteStrokeWidth: false }),
  )
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`
}

function rasterise(source: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = ICON_PX
      canvas.height = ICON_PX
      const context = canvas.getContext('2d')
      if (!context) {
        reject(new Error('2d context unavailable'))
        return
      }
      context.clearRect(0, 0, ICON_PX, ICON_PX)
      context.drawImage(image, 0, 0, ICON_PX, ICON_PX)
      resolve(context.getImageData(0, 0, ICON_PX, ICON_PX))
    }
    image.onerror = () => reject(new Error('icon failed to load'))
    image.src = source
  })
}

async function addSdfImage(map: maplibregl.Map, id: string, icon: LucideIcon): Promise<void> {
  const data = await rasterise(renderIconDataUrl(icon))
  if (!map.hasImage(id)) map.addImage(id, data, { sdf: true, pixelRatio: PIXEL_RATIO })
}

/**
 * Adds every weather glyph plus the wind arrow. A single failing image is
 * swallowed so the basemap and the remaining layers still render.
 */
export async function ensureMapIcons(map: maplibregl.Map): Promise<void> {
  const jobs: Array<Promise<void>> = []
  for (const name of Object.keys(WEATHER_ICONS) as WeatherIconName[]) {
    const id = `${WEATHER_IMAGE_PREFIX}${name}`
    if (!map.hasImage(id)) jobs.push(addSdfImage(map, id, WEATHER_ICONS[name]).catch(() => {}))
  }
  if (!map.hasImage(WIND_ARROW_IMAGE)) {
    jobs.push(addSdfImage(map, WIND_ARROW_IMAGE, ArrowUp).catch(() => {}))
  }
  await Promise.all(jobs)
}
