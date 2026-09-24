import maplibregl from 'maplibre-gl'
import { useEffect, useMemo, useRef } from 'react'

import type { CurrentWeather } from '@/api/schemas'
import { usePreferences } from '@/app/preferences'
import { cn } from '@/lib/cn'
import { formatTemperature, isNum } from '@/lib/format'

import 'maplibre-gl/dist/maplibre-gl.css'

const STYLES = {
  dark: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  light: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
} as const

const IBERIA_CENTER: [number, number] = [-3.9, 39.9]

interface StationMapProps {
  stations: CurrentWeather[]
  selectedCity: string | null
  onSelect: (city: string) => void
  className?: string
}

export function StationMap({ stations, selectedCity, onSelect, className }: StationMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const markersRef = useRef(new globalThis.Map<string, maplibregl.Marker>())
  const { isDark, units } = usePreferences()

  const geoStations = useMemo(
    () => stations.filter((station) => isNum(station.latitude) && isNum(station.longitude)),
    [stations],
  )

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: isDark ? STYLES.dark : STYLES.light,
      center: IBERIA_CENTER,
      zoom: 5.1,
      attributionControl: { compact: true },
    })

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    mapRef.current = map

    // The container can settle after mount (lazy load, fonts, layout shifts),
    // which leaves the canvas short of its box. Track it explicitly.
    const observer = new ResizeObserver(() => map.resize())
    observer.observe(containerRef.current)
    map.once('load', () => map.resize())

    return () => {
      observer.disconnect()
      markersRef.current.forEach((marker) => marker.remove())
      markersRef.current.clear()
      map.remove()
      mapRef.current = null
    }
    // The map is created once; theme swaps are handled below via setStyle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    mapRef.current?.setStyle(isDark ? STYLES.dark : STYLES.light)
  }, [isDark])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const seen = new Set<string>()
    for (const station of geoStations) {
      const city = station.city
      seen.add(city)
      const selected = city === selectedCity
      const label = isNum(station.temperature)
        ? formatTemperature(station.temperature, units, 0)
        : '—'

      const existing = markersRef.current.get(city)
      const element = existing?.getElement()
      if (element) {
        element.dataset.selected = String(selected)
        element.innerHTML = renderPin(city, label)
      } else {
        const node = document.createElement('button')
        node.type = 'button'
        node.className = 'station-pin'
        node.dataset.selected = String(selected)
        node.innerHTML = renderPin(city, label)
        node.addEventListener('click', (event) => {
          event.stopPropagation()
          onSelect(city)
        })
        const marker = new maplibregl.Marker({ element: node, anchor: 'bottom' })
          .setLngLat([station.longitude as number, station.latitude as number])
          .addTo(map)
        markersRef.current.set(city, marker)
      }
    }

    for (const [city, marker] of markersRef.current) {
      if (!seen.has(city)) {
        marker.remove()
        markersRef.current.delete(city)
      }
    }
  }, [geoStations, selectedCity, units, onSelect])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !selectedCity) return
    const station = geoStations.find((candidate) => candidate.city === selectedCity)
    if (!station) return
    map.easeTo({
      center: [station.longitude as number, station.latitude as number],
      zoom: Math.max(map.getZoom(), 7),
      duration: 700,
    })
  }, [selectedCity, geoStations])

  return (
    <div
      ref={containerRef}
      className={cn('h-full w-full bg-panel-2', className)}
      role="application"
      aria-label="Mapa de estaciones meteorológicas"
    />
  )
}

function renderPin(city: string, temperature: string) {
  return `
    <span class="station-pin__dot" aria-hidden="true"></span>
    <span class="station-pin__body">
      <span class="station-pin__city">${escapeHtml(city)}</span>
      <span class="station-pin__temp">${escapeHtml(temperature)}</span>
    </span>
  `
}

function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, (character) => {
    switch (character) {
      case '&':
        return '&amp;'
      case '<':
        return '&lt;'
      case '>':
        return '&gt;'
      case '"':
        return '&quot;'
      default:
        return '&#39;'
    }
  })
}
