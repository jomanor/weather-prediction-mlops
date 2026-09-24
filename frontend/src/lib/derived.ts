/**
 * Atmospheric derivations.
 *
 * These are deterministic physical formulae applied to *real* observations —
 * not model output and not synthetic data. They are computed client-side so the
 * station view can show the same quantities the Spark feature pipeline derives
 * server-side; each value is labelled as derived in the UI.
 */

/** Saturation vapour pressure over water (hPa), Bolton (1980). */
export function saturationVapourPressure(temperatureC: number): number {
  return 6.112 * Math.exp((17.67 * temperatureC) / (temperatureC + 243.5))
}

/** Specific humidity q (g/kg) from temperature, relative humidity and pressure. */
export function specificHumidity(
  temperatureC: number,
  relativeHumidity: number,
  pressureHpa: number,
): number | null {
  if (pressureHpa <= 0 || relativeHumidity < 0) return null
  const vapourPressure = (relativeHumidity / 100) * saturationVapourPressure(temperatureC)
  const denominator = pressureHpa - 0.378 * vapourPressure
  if (denominator <= 0) return null
  return ((0.622 * vapourPressure) / denominator) * 1000
}

/** Dew point (°C), Magnus-Tetens approximation. */
export function dewPoint(temperatureC: number, relativeHumidity: number): number | null {
  if (relativeHumidity <= 0) return null
  const a = 17.27
  const b = 237.7
  const gamma =
    (a * temperatureC) / (b + temperatureC) + Math.log(Math.min(relativeHumidity, 100) / 100)
  return (b * gamma) / (a - gamma)
}

/** Cartesian wind components: u = zonal (east+), v = meridional (north+), in m/s. */
export function windComponents(
  speedKmh: number,
  directionDeg: number,
): { u: number; v: number } {
  const speed = speedKmh / 3.6
  const radians = (directionDeg * Math.PI) / 180
  // Meteorological direction is where the wind blows *from*.
  return {
    u: -speed * Math.sin(radians),
    v: -speed * Math.cos(radians),
  }
}
