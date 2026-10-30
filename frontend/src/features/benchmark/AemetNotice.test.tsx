import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { City } from '@/api/schemas'
import { AemetNotice, isOutsideAemetCoverage } from '@/features/benchmark/AemetNotice'

const PASADENA: City = { name: 'Pasadena', latitude: 34.15, longitude: -118.14 }
const MADRID: City = { name: 'Madrid', latitude: 40.42, longitude: -3.7 }

describe('AemetNotice', () => {
  it('explains that AEMET does not cover US stations instead of showing a blank panel', () => {
    render(<AemetNotice available={false} city={PASADENA} />)
    expect(screen.getByText('AEMET no cubre estaciones de EE. UU.')).toBeInTheDocument()
    expect(screen.getByText(/fuera de la red de AEMET OpenData \(España\)/)).toBeInTheDocument()
    expect(screen.getByRole('status')).toBeInTheDocument()
  })

  it('keeps the generic Spanish message for a Spanish station outage', () => {
    render(<AemetNotice available={false} error="Sin respuesta" city={MADRID} />)
    expect(screen.getByText('Benchmark AEMET no disponible')).toBeInTheDocument()
    expect(screen.getByText(/Sin respuesta/)).toBeInTheDocument()
  })

  it('renders nothing when AEMET is available', () => {
    const { container } = render(<AemetNotice available city={MADRID} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('detects coverage from longitude, not from a hardcoded city name', () => {
    expect(isOutsideAemetCoverage(PASADENA)).toBe(true)
    expect(isOutsideAemetCoverage(MADRID)).toBe(false)
    expect(isOutsideAemetCoverage(undefined)).toBe(false)
  })
})
