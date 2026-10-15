import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { CityManager } from '@/features/overview/CityManager'

const mocks = vi.hoisted(() => ({
  removeMutate: vi.fn(),
  cities: [
    { name: 'Sevilla', latitude: 37.38, longitude: -5.98 },
    { name: 'Valencia', latitude: 39.47, longitude: -0.37 },
  ],
}))

vi.mock('@/api/queries', () => ({
  useCities: () => ({ data: mocks.cities, isLoading: false }),
}))

vi.mock('@/features/overview/queries', () => ({
  useAddCity: () => ({ mutate: vi.fn(), isPending: false }),
  useGeoSearch: () => ({ data: { results: [] }, isLoading: false, isError: false }),
  useRemoveCity: () => ({ mutate: mocks.removeMutate, isPending: false }),
}))

const confirmFor = (city: string) => screen.queryByRole('group', { name: `Confirmar quitar ${city}` })

describe('CityManager remove confirmation', () => {
  beforeEach(() => mocks.removeMutate.mockReset())

  it('does not issue a delete request until the confirmation is confirmed', async () => {
    render(<CityManager />)

    await userEvent.click(screen.getByRole('button', { name: 'Quitar Sevilla' }))

    expect(mocks.removeMutate).not.toHaveBeenCalled()
    expect(confirmFor('Sevilla')).toBeInTheDocument()
    expect(screen.getByText(/Ingesta y datos dejarán de actualizarse para/)).toBeInTheDocument()
  })

  it('issues nothing when cancelled', async () => {
    render(<CityManager />)
    await userEvent.click(screen.getByRole('button', { name: 'Quitar Sevilla' }))

    await userEvent.click(screen.getByRole('button', { name: 'Cancelar' }))

    expect(mocks.removeMutate).not.toHaveBeenCalled()
    expect(confirmFor('Sevilla')).not.toBeInTheDocument()
  })

  it('calls the existing mutation with the city name when confirmed', async () => {
    render(<CityManager />)
    await userEvent.click(screen.getByRole('button', { name: 'Quitar Sevilla' }))

    await userEvent.click(screen.getByRole('button', { name: 'Quitar estación' }))

    expect(mocks.removeMutate).toHaveBeenCalledTimes(1)
    expect(mocks.removeMutate.mock.calls[0][0]).toBe('Sevilla')
  })

  it('defaults focus to Cancel and cancels on Escape', async () => {
    render(<CityManager />)
    await userEvent.click(screen.getByRole('button', { name: 'Quitar Sevilla' }))

    expect(screen.getByRole('button', { name: 'Cancelar' })).toHaveFocus()

    await userEvent.keyboard('{Escape}')

    expect(mocks.removeMutate).not.toHaveBeenCalled()
    expect(confirmFor('Sevilla')).not.toBeInTheDocument()
  })

  it('scopes the confirmation to the chosen city', async () => {
    render(<CityManager />)

    await userEvent.click(screen.getByRole('button', { name: 'Quitar Valencia' }))

    expect(confirmFor('Valencia')).toBeInTheDocument()
    expect(confirmFor('Sevilla')).not.toBeInTheDocument()
  })
})
