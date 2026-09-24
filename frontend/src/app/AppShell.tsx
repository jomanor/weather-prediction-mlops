import { CloudSun } from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'

import { NAV_ITEMS } from '@/app/nav'
import { ApiStatus } from '@/components/shell/ApiStatus'
import { StationSearch } from '@/components/shell/StationSearch'
import { ThemeToggle } from '@/components/shell/ThemeToggle'
import { UnitToggle } from '@/components/shell/UnitToggle'
import { cn } from '@/lib/cn'

export function AppShell() {
  return (
    <div className="flex min-h-screen flex-col bg-bg">
      <header className="sticky top-0 z-40 border-b border-line bg-panel/85 backdrop-blur-md">
        <div className="flex h-14 items-center gap-4 px-4">
          <NavLink to="/" className="flex shrink-0 items-center gap-2.5">
            <span className="flex h-7 w-7 items-center justify-center rounded border border-accent/30 bg-accent-soft text-accent">
              <CloudSun className="h-4 w-4" />
            </span>
            <span className="hidden leading-none sm:block">
              <span className="block text-sm font-semibold tracking-tight text-fg">MeteoML</span>
              <span className="mt-0.5 block text-[10px] uppercase tracking-[0.14em] text-fg-3">
                Red predictiva
              </span>
            </span>
          </NavLink>

          <div className="ml-auto flex items-center gap-3">
            <div className="hidden md:block">
              <StationSearch />
            </div>
            <div className="hidden items-center gap-3 lg:flex">
              <ApiStatus />
            </div>
            <div className="hidden sm:block">
              <UnitToggle />
            </div>
            <ThemeToggle />
          </div>
        </div>
        <div className="border-t border-line px-4 py-2 md:hidden">
          <StationSearch />
        </div>
      </header>

      <div className="flex flex-1 flex-col md:flex-row">
        <nav
          aria-label="Secciones"
          className="hidden shrink-0 border-r border-line bg-panel/40 md:flex md:w-56 md:flex-col md:gap-0.5 md:p-2"
        >
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                cn(
                  'group flex items-start gap-2.5 rounded-md px-2.5 py-2 transition-colors',
                  isActive ? 'bg-panel text-fg' : 'text-fg-2 hover:bg-panel-2 hover:text-fg',
                )
              }
            >
              {({ isActive }) => (
                <>
                  <item.icon
                    className={cn('mt-0.5 h-4 w-4 shrink-0', isActive ? 'text-accent' : 'text-fg-3')}
                  />
                  <span className="min-w-0">
                    <span className="block text-[13px] font-medium leading-tight">{item.label}</span>
                    <span className="mt-0.5 block truncate text-[11px] leading-tight text-fg-3">
                      {item.description}
                    </span>
                  </span>
                </>
              )}
            </NavLink>
          ))}

          <div className="mt-auto px-2.5 pb-1 pt-4 text-[10px] leading-relaxed text-fg-3">
            Datos: Open-Meteo · AEMET OpenData
          </div>
        </nav>

        <main className="min-w-0 flex-1 pb-16 md:pb-0">
          <Outlet />
        </main>
      </div>

      <nav
        aria-label="Secciones"
        className="fixed inset-x-0 bottom-0 z-40 grid grid-cols-4 border-t border-line bg-panel/95 backdrop-blur-md md:hidden"
      >
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              cn(
                'flex flex-col items-center gap-1 py-2 text-[10px]',
                isActive ? 'text-accent' : 'text-fg-3',
              )
            }
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </div>
  )
}
