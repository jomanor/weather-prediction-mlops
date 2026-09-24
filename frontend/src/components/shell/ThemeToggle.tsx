import { Monitor, Moon, Sun } from 'lucide-react'

import { usePreferences } from '@/app/preferences'
import { Segmented } from '@/components/ui/Segmented'
import type { ThemeChoice } from '@/lib/chart-theme'

const OPTIONS = [
  { value: 'light' as ThemeChoice, label: <Sun className="h-3.5 w-3.5" />, title: 'Tema claro' },
  { value: 'dark' as ThemeChoice, label: <Moon className="h-3.5 w-3.5" />, title: 'Tema oscuro' },
  { value: 'system' as ThemeChoice, label: <Monitor className="h-3.5 w-3.5" />, title: 'Sistema' },
]

export function ThemeToggle() {
  const { theme, setTheme } = usePreferences()
  return <Segmented value={theme} onChange={setTheme} options={OPTIONS} label="Tema" />
}
