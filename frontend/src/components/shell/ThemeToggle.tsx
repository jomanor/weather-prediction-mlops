import { Monitor, Moon, Sun } from 'lucide-react'

import { usePreferences } from '@/app/preferences'
import { Segmented } from '@/components/ui/Segmented'
import type { ThemeChoice } from '@/lib/chart-theme'

/* Icon-only options, so `title` is the accessible name: the third one must not
 * read "Sistema", which is also the name of a navigation section. */
const OPTIONS = [
  { value: 'light' as ThemeChoice, label: <Sun className="h-3.5 w-3.5" />, title: 'Tema claro' },
  { value: 'dark' as ThemeChoice, label: <Moon className="h-3.5 w-3.5" />, title: 'Tema oscuro' },
  {
    value: 'system' as ThemeChoice,
    label: <Monitor className="h-3.5 w-3.5" />,
    title: 'Tema del sistema',
  },
]

export function ThemeToggle() {
  const { theme, setTheme } = usePreferences()
  return <Segmented value={theme} onChange={setTheme} options={OPTIONS} label="Tema" />
}
