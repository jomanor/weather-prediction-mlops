import { usePreferences } from '@/app/preferences'
import { Segmented } from '@/components/ui/Segmented'
import type { UnitSystem } from '@/lib/format'

const OPTIONS = [
  { value: 'metric' as UnitSystem, label: '°C', title: 'Sistema métrico' },
  { value: 'imperial' as UnitSystem, label: '°F', title: 'Sistema imperial' },
]

export function UnitToggle() {
  const { units, setUnits } = usePreferences()
  return <Segmented value={units} onChange={setUnits} options={OPTIONS} label="Unidades" />
}
