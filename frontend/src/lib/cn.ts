import { clsx, type ClassValue } from 'clsx'
import { extendTailwindMerge } from 'tailwind-merge'

/**
 * The custom type scale uses the same `text-*` prefix as colour utilities.
 * Stock `tailwind-merge` classifies `text-label`/`text-body`/… as colours, so
 * `cn('text-label', 'text-fg-2')` silently dropped the size and every control
 * inherited the 16 px root. Registering the scale as font sizes keeps both.
 */
export const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      'font-size': [{ text: ['badge', 'label', 'body', 'title', 'value', 'lead', 'page', 'hero'] }],
    },
  },
})

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
