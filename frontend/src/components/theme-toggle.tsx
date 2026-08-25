"use client"

import { Check, Monitor, Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

const THEME_CHOICES = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
] as const

export function ThemeToggle() {
  const { setTheme, resolvedTheme, theme } = useTheme()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="focus:outline-none">
        <div className="inline-flex size-9 items-center justify-center rounded-full border border-border/60 bg-background/60 text-sm font-medium shadow-xs backdrop-blur-sm transition-colors duration-150 hover:bg-muted/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/35 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0">
          {/* Pure CSS resolution keeps server and client markup identical. */}
          <Sun className="h-[1.15rem] w-[1.15rem] rotate-0 scale-100 text-muted-foreground transition-transform duration-200 dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-[1.15rem] w-[1.15rem] rotate-90 scale-0 text-muted-foreground transition-transform duration-200 dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </div>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-40 rounded-xl">
        {THEME_CHOICES.map(({ value, label, icon: Icon }) => {
          const selected =
            value === "system" ? theme === "system" : resolvedTheme === value
          return (
            <DropdownMenuItem
              key={value}
              onClick={() => setTheme(value)}
              className="cursor-pointer gap-2.5"
            >
              <Icon className="size-4 text-muted-foreground" />
              <span className="flex-1">{label}</span>
              {selected && <Check className="size-3.5 text-primary" />}
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
