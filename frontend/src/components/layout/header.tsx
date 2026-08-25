"use client"

import * as React from "react"
import { Menu, MessageSquarePlus } from "lucide-react"
import { Tooltip } from "@base-ui/react/tooltip"

import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/theme-toggle"

export function Header({ onMenuClick }: { onMenuClick: () => void }) {
  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border/70 bg-background/80 px-3 backdrop-blur-md sm:px-5">
      <Button
        variant="ghost"
        size="icon"
        className="lg:hidden"
        onClick={onMenuClick}
        aria-label="Open menu"
      >
        <Menu />
      </Button>

      <div className="min-w-0 flex-1">
        <h1 className="truncate text-sm font-semibold tracking-tight">New chat</h1>
        <p className="truncate text-xs text-muted-foreground">
          Chat with your knowledge base
        </p>
      </div>

      <span className="hidden items-center gap-1.5 rounded-full border border-amber-500/25 bg-amber-500/10 px-2.5 py-1 text-[11px] font-medium text-amber-700 dark:text-amber-400 sm:inline-flex">
        <span className="size-1.5 rounded-full bg-amber-500" />
        Preview — backend not connected
      </span>

      <Tooltip.Root>
        <Tooltip.Trigger
          render={
            <Button variant="ghost" size="icon" aria-label="Start new chat">
              <MessageSquarePlus />
            </Button>
          }
        />
        <Tooltip.Portal>
          <Tooltip.Positioner sideOffset={6}>
            <Tooltip.Popup className="rounded-md bg-primary px-2 py-1 text-xs text-primary-foreground shadow-md data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95">
              New chat
            </Tooltip.Popup>
          </Tooltip.Positioner>
        </Tooltip.Portal>
      </Tooltip.Root>

      <ThemeToggle />
    </header>
  )
}
