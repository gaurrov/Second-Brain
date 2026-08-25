"use client"

import { Menu, PenLine } from "lucide-react"
import { usePathname, useRouter } from "next/navigation"
import { Tooltip } from "@base-ui/react/tooltip"

import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/theme-toggle"
import { useChatStore } from "@/stores/chat-store"

/** Contextual page label for the slim top bar. */
function usePageHeading(): { title: string; subtitle: string } {
  const pathname = usePathname()
  if (pathname.startsWith("/settings")) {
    return { title: "Settings", subtitle: "Account & preferences" }
  }
  return { title: "New chat", subtitle: "Chat with your knowledge base" }
}

export function Header({ onMenuClick }: { onMenuClick: () => void }) {
  const router = useRouter()
  const { title, subtitle } = usePageHeading()

  const startNewChat = () => {
    useChatStore.getState().newConversation()
    router.push("/dashboard")
  }

  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border/60 bg-background/80 px-3 backdrop-blur-md sm:px-5">
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
        <h1 className="truncate font-heading text-sm font-semibold tracking-tight">
          {title}
        </h1>
        <p className="truncate text-xs text-muted-foreground/90">{subtitle}</p>
      </div>

      <Tooltip.Root>
        <Tooltip.Trigger
          render={
            <Button
              variant="ghost"
              size="sm"
              onClick={startNewChat}
              className="gap-2 text-muted-foreground hover:text-foreground"
              aria-label="Start new chat"
            >
              <PenLine />
              <span className="hidden sm:inline">New</span>
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
