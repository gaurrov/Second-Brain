"use client"

import * as React from "react"
import { Brain, Settings, X } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"

import { Button } from "@/components/ui/button"
import { DocumentSidebar } from "@/components/layout/document-sidebar"
import { ConversationsSidebar } from "@/components/layout/conversations-sidebar"
import { UserMenu } from "@/components/layout/user-menu"
import { cn } from "@/lib/utils"

export function Sidebar({
  open,
  onClose,
  className,
}: {
  /** Drawer state below `lg`. Ignored on desktop where the sidebar is static. */
  open: boolean
  onClose: () => void
  className?: string
}) {
  const router = useRouter()

  return (
    <aside
      aria-label="Knowledge base"
      aria-hidden={!open ? undefined : false}
      className={cn(
        // Mobile / tablet: off-canvas drawer. Desktop: static column.
        "fixed inset-y-0 left-0 z-50 flex w-[300px] max-w-[85vw] shrink-0 flex-col",
        "border-r border-sidebar-border bg-sidebar text-sidebar-foreground",
        "-translate-x-full transition-transform duration-300 ease-in-out",
        "lg:static lg:w-[300px] lg:max-w-none lg:translate-x-0",
        open && "translate-x-0",
        className
      )}
    >
      {/* Brand */}
      <div className="flex h-14 items-center justify-between gap-2 px-4">
        <Link href="/dashboard" className="flex items-center gap-2.5 rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-ring/50">
          <span className="flex size-8 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary/60 text-primary-foreground shadow-sm">
            <Brain className="size-4.5" />
          </span>
          <span className="flex flex-col leading-none">
            <span className="text-[15px] font-semibold tracking-tight">Memora</span>
            <span className="mt-0.5 text-[11px] text-muted-foreground">Your Knowledge</span>
          </span>
        </Link>
        <Button
          variant="ghost"
          size="icon-sm"
          className="lg:hidden"
          onClick={onClose}
          aria-label="Close menu"
        >
          <X />
        </Button>
      </div>

      {/* Body — conversations on top, knowledge library below */}
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-hidden px-3 pb-2">
        <ConversationsSidebar />
        <div
          role="presentation"
          aria-hidden="true"
          className="h-px shrink-0 bg-sidebar-border"
        />
        <DocumentSidebar />
      </div>

      {/* Footer */}
      <div className="space-y-1 border-t border-sidebar-border p-3">
        <Button
          variant="ghost"
          className="w-full justify-start gap-2.5 px-2 text-muted-foreground hover:text-foreground"
          onClick={() => router.push("/settings")}
        >
          <Settings data-icon="inline-start" className="opacity-80" />
          Settings
        </Button>
        <UserMenu />
      </div>
    </aside>
  )
}
