"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { ChevronUp, LogOut, Settings, UserRound } from "lucide-react"

import { useAuthStore } from "@/stores/auth-store"
import { useDocumentsStore } from "@/stores/documents-store"
import { buttonVariants } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"

function initials(name: string): string {
  const parts = name.trim().split(/[\s._-]+/).filter(Boolean)
  if (parts.length === 0) return "U"
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
}

export function UserMenu({ className }: { className?: string }) {
  const router = useRouter()
  const user = useAuthStore((state) => state.user)
  const logout = useAuthStore((state) => state.logout)

  const displayName = user?.username ?? "Guest"
  const email = user?.email ?? "Not signed in"

  const handleLogout = () => {
    logout()
    useDocumentsStore.getState().reset()
    router.push("/login")
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        className={cn(
          buttonVariants({ variant: "ghost" }),
          "h-auto w-full justify-start gap-2.5 px-2 py-2",
          className
        )}
      >
        <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary/80 to-primary/50 text-xs font-semibold text-primary-foreground">
          {initials(displayName)}
        </span>
        <span className="flex min-w-0 flex-1 flex-col items-start">
          <span className="w-full truncate text-sm font-medium">{displayName}</span>
          <span className="w-full truncate text-xs text-muted-foreground">{email}</span>
        </span>
        <ChevronUp className="size-4 shrink-0 text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="start" className="w-60">
        <div className="flex items-center gap-2.5 px-2 py-1.5">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary/80 to-primary/50 text-xs font-semibold text-primary-foreground">
            {initials(displayName)}
          </span>
          <span className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-sm font-medium">{displayName}</span>
            <span className="truncate text-xs text-muted-foreground">{email}</span>
          </span>
        </div>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => router.push("/settings#profile")}>
          <UserRound />
          Profile
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => router.push("/settings")}>
          <Settings />
          Settings
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onClick={handleLogout}>
          <LogOut />
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
