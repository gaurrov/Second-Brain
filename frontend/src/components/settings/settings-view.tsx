"use client"

import * as React from "react"
import { Check, Monitor, Moon, Sun, UserRound } from "lucide-react"
import { useTheme } from "next-themes"

import { apiClient } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useAuthStore } from "@/stores/auth-store"
import type { User } from "@/types"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

function initials(name: string): string {
  const parts = name.trim().split(/[\s._-]+/).filter(Boolean)
  if (parts.length === 0) return "U"
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
}

const THEME_OPTIONS = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
] as const

const emptySubscribe = () => () => {}

function ProfileField({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2.5 first:pt-0 last:pb-0">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="min-w-0 truncate text-sm font-medium">{value}</dd>
    </div>
  )
}

function AppearanceCard() {
  const { theme, setTheme } = useTheme()
  // False during SSR/first paint, true once hydrated client-side — lets the
  // control render without knowing the persisted theme until it is safe.
  const mounted = React.useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  )

  return (
    <section id="appearance" aria-labelledby="appearance-heading">
      <Card>
        <CardHeader>
          <CardTitle id="appearance-heading">Appearance</CardTitle>
          <CardDescription>
            Choose how Memora looks on this device. The selection is saved
            automatically.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            role="radiogroup"
            aria-label="Theme"
            className="grid gap-2 sm:grid-cols-3"
          >
            {THEME_OPTIONS.map((option) => {
              const Icon = option.icon
              const selected = mounted && theme === option.value
              return (
                <button
                  key={option.value}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  onClick={() => setTheme(option.value)}
                  className={cn(
                    "inline-flex h-9 items-center justify-start gap-2 rounded-lg border border-border bg-background px-3 text-sm font-medium transition-colors",
                    "hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    selected &&
                      "border-primary/60 bg-primary/5 text-foreground ring-1 ring-primary/40 hover:bg-primary/10",
                  )}
                >
                  <Icon className="size-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{option.label}</span>
                  {selected ? (
                    <Check className="ml-auto size-4 shrink-0 text-primary" />
                  ) : null}
                </button>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </section>
  )
}

export function SettingsView() {
  const storeUser = useAuthStore((state) => state.user)
  const [profileUser, setProfileUser] = React.useState<User | null>(null)
  const user = profileUser ?? storeUser

  // Refresh the profile from the backend on mount; the cached user is
  // shown meanwhile so the page paints instantly.
  React.useEffect(() => {
    let cancelled = false
    apiClient
      .get<User>("/users/profile")
      .then((response) => {
        if (cancelled) return
        setProfileUser(response.data)
        // Keep the cached identity in sync with the server.
        useAuthStore.getState().setUser(response.data)
      })
      .catch(() => {
        // Read-only profile: the cached user remains a fine fallback.
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Deep links from the user menu land on #profile.
  React.useEffect(() => {
    if (typeof window !== "undefined" && window.location.hash === "#profile") {
      document.getElementById("profile")?.scrollIntoView({ block: "start" })
    }
  }, [])

  const displayName = user?.username ?? "Not signed in"
  const email = user?.email ?? "—"
  const memberSince = user?.created_at
    ? new Date(user.created_at).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : "—"

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-2xl px-4 py-8 sm:px-6">
        <header className="mb-6">
          <h1 className="font-heading text-2xl font-semibold tracking-tight">
            Settings
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Manage your account and how Memora looks and feels.
          </p>
        </header>

        <div className="space-y-6">
          <section id="profile" aria-labelledby="profile-heading" className="scroll-mt-6">
            <Card>
              <CardHeader>
                <CardTitle id="profile-heading">Profile</CardTitle>
                <CardDescription>
                  Your account details. Profile editing is not available yet.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4 pb-4">
                  <span className="flex size-12 items-center justify-center rounded-full bg-gradient-to-br from-primary/80 to-primary/50 text-base font-semibold text-primary-foreground">
                    {initials(displayName === "Not signed in" ? "" : displayName)}
                  </span>
                  <div className="min-w-0">
                    <p className="truncate text-base font-medium">{displayName}</p>
                    <p className="truncate text-sm text-muted-foreground">{email}</p>
                  </div>
                </div>
                <dl className="divide-y divide-border border-t border-border pt-1">
                  <ProfileField label="Username" value={user?.username ?? "—"} />
                  <ProfileField label="Email" value={user?.email ?? "—"} />
                  <ProfileField label="Role" value={user?.role ?? "—"} />
                  <ProfileField label="Member since" value={<span suppressHydrationWarning>{memberSince}</span>} />
                </dl>
              </CardContent>
            </Card>
          </section>

          <AppearanceCard />

          <p className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
            <UserRound className="size-3.5 shrink-0" aria-hidden="true" />
            Signed in as {email}
          </p>
        </div>
      </div>
    </div>
  )
}
