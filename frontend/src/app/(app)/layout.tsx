import type { Metadata } from "next"

import { AppShell } from "@/components/layout/app-shell"

export const metadata: Metadata = {
  title: "Memora",
  description: "Your intelligent second brain.",
}

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return <AppShell>{children}</AppShell>
}
