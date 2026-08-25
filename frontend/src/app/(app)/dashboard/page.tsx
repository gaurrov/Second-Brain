import type { Metadata } from "next"

import { ChatArea } from "@/components/layout/chat-area"

export const metadata: Metadata = {
  title: "Memora",
  description: "Chat with your knowledge base.",
}

export default function DashboardPage() {
  return <ChatArea />
}
