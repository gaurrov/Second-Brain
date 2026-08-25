"use client"

import * as React from "react"
import { Brain } from "lucide-react"

import type { ChatMessage } from "@/stores/chat-store"
import { AssistantMessage } from "@/components/chat/assistant-message"
import { UserMessage } from "@/components/chat/user-message"

/**
 * Scrollable transcript. Renders the persisted messages and, while a
 * stream is in flight with nothing produced yet, the thinking indicator.
 */
export function MessageList({
  messages,
  isStreaming,
}: {
  messages: ChatMessage[]
  isStreaming: boolean
}) {
  const viewportRef = React.useRef<HTMLDivElement>(null)
  const pinnedToBottom = React.useRef(true)

  // Track whether the user has scrolled away (so we don't yank them back).
  const handleScroll = () => {
    const viewport = viewportRef.current
    if (!viewport) return
    const distance = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight
    pinnedToBottom.current = distance < 80
  }

  React.useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport || !pinnedToBottom.current) return
    viewport.scrollTo({ top: viewport.scrollHeight })
  }, [messages, isStreaming])

  const waitingForFirstToken =
    isStreaming && messages[messages.length - 1]?.content === ""

  return (
    <div
      ref={viewportRef}
      onScroll={handleScroll}
      className="min-h-0 flex-1 overflow-y-auto"
    >
      <div className="mx-auto flex max-w-3xl flex-col gap-7 px-4 py-8 sm:px-6">
        {messages.map((message, index) =>
          message.role === "user" ? (
            <UserMessage key={message.id} message={message} />
          ) : (
            <AssistantMessage
              key={message.id}
              message={message}
              isStreaming={isStreaming && index === messages.length - 1}
            />
          ),
        )}
        {waitingForFirstToken && <ThinkingIndicator />}
      </div>
    </div>
  )
}

function ThinkingIndicator() {
  return (
    <div className="flex animate-in gap-3 fade-in slide-in-from-bottom-2 duration-300">
      <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
        <Brain className="size-3.5" />
      </span>
      <div
        className="flex items-center gap-1 py-2"
        role="status"
        aria-label="Assistant is thinking"
      >
        {[0, 180, 360].map((delay) => (
          <span
            key={delay}
            className="size-[5px] animate-pulse rounded-full bg-muted-foreground/60 [animation-duration:1.2s]"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
    </div>
  )
}
