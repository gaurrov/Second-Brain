"use client"

import { Brain } from "lucide-react"

import { useChatStore } from "@/stores/chat-store"
import { ChatComposer } from "@/components/chat/chat-composer"
import { EmptyChatState } from "@/components/chat/empty-chat-state"
import { MessageList } from "@/components/chat/message-list"

/**
 * The chat surface — occupies the full content column to the right of
 * the sidebar. Composes the empty state, the message list and the
 * composer. All conversation logic lives in stores/chat-store.ts, which
 * talks only to the real backend (POST /chat/stream + /conversations).
 */
export function ChatArea() {
  const messages = useChatStore((state) => state.messages)
  const isStreaming = useChatStore((state) => state.isStreaming)
  const loadingHistory = useChatStore((state) => state.loadingHistory)
  const sendMessage = useChatStore((state) => state.sendMessage)
  const stopStreaming = useChatStore((state) => state.stopStreaming)

  const hasMessages = messages.length > 0

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {hasMessages ? (
        <MessageList messages={messages} isStreaming={isStreaming} />
      ) : loadingHistory ? (
        <LoadingHistory />
      ) : (
        <EmptyChatState onPick={(prompt) => void sendMessage(prompt)} />
      )}

      <ChatComposer
        isStreaming={isStreaming}
        onSend={(text) => void sendMessage(text)}
        onStop={stopStreaming}
      />
    </div>
  )
}

function LoadingHistory() {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center" aria-busy="true">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span className="flex size-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 text-primary-foreground shadow-sm">
          <Brain className="size-4 animate-pulse" />
        </span>
        Loading conversation…
      </div>
    </div>
  )
}
