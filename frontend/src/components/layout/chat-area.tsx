"use client"

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
    <div
      className="mx-auto flex w-full max-w-3xl min-h-0 flex-1 flex-col gap-7 px-4 py-8 sm:px-6"
      aria-busy="true"
      aria-label="Loading conversation"
    >
      {/* Skeleton transcript: user turn, then assistant reply. */}
      <div className="flex justify-end">
        <div className="space-y-2" aria-hidden="true">
          <span className="block h-9 w-56 animate-pulse rounded-xl rounded-br-md bg-muted [animation-duration:1.5s]" />
        </div>
      </div>
      <div className="flex gap-3" aria-hidden="true">
        <span className="mt-0.5 flex size-7 shrink-0 animate-pulse items-center justify-center rounded-lg bg-muted [animation-duration:1.5s]" />
        <div className="min-w-0 flex-1 space-y-2.5 pt-1">
          <span className="block h-3 w-[92%] animate-pulse rounded bg-muted [animation-duration:1.5s]" />
          <span className="block h-3 w-[78%] animate-pulse rounded bg-muted [animation-duration:1.5s] [animation-delay:120ms]" />
          <span className="block h-3 w-[54%] animate-pulse rounded bg-muted [animation-duration:1.5s] [animation-delay:240ms]" />
        </div>
      </div>
      <span className="sr-only">Loading conversation…</span>
    </div>
  )
}
