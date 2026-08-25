"use client"

import { cn } from "@/lib/utils"
import type { ChatMessage } from "@/stores/chat-store"

/**
 * A message from the user — right-aligned primary bubble, plain text
 * (user input is never markdown-rendered so typed syntax stays literal).
 */
export function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="flex animate-in justify-end fade-in slide-in-from-bottom-2 duration-300">
      <div
        className={cn(
          "max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5",
          "text-sm leading-relaxed whitespace-pre-wrap text-primary-foreground shadow-sm",
        )}
      >
        {message.content}
      </div>
    </div>
  )
}
