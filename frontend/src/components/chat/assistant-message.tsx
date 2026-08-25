"use client"

import * as React from "react"
import { Brain, Check, Copy, TriangleAlert } from "lucide-react"

import type { ChatMessage } from "@/stores/chat-store"
import { cn } from "@/lib/utils"
import { MarkdownContent } from "@/components/chat/markdown-content"
import { SourceCards } from "@/components/chat/source-cards"
import { Button } from "@/components/ui/button"

/**
 * A message from the assistant — Memora avatar, markdown-rendered answer
 * (headings, bold, lists, code, code blocks, links), and the source chips
 * returned by the backend's final stream event. Failed exchanges render
 * the backend's error message in a destructive variant.
 *
 * While `isStreaming`, a blinking caret trails the answer; once finished
 * (and not failed), a copy action appears beneath it, ChatGPT-style.
 */
export function AssistantMessage({
  message,
  isStreaming = false,
}: {
  message: ChatMessage
  isStreaming?: boolean
}) {
  const empty = !message.content

  return (
    <div className="group/assistant flex animate-in gap-3 fade-in slide-in-from-bottom-2 duration-300">
      <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 text-primary-foreground shadow-sm">
        <Brain className="size-3.5" />
      </span>
      <div className="min-w-0 flex-1">
        {empty ? null : message.failed ? (
          <div className="flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/5 px-3.5 py-2.5 text-sm leading-relaxed text-destructive">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" />
            <span>{message.content}</span>
          </div>
        ) : (
          <>
            <MarkdownContent content={message.content} />
            {isStreaming && (
              <span
                aria-hidden="true"
                className="ml-0.5 inline-block h-4 w-[2px] animate-pulse rounded-full bg-primary align-text-bottom"
              />
            )}
          </>
        )}

        {!message.failed && message.sources && message.sources.length > 0 && (
          <SourceCards sources={message.sources} />
        )}

        {!empty && !message.failed && !isStreaming && <CopyAction content={message.content} />}
      </div>
    </div>
  )
}

/** Hover-revealed copy button with transient "copied" feedback. */
function CopyAction({ content }: { content: string }) {
  const [copied, setCopied] = React.useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard unavailable (permissions); silently keep the button.
    }
  }

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-xs"
      onClick={() => void handleCopy()}
      aria-label={copied ? "Copied" : "Copy answer"}
      className={cn(
        "mt-1 text-muted-foreground opacity-0 transition-opacity",
        "focus-visible:opacity-100 group-hover/assistant:opacity-100",
        copied && "opacity-100 text-emerald-600 dark:text-emerald-400",
      )}
    >
      {copied ? <Check /> : <Copy />}
    </Button>
  )
}
