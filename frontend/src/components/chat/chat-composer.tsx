"use client"

import * as React from "react"
import { ArrowUp, Square } from "lucide-react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/**
 * Chat composer: auto-growing multiline textarea, Enter sends /
 * Shift+Enter inserts a newline, send button with loading state that
 * becomes a stop button while the assistant is streaming.
 */
export function ChatComposer({
  onSend,
  onStop,
  isStreaming,
}: {
  onSend: (text: string) => void
  onStop: () => void
  isStreaming: boolean
}) {
  const [value, setValue] = React.useState("")
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)

  // Keep focus after sending (e.g. from a suggested prompt).
  React.useEffect(() => {
    if (!isStreaming) textareaRef.current?.focus()
  }, [isStreaming])

  const canSend = value.trim().length > 0 && !isStreaming

  const submit = () => {
    const text = value.trim()
    if (!text || isStreaming) return
    setValue("")
    onSend(text)
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter") return
    // Let IME composition confirmations behave natively.
    if (event.shiftKey || event.nativeEvent.isComposing) return
    event.preventDefault()
    submit()
  }

  return (
    <div className="shrink-0 bg-gradient-to-t from-background via-background to-transparent px-4 pb-4 pt-2 sm:px-6">
      <form
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
        className="mx-auto max-w-3xl"
      >
        <div
          className={cn(
            "rounded-2xl border border-border bg-card shadow-xs shadow-black/[0.03] transition-[border-color,box-shadow] duration-200 ease-out",
            "focus-within:border-ring/50 focus-within:ring-[3px] focus-within:ring-ring/10",
          )}
        >
          <label htmlFor="chat-input" className="sr-only">
            Message Memora
          </label>
          <textarea
            id="chat-input"
            ref={textareaRef}
            rows={1}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything, or explore what you've stored…"
            aria-busy={isStreaming}
            className={cn(
              "block max-h-40 w-full resize-none bg-transparent px-4 pt-3.5 pb-1 text-sm outline-none",
              "placeholder:text-muted-foreground/80 field-sizing-content",
            )}
          />
          <div className="flex items-center justify-between gap-2 px-2.5 pb-2.5">
            <span className="hidden text-xs text-muted-foreground/90 sm:block">
              {isStreaming ? "Memora is responding…" : "Enter to send · Shift+Enter for a new line"}
            </span>
            {isStreaming ? (
              <Button
                type="button"
                size="icon-sm"
                onClick={onStop}
                aria-label="Stop generating"
                variant="secondary"
                className="rounded-full"
              >
                <Square className="size-3 fill-current" />
              </Button>
            ) : (
              <Button
                type="submit"
                size="icon-sm"
                disabled={!canSend}
                aria-label="Send message"
                className="rounded-full"
              >
                <ArrowUp />
              </Button>
            )}
          </div>
        </div>
        <p className="mt-2.5 text-center text-[11px] leading-relaxed text-muted-foreground/70">
          Memora answers from your knowledge base and may make mistakes — verify important information.
        </p>
      </form>
    </div>
  )
}
