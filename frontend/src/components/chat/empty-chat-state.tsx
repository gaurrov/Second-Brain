"use client"

import {
  BookOpenText,
  Boxes,
  Brain,
  Container,
  FileText,
  Layers,
} from "lucide-react"

import { cn } from "@/lib/utils"

/**
 * Initial chat state: Memora logo, tagline, and the suggested prompts.
 * Picking a prompt sends it immediately against the real backend.
 */
const SUGGESTED_PROMPTS = [
  {
    icon: Boxes,
    prompt: "What does my knowledge base say about microservices?",
  },
  {
    icon: FileText,
    prompt: "Summarize my uploaded documents.",
  },
  {
    icon: BookOpenText,
    prompt: "What did I write about RAG?",
  },
  {
    icon: Container,
    prompt: "What is Docker?",
  },
  {
    icon: Layers,
    prompt: "What is CQRS?",
  },
] as const

export function EmptyChatState({
  onPick,
}: {
  onPick: (prompt: string) => void
}) {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto">
      <div className="mx-auto flex w-full max-w-2xl flex-col items-center justify-center px-4 py-10 text-center sm:px-6">
        <div className="mb-6 flex size-14 animate-in items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-primary/70 text-primary-foreground shadow-md shadow-primary/15 fade-in zoom-in-95 duration-500">
          <Brain className="size-7" />
        </div>

        <h1 className="animate-in font-display text-[1.75rem] leading-tight tracking-tight text-foreground fade-in slide-in-from-bottom-2 duration-500">
          Your knowledge, at your fingertips.
        </h1>
        <p className="mt-2.5 max-w-md animate-in text-sm leading-relaxed text-muted-foreground fade-in slide-in-from-bottom-2 duration-500">
          Ask anything, or explore what you&apos;ve stored.
        </p>

        <div
          className={cn(
            "mt-9 grid w-full animate-in grid-cols-1 gap-2.5 fade-in slide-in-from-bottom-3",
            "duration-700 sm:grid-cols-2",
          )}
        >
          {SUGGESTED_PROMPTS.map(({ icon: Icon, prompt }) => (
            <button
              key={prompt}
              type="button"
              onClick={() => onPick(prompt)}
              className={cn(
                "group/suggestion flex items-start gap-3 rounded-xl border border-border/80 bg-card p-3.5 text-left shadow-xs outline-none",
                "transition-[border-color,background-color,box-shadow] duration-150 ease-out",
                "hover:border-ring/35 hover:bg-muted/40 hover:shadow-sm",
                "focus-visible:border-ring/60 focus-visible:ring-2 focus-visible:ring-ring/25",
              )}
            >
              <span
                className={cn(
                  "flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/[0.07] text-muted-foreground transition-colors duration-150",
                  "group-hover/suggestion:bg-primary group-hover/suggestion:text-primary-foreground",
                )}
              >
                <Icon className="size-4" />
              </span>
              <span className="min-w-0 pt-0.5 text-sm leading-snug font-medium">
                {prompt}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
