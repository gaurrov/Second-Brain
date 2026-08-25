"use client"

import * as React from "react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"

import { cn } from "@/lib/utils"

/**
 * Markdown renderer for assistant answers.
 *
 * Uses react-markdown (AST-based, no raw HTML injection by default) with
 * remark-gfm for tables, task lists, strikethrough and autolinks. Every
 * element is mapped to explicit Tailwind classes because the project does
 * not ship the typography plugin — this keeps headings, bold text, lists,
 * inline code, fenced code blocks and links readable in both themes.
 */

const markdownComponents: Components = {
  h1: ({ className, ...props }) => (
    <h1 className={cn("mt-5 mb-3 text-lg font-semibold tracking-tight first:mt-0", className)} {...props} />
  ),
  h2: ({ className, ...props }) => (
    <h2 className={cn("mt-5 mb-2.5 text-base font-semibold tracking-tight first:mt-0", className)} {...props} />
  ),
  h3: ({ className, ...props }) => (
    <h3 className={cn("mt-4 mb-2 text-sm font-semibold tracking-tight first:mt-0", className)} {...props} />
  ),
  h4: ({ className, ...props }) => (
    <h4 className={cn("mt-4 mb-2 text-sm font-semibold first:mt-0", className)} {...props} />
  ),
  p: ({ className, ...props }) => (
    <p className={cn("my-2.5 leading-relaxed first:mt-0 last:mb-0", className)} {...props} />
  ),
  strong: ({ className, ...props }) => (
    <strong className={cn("font-semibold text-foreground", className)} {...props} />
  ),
  em: ({ className, ...props }) => <em className={cn("italic", className)} {...props} />,
  a: ({ className, ...props }) => (
    <a
      target="_blank"
      rel="noopener noreferrer"
      className={cn("font-medium text-primary underline underline-offset-2 hover:opacity-80", className)}
      {...props}
    />
  ),
  ul: ({ className, ...props }) => (
    <ul className={cn("my-2.5 ml-5 list-disc space-y-1 marker:text-muted-foreground", className)} {...props} />
  ),
  ol: ({ className, ...props }) => (
    <ol className={cn("my-2.5 ml-5 list-decimal space-y-1 marker:text-muted-foreground", className)} {...props} />
  ),
  li: ({ className, ...props }) => (
    <li className={cn("leading-relaxed pl-1 [&>p]:my-0.5", className)} {...props} />
  ),
  blockquote: ({ className, ...props }) => (
    <blockquote
      className={cn(
        "my-3 border-l-2 border-border pl-3.5 text-muted-foreground italic [&>p]:my-1",
        className,
      )}
      {...props}
    />
  ),
  hr: ({ className, ...props }) => (
    <hr className={cn("my-4 border-border/70", className)} {...props} />
  ),
  table: ({ className, ...props }) => (
    <div className="my-3 overflow-x-auto rounded-lg border border-border">
      <table className={cn("w-full border-collapse text-xs", className)} {...props} />
    </div>
  ),
  thead: ({ className, ...props }) => (
    <thead className={cn("bg-muted/60", className)} {...props} />
  ),
  th: ({ className, ...props }) => (
    <th className={cn("border-b border-border px-3 py-2 text-left font-semibold", className)} {...props} />
  ),
  td: ({ className, ...props }) => (
    <td className={cn("border-b border-border/60 px-3 py-2 align-top", className)} {...props} />
  ),
  // Fenced code blocks render as `pre > code`; the reset utilities below
  // strip the inline-code chrome so only the pre's own styling shows.
  pre: ({ className, ...props }) => (
    <pre
      className={cn(
        "my-3 overflow-x-auto rounded-xl bg-zinc-950 p-3.5 text-xs leading-relaxed text-zinc-100",
        "border border-zinc-800 shadow-xs",
        "[&_code]:border-0 [&_code]:bg-transparent [&_code]:p-0 [&_code]:text-[inherit]",
        className,
      )}
      {...props}
    />
  ),
  code: ({ className, ...props }) => (
    <code
      className={cn(
        "rounded-md border border-border/60 bg-muted px-1.5 py-0.5 font-mono text-[0.8em] break-words",
        className,
      )}
      {...props}
    />
  ),
}

export function MarkdownContent({
  content,
  className,
}: {
  content: string
  className?: string
}) {
  return (
    <div className={cn("text-sm text-foreground", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {content}
      </ReactMarkdown>
    </div>
  )
}
