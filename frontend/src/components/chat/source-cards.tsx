"use client"

import * as React from "react"
import { BookOpenText, FileText, FileType2, LoaderCircle, NotebookPen, TriangleAlert } from "lucide-react"

import type { ChatSource, Document } from "@/types"
import { apiClient } from "@/lib/api"
import { cn, formatBytes } from "@/lib/utils"
import {
  Dialog,
  DialogCloseButton,
  DialogDescription,
  DialogHeader,
  DialogPopup,
  DialogTitle,
} from "@/components/ui/dialog"

/**
 * Citation cards for one assistant answer.
 *
 * The cards render ONLY the fields the backend's final "sources" event
 * actually carries (document_id, filename, page, chunk_index — see
 * rag_service.answer_stream). Clicking a card fetches that document's
 * real metadata from GET /documents/{id} and shows it in a dialog; if
 * that request fails, an honest error state is shown instead of data.
 */

type DetailState =
  | { phase: "loading" }
  | { phase: "ready"; document: Document }
  | { phase: "error" }

/** One card per unique (document, page) pair the backend cited. */
function dedupeByDocumentPage(sources: ChatSource[]) {
  const seen = new Set<string>()
  return sources.filter((source) => {
    const key = `${source.document_id}:${source.page ?? "-"}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export function SourceCards({ sources }: { sources: ChatSource[] }) {
  const [selected, setSelected] = React.useState<ChatSource | null>(null)
  const [detail, setDetail] = React.useState<DetailState>({ phase: "loading" })

  const cards = dedupeByDocumentPage(sources)

  const openCard = (source: ChatSource) => {
    setSelected(source)
    setDetail({ phase: "loading" })
  }

  // Fetch live document details whenever a card is opened (the loading
  // state is set by openCard; this effect only applies the async result).
  React.useEffect(() => {
    if (!selected) return
    let ignore = false
    apiClient
      .get<Document>(`/documents/${selected.document_id}`)
      .then((response) => {
        if (!ignore) setDetail({ phase: "ready", document: response.data })
      })
      .catch(() => {
        if (!ignore) setDetail({ phase: "error" })
      })
    return () => {
      ignore = true
    }
  }, [selected])

  return (
    <div className="mt-3">
      <p className="mb-1.5 flex items-center gap-1 text-xs font-medium text-muted-foreground">
        <BookOpenText className="size-3" />
        Sources
      </p>
      <div className="flex flex-wrap gap-2">
        {cards.map((source) => (
          <button
            key={`${source.document_id}-${source.page ?? "nopage"}`}
            type="button"
            onClick={() => openCard(source)}
            title={typeof source.page === "number" ? `${source.filename} — page ${source.page}` : source.filename}
            aria-label={`Open details for ${source.filename}${typeof source.page === "number" ? `, page ${source.page}` : ""}`}
            className={cn(
              "group/source flex items-center gap-2 rounded-xl border border-border/80 bg-card px-2.5 py-2 text-left shadow-xs outline-none transition-all",
              "hover:-translate-y-0.5 hover:border-ring/40 hover:shadow-md",
              "focus-visible:ring-2 focus-visible:ring-ring/50 active:translate-y-0",
            )}
          >
            <SourceFileIcon filename={source.filename} />
            <span className="min-w-0">
              <span className="block max-w-[180px] truncate text-[13px] leading-tight font-medium">
                {source.filename}
              </span>
              {typeof source.page === "number" && (
                <span className="block text-[11px] leading-tight text-muted-foreground">
                  Page {source.page}
                </span>
              )}
            </span>
          </button>
        ))}
      </div>

      <Dialog open={selected !== null} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogPopup className="sm:max-w-md">
          <DialogCloseButton />
          {selected === null ? null : detail.phase === "loading" ? (
            <div className="flex items-center justify-center gap-2 py-10 text-sm text-muted-foreground">
              <LoaderCircle className="size-4 animate-spin" />
              Loading document details…
            </div>
          ) : detail.phase === "error" ? (
            <div className="flex items-start gap-2 rounded-xl border border-destructive/30 bg-destructive/5 px-3.5 py-3 text-sm text-destructive">
              <TriangleAlert className="mt-0.5 size-4 shrink-0" />
              <span>Could not load details for this document. It may have been deleted.</span>
            </div>
          ) : (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2 pr-8">
                  <SourceFileIcon filename={detail.document.filename} large />
                  <span className="min-w-0 break-all">{detail.document.filename}</span>
                </DialogTitle>
                <DialogDescription>Details returned for the source behind this answer.</DialogDescription>
              </DialogHeader>
              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2.5 text-sm">
                <dt className="text-muted-foreground">Type</dt>
                <dd className="font-medium uppercase">{detail.document.file_type}</dd>
                <dt className="text-muted-foreground">Status</dt>
                <dd className="font-medium capitalize">{detail.document.processing_status}</dd>
                <dt className="text-muted-foreground">Indexed chunks</dt>
                <dd className="font-medium tabular-nums">{detail.document.chunk_count}</dd>
                <dt className="text-muted-foreground">File size</dt>
                <dd className="font-medium">{formatBytes(detail.document.file_size_bytes)}</dd>
                <dt className="text-muted-foreground">Uploaded</dt>
                <dd className="font-medium">{formatDate(detail.document.upload_date)}</dd>
                {detail.document.error_message && (
                  <>
                    <dt className="text-destructive">Error</dt>
                    <dd className="text-destructive">{detail.document.error_message}</dd>
                  </>
                )}
              </dl>
            </>
          )}
        </DialogPopup>
      </Dialog>
    </div>
  )
}

/** Extension-tinted glyph, mirroring upload-dialog's icon treatment. */
function SourceFileIcon({ filename, large = false }: { filename: string; large?: boolean }) {
  const extension = filename.slice(filename.lastIndexOf(".")).toLowerCase()
  const shared = cn(
    "flex shrink-0 items-center justify-center rounded-lg border",
    large ? "size-10 [&_svg]:size-5" : "size-8 [&_svg]:size-4",
  )
  if (extension === ".pdf") {
    return (
      <span className={cn(shared, "border-red-500/15 bg-red-500/10 text-red-600 dark:text-red-400")}>
        <FileText />
      </span>
    )
  }
  if (extension === ".docx") {
    return (
      <span className={cn(shared, "border-blue-500/15 bg-blue-500/10 text-blue-600 dark:text-blue-400")}>
        <FileType2 />
      </span>
    )
  }
  return (
    <span className={cn(shared, "border-amber-500/15 bg-amber-500/10 text-amber-600 dark:text-amber-400")}>
      <NotebookPen />
    </span>
  )
}

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  } catch {
    return iso
  }
}
