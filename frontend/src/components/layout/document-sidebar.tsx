"use client"

import * as React from "react"
import {
  Check,
  FileText,
  FileType2,
  LoaderCircle,
  NotebookPen,
  Search,
  Trash2,
  TriangleAlert,
  X,
} from "lucide-react"

import type { Document } from "@/types"
import { toast } from "@/components/ui/toast"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { UploadDialogButton } from "@/components/layout/upload-dialog"
import { useDocumentsStore } from "@/stores/documents-store"
import { handleApiError } from "@/lib/errors"
import { cn, formatBytes } from "@/lib/utils"

const STATUS_STYLES: Record<
  Document["processing_status"],
  { dot: string; label: string }
> = {
  completed: { dot: "bg-emerald-500", label: "Completed" },
  processing: { dot: "bg-amber-500 animate-pulse", label: "Processing" },
  pending: { dot: "bg-muted-foreground/40 animate-pulse", label: "Processing" },
  failed: { dot: "bg-destructive", label: "Failed" },
}

function FileIcon({ fileType }: { fileType: string }) {
  const extension = fileType.replace(".", "").toLowerCase()
  const shared =
    "flex size-8 shrink-0 items-center justify-center rounded-lg border [&_svg]:size-4"
  switch (extension) {
    case "pdf":
      return (
        <span className={cn(shared, "border-red-500/15 bg-red-500/10 text-red-600 dark:text-red-400")}>
          <FileText />
        </span>
      )
    case "docx":
    case "doc":
      return (
        <span className={cn(shared, "border-blue-500/15 bg-blue-500/10 text-blue-600 dark:text-blue-400")}>
          <FileType2 />
        </span>
      )
    default:
      return (
        <span className={cn(shared, "border-amber-500/15 bg-amber-500/10 text-amber-600 dark:text-amber-400")}>
          <NotebookPen />
        </span>
      )
  }
}

export function DocumentSidebar() {
  const documents = useDocumentsStore((state) => state.documents)
  const total = useDocumentsStore((state) => state.total)
  const loadStatus = useDocumentsStore((state) => state.loadStatus)
  const error = useDocumentsStore((state) => state.error)
  const uploads = useDocumentsStore((state) => state.uploads)
  const fetchDocuments = useDocumentsStore((state) => state.fetchDocuments)
  const deleteDocument = useDocumentsStore((state) => state.deleteDocument)

  const [query, setQuery] = React.useState("")
  const [deletingId, setDeletingId] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (loadStatus === "idle") void fetchDocuments()
  }, [loadStatus, fetchDocuments])

  const filtered = React.useMemo(() => {
    // The backend list endpoint has no search parameter (only
    // limit/offset), so filtering happens over the loaded page.
    const q = query.trim().toLowerCase()
    if (!q) return documents
    return documents.filter((doc) => doc.filename.toLowerCase().includes(q))
  }, [documents, query])

  const handleDelete = async (doc: Document) => {
    setDeletingId(doc.id)
    try {
      await deleteDocument(doc.id)
      toast.add({ type: "success", title: `${doc.filename} deleted` })
    } catch (deleteError) {
      const appError = handleApiError(deleteError)
      if (appError.statusCode === 401) {
        toast.add({ type: "error", title: "Session expired", description: "Please sign in again." })
      } else if (appError.statusCode === 404) {
        toast.add({ type: "warning", title: "Already deleted", description: "Refreshing your library." })
        void fetchDocuments({ silent: true })
      } else {
        toast.add({
          type: "error",
          title: `Could not delete ${doc.filename}`,
          description: appError.message,
        })
      }
    } finally {
      setDeletingId(null)
    }
  }

  const isLoading = loadStatus === "loading" && documents.length === 0
  const showLibraryEmpty =
    !isLoading && loadStatus !== "error" && total === 0 && uploads.length === 0
  const showNoResults =
    !showLibraryEmpty && documents.length > 0 && filtered.length === 0

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      {/* Actions */}
      <div className="space-y-2">
        <UploadDialogButton />
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search documents…"
            className="h-9 pl-8"
            aria-label="Search documents"
          />
        </div>
      </div>

      {/* List */}
      <div className="flex min-h-0 flex-1 flex-col" aria-busy={isLoading}>
        <div className="flex items-center justify-between px-1 pb-1.5">
          <h2 className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
            Documents
          </h2>
          {total > 0 && (
            <span className="rounded-full bg-muted px-1.5 py-0.5 text-[11px] font-medium text-muted-foreground tabular-nums">
              {total}
            </span>
          )}
        </div>

        {isLoading ? (
          <SkeletonList />
        ) : loadStatus === "error" && documents.length === 0 ? (
          <LoadErrorState
            message={error}
            onRetry={() => void fetchDocuments()}
          />
        ) : showLibraryEmpty ? (
          <EmptyLibrary />
        ) : showNoResults ? (
          <NoResults query={query} onClear={() => setQuery("")} />
        ) : (
          <ul
            className="-mx-1 min-h-0 flex-1 space-y-0.5 overflow-y-auto px-1 pb-1"
            role="list"
          >
            {/* In-flight uploads */}
            {uploads.map((task) => (
              <li key={task.id}>
                <div className="flex w-full items-center gap-2.5 rounded-lg p-2 opacity-80">
                  <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-primary/20 bg-primary/10 text-primary">
                    <LoaderCircle className="size-4 animate-spin" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">{task.filename}</span>
                    <span className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                      Uploading…
                      <span className="tabular-nums">{task.progress}%</span>
                    </span>
                    <span
                      role="progressbar"
                      aria-valuenow={task.progress}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      className="mt-1 block h-0.5 overflow-hidden rounded-full bg-muted"
                    >
                      <span
                        className="block h-full rounded-full bg-primary transition-[width] duration-200"
                        style={{ width: `${task.progress}%` }}
                      />
                    </span>
                  </span>
                </div>
              </li>
            ))}

            {/* Server-side documents */}
            {filtered.map((doc) => (
              <DocumentItem
                key={doc.id}
                doc={doc}
                deleting={deletingId === doc.id}
                onDelete={() => void handleDelete(doc)}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

function DocumentItem({
  doc,
  deleting,
  onDelete,
}: {
  doc: Document
  deleting: boolean
  onDelete: () => void
}) {
  const [confirming, setConfirming] = React.useState(false)
  const status = STATUS_STYLES[doc.processing_status]
  const failed = doc.processing_status === "failed"

  return (
    <li className="group/doc relative">
      <button
        type="button"
        title={
          failed && doc.error_message
            ? `${doc.filename}: ${doc.error_message}`
            : undefined
        }
        className={cn(
          "flex w-full items-center gap-2.5 rounded-lg p-2 pr-8 text-left outline-none transition-colors",
          "hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-ring/50"
        )}
      >
        <FileIcon fileType={doc.file_type} />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">{doc.filename}</span>
          <span className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className={cn("size-1.5 shrink-0 rounded-full", status.dot)} />
            <span className={failed ? "text-destructive" : undefined}>{status.label}</span>
            <span aria-hidden="true">·</span>
            <span className="truncate">{formatBytes(doc.file_size_bytes)}</span>
          </span>
        </span>
      </button>

      {/* Delete affordance */}
      <span
        className={cn(
          "absolute inset-y-0 right-1 z-10 flex items-center transition-opacity",
          confirming ? "opacity-100" : "opacity-0 group-hover/doc:opacity-100 focus-within:opacity-100"
        )}
      >
        {confirming ? (
          <>
            <Button
              variant="ghost"
              size="icon-xs"
              disabled={deleting}
              onClick={() => {
                setConfirming(false)
                onDelete()
              }}
              aria-label={`Confirm deleting ${doc.filename}`}
              className="text-destructive hover:bg-destructive/10 hover:text-destructive"
            >
              {deleting ? <LoaderCircle className="animate-spin" /> : <Check />}
            </Button>
            <Button
              variant="ghost"
              size="icon-xs"
              disabled={deleting}
              onClick={() => setConfirming(false)}
              aria-label={`Cancel deleting ${doc.filename}`}
            >
              <X />
            </Button>
          </>
        ) : (
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => setConfirming(true)}
            onBlur={() => setConfirming(false)}
            aria-label={`Delete ${doc.filename}`}
            className="text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
          >
            <Trash2 />
          </Button>
        )}
      </span>
    </li>
  )
}

function SkeletonList() {
  return (
    <ul className="-mx-1 space-y-0.5 px-1" aria-hidden="true">
      {[0, 1, 2, 3].map((index) => (
        <li key={index} className="flex items-center gap-2.5 rounded-lg p-2">
          <span className="size-8 shrink-0 animate-pulse rounded-lg bg-muted" />
          <span className="min-w-0 flex-1 space-y-1.5">
            <span
              className="block h-3.5 animate-pulse rounded bg-muted"
              style={{ width: `${72 - index * 9}%` }}
            />
            <span className="block h-2.5 w-2/5 animate-pulse rounded bg-muted" />
          </span>
        </li>
      ))}
    </ul>
  )
}

function LoadErrorState({
  message,
  onRetry,
}: {
  message: string | null
  onRetry: () => void
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 py-8 text-center">
      <div className="mb-3 flex size-10 items-center justify-center rounded-full bg-destructive/10">
        <TriangleAlert className="size-5 text-destructive" />
      </div>
      <p className="text-sm font-medium">Couldn&apos;t load your library</p>
      <p className="mt-1 max-w-[220px] truncate text-xs text-muted-foreground">
        {message ?? "Something went wrong."}
      </p>
      <Button variant="outline" size="sm" className="mt-4" onClick={onRetry}>
        Try again
      </Button>
    </div>
  )
}

function EmptyLibrary() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center rounded-xl bg-muted/30 px-4 py-8 text-center">
      <div className="mb-3 flex size-10 items-center justify-center rounded-full bg-primary/10">
        <FileText className="size-5 text-primary" />
      </div>
      <p className="text-sm font-medium">Your knowledge base is empty</p>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
        Upload a document to start chatting with your knowledge.
      </p>
      <UploadDialogButton
        variant="outline"
        size="sm"
        label="Upload"
        className="mt-4"
      />
    </div>
  )
}

function NoResults({ query, onClear }: { query: string; onClear: () => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center bg-muted/30 px-4 py-8 text-center">
      <Search className="mb-2 size-5 text-muted-foreground/60" />
      <p className="text-sm font-medium">No matches</p>
      <p className="mt-1 text-xs text-muted-foreground">
        Nothing found for &ldquo;{query.trim()}&rdquo;.
      </p>
      <Button variant="link" size="xs" className="mt-2" onClick={onClear}>
        Clear search
      </Button>
    </div>
  )
}
