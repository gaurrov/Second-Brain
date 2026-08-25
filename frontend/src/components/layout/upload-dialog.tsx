"use client"

import * as React from "react"
import {
  Check,
  CircleAlert,
  CloudUpload,
  FileText,
  FileType2,
  LoaderCircle,
  NotebookPen,
  X,
} from "lucide-react"

import type { Document } from "@/types"
import { toast } from "@/components/ui/toast"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogCloseButton,
  DialogDescription,
  DialogHeader,
  DialogPopup,
  DialogTitle,
} from "@/components/ui/dialog"
import { useDocumentsStore, FALLBACK_MAX_UPLOAD_SIZE_MB } from "@/stores/documents-store"
import { handleApiError } from "@/lib/errors"
import { cn, formatBytes } from "@/lib/utils"

// Formats supported by this dialog (the product accepts PDF, DOCX, TXT;
// the backend allow-list is broader but this UI intentionally narrows it).
const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".txt"] as const
const ACCEPT_ATTRIBUTE = ".pdf,.docx,.txt"
/** Auto-dismiss a row this long after its document finishes indexing. */
const COMPLETED_ROW_TTL_MS = 4000
/** After this long in processing, reassure instead of looking stuck. */
const SLOW_PROCESSING_AFTER_MS = 20000

type QueuePhase = "uploading" | "active" | "rejected"

interface QueueItem {
  key: string
  filename: string
  size: number
  phase: QueuePhase
  /** Upload progress 0–100 (phase === "uploading"). */
  progress: number
  /** Wall-clock start of the transfer, for slow-processing reassurance. */
  startedAt: number
  /** Flipped by the in-flight ticker once processing runs unusually long. */
  slow?: boolean
  /** Server document id once the upload was accepted (phase === "active"). */
  docId?: string
  /** Client-side rejection or server-error reason (phase === "rejected"). */
  reason?: string
}

function FileIcon({ filename }: { filename: string }) {
  const extension = filename.slice(filename.lastIndexOf(".")).toLowerCase()
  const shared =
    "flex size-9 shrink-0 items-center justify-center rounded-lg border [&_svg]:size-4.5"
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

/**
 * Visual processing pipeline, mapped strictly onto real backend signals:
 *
 *   Upload     – actual client-side transfer progress (axios onUploadProgress)
 *   Processing – API processing_status PENDING ("queued") / PROCESSING ("running")
 *   Indexed    – API processing_status COMPLETED + real chunk_count from
 *                GET /documents/{id}/status
 *   Ready      – API processing_status COMPLETED
 *
 * The backend performs extract → clean → chunk → embed → store inside its
 * PROCESSING state but exposes no distinct embedding status, so Embedding
 * is NOT shown as its own stage — displaying it would mean simulating
 * progress that the API cannot verify.
 */
type StepState = "done" | "active" | "failed" | "idle"

function Pipeline({
  status,
  chunkCount,
}: {
  status: Document["processing_status"]
  chunkCount?: number
}) {
  const done = status === "completed"
  const failed = status === "failed"
  const processingSublabel =
    status === "pending" ? "queued on the server…" : "extracting & chunking…"

  const steps: { key: string; label: string; hint?: string; state: StepState }[] = [
    { key: "upload", label: "Upload", state: "done" },
    {
      key: "processing",
      label: failed ? "Processing failed" : "Processing",
      hint: done ? undefined : failed ? undefined : processingSublabel,
      state: done ? "done" : failed ? "failed" : "active",
    },
    {
      key: "indexed",
      label: "Indexed",
      hint:
        done && typeof chunkCount === "number"
          ? `${chunkCount} chunk${chunkCount === 1 ? "" : "s"}`
          : undefined,
      state: done ? "done" : "idle",
    },
    { key: "ready", label: "Ready", state: done ? "done" : "idle" },
  ]

  return (
    <ol className="space-y-1.5 rounded-lg bg-muted/50 p-2.5">
      {steps.map((step) => (
        <li
          key={step.key}
          className={cn(
            "flex items-center gap-2 text-xs [&_svg]:size-2.5",
            step.state === "done" && step.key === "ready"
              ? "font-medium text-emerald-700 dark:text-emerald-400"
              : step.state === "done"
                ? "text-muted-foreground"
                : step.state === "active"
                  ? "font-medium text-foreground"
                  : step.state === "failed"
                    ? "font-medium text-destructive"
                    : "text-muted-foreground/60",
          )}
        >
          <span
            className={cn(
              "flex size-4 shrink-0 items-center justify-center rounded-full border bg-card",
              step.state === "done" && step.key === "ready"
                ? "border-emerald-500/40"
                : step.state === "failed"
                  ? "border-destructive/40"
                  : "border-border",
            )}
          >
            {step.state === "done" ? (
              <Check className="text-emerald-600" />
            ) : step.state === "failed" ? (
              <CircleAlert className="text-destructive" />
            ) : step.state === "active" ? (
              <LoaderCircle className="animate-spin text-primary" />
            ) : (
              <span className="size-1 rounded-full bg-muted-foreground/30" />
            )}
          </span>
          {step.label}
          {step.hint && <span className="text-muted-foreground">· {step.hint}</span>}
        </li>
      ))}
    </ol>
  )
}

export function UploadDialogButton({
  variant = "default",
  size = "default",
  label = "Upload document",
  className,
}: {
  variant?: "default" | "outline"
  size?: "default" | "sm"
  label?: string
  className?: string
}) {
  const [open, setOpen] = React.useState(false)
  const [dragging, setDragging] = React.useState(false)
  const dragDepth = React.useRef(0)

  const uploadDocument = useDocumentsStore((state) => state.uploadDocument)
  const fetchUploadConfig = useDocumentsStore((state) => state.fetchUploadConfig)
  const getMaxUploadSizeMb = useDocumentsStore((state) => state.getMaxUploadSizeMb)
  const uploadConfig = useDocumentsStore((state) => state.uploadConfig)
  const documents = useDocumentsStore((state) => state.documents)

  const [items, setItems] = React.useState<QueueItem[]>([])
  const inputRef = React.useRef<HTMLInputElement>(null)
  // Keys whose auto-dismiss timer has already been scheduled, so re-renders
  // caused by other rows never reset an existing countdown.
  const dismissScheduled = React.useRef(new Set<string>())

  const maxUploadSizeMb =
    uploadConfig?.max_upload_size_mb ?? FALLBACK_MAX_UPLOAD_SIZE_MB

  // Pull the authoritative limits from the backend when the dialog opens.
  React.useEffect(() => {
    if (!open) return
    fetchUploadConfig().catch(() => undefined)
  }, [open, fetchUploadConfig])

  // Resolve live pipeline state for accepted rows.
  const docById = React.useMemo(
    () => new Map(documents.map((doc) => [doc.id, doc])),
    [documents],
  )

  // Auto-dismiss completed rows once (timer is not reset by other updates).
  React.useEffect(() => {
    for (const item of items) {
      if (item.phase !== "active" || !item.docId) continue
      if (dismissScheduled.current.has(item.key)) continue
      if (docById.get(item.docId)?.processing_status !== "completed") continue

      dismissScheduled.current.add(item.key)
      setTimeout(() => {
        dismissScheduled.current.delete(item.key)
        setItems((prev) => prev.filter((row) => row.key !== item.key))
      }, COMPLETED_ROW_TTL_MS)
    }
  }, [items, docById])

  const validate = (file: File): string | null => {
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase()
    if (!ACCEPTED_EXTENSIONS.includes(extension as (typeof ACCEPTED_EXTENSIONS)[number])) {
      return `“${extension || file.name}” is not supported — PDF, DOCX or TXT only.`
    }
    if (file.size === 0) {
      return "File is empty."
    }
    const limitMb = getMaxUploadSizeMb()
    if (file.size > limitMb * 1024 * 1024) {
      return `${formatBytes(file.size)} exceeds the ${limitMb} MB limit.`
    }
    return null
  }

  const enqueue = async (files: File[]) => {
    if (files.length === 0) return
    const rows: QueueItem[] = files.map((file) => ({
      key: crypto.randomUUID(),
      filename: file.name,
      size: file.size,
      phase: "uploading",
      progress: 0,
      startedAt: Date.now(),
      reason: validate(file) ?? undefined,
    }))
    setItems((prev) => [...prev, ...rows])

    await Promise.allSettled(
      rows.map(async (row, index) => {
        const file = files[index]
        if (row.reason) {
          setItems((prev) =>
            prev.map((entry) =>
              entry.key === row.key ? { ...entry, phase: "rejected" as const } : entry,
            ),
          )
          return
        }
        try {
          const doc = await uploadDocument(file, (percent) => {
            setItems((prev) =>
              prev.map((entry) =>
                entry.key === row.key ? { ...entry, progress: percent } : entry,
              ),
            )
          })
          setItems((prev) =>
            prev.map((entry) =>
              entry.key === row.key
                ? { ...entry, phase: "active" as const, docId: doc.id, progress: 100 }
                : entry,
            ),
          )
        } catch (error) {
          const appError = handleApiError(error)
          setItems((prev) =>
            prev.map((entry) =>
              entry.key === row.key
                ? { ...entry, phase: "rejected" as const, reason: appError.message }
                : entry,
            ),
          )
          toast.add({
            type: "error",
            title: `Could not upload ${row.filename}`,
            description: appError.message,
          })
        }
      }),
    )
  }

  const resetAndClose = (nextOpen: boolean) => {
    setOpen(nextOpen)
    if (!nextOpen) {
      setItems([])
      setDragging(false)
      dragDepth.current = 0
      dismissScheduled.current.clear()
    }
  }

  const removeRow = (key: string) => {
    setItems((prev) => prev.filter((row) => row.key !== key))
  }

  const settledCount = items.filter(
    (item) =>
      item.phase === "rejected" ||
      (item.docId &&
        ["completed", "failed"].includes(
          docById.get(item.docId)?.processing_status ?? "",
        )),
  ).length
  const hasUnfinished = items.length > settledCount

  // Flag rows that have been processing unusually long so the UI can
  // reassure instead of looking stuck. Runs on an interval (not during
  // render) and only flips one way — a row never un-becomes slow.
  React.useEffect(() => {
    if (!open || !hasUnfinished) return
    const id = setInterval(() => {
      const now = Date.now()
      setItems((prev) => {
        let changed = false
        const next = prev.map((item) => {
          if (
            item.slow ||
            item.phase !== "active" ||
            now - item.startedAt <= SLOW_PROCESSING_AFTER_MS
          ) {
            return item
          }
          changed = true
          return { ...item, slow: true }
        })
        return changed ? next : prev
      })
    }, 5000)
    return () => clearInterval(id)
  }, [open, hasUnfinished])

  return (
    <Dialog open={open} onOpenChange={resetAndClose}>
      <Button variant={variant} size={size} className={cn("w-full justify-start", className)} onClick={() => setOpen(true)}>
        <CloudUpload data-icon="inline-start" />
        {label}
      </Button>
      <DialogPopup>
        <DialogCloseButton />
        <DialogHeader>
          <DialogTitle>Upload documents</DialogTitle>
          <DialogDescription>
            PDF, DOCX or TXT · up to {maxUploadSizeMb} MB each.
            Documents become searchable in your knowledge base once indexed.
          </DialogDescription>
        </DialogHeader>

        {/* Dropzone */}
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload files: drop files here or browse"
          onClick={() => inputRef.current?.click()}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault()
              inputRef.current?.click()
            }
          }}
          onDragEnter={(event) => {
            event.preventDefault()
            dragDepth.current += 1
            setDragging(true)
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => {
            event.preventDefault()
            dragDepth.current -= 1
            if (dragDepth.current <= 0) setDragging(false)
          }}
          onDrop={(event) => {
            event.preventDefault()
            dragDepth.current = 0
            setDragging(false)
            void enqueue(Array.from(event.dataTransfer.files))
          }}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-center outline-none transition-all",
            "hover:border-ring/50 hover:bg-muted/40",
            "focus-visible:border-ring focus-visible:ring-4 focus-visible:ring-ring/20",
            dragging ? "border-primary bg-primary/5 scale-[1.01]" : "border-border"
          )}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPT_ATTRIBUTE}
            className="sr-only hidden"
            onChange={(event) => {
              void enqueue(Array.from(event.target.files ?? []))
              event.target.value = ""
            }}
            onClick={(event) => event.stopPropagation()}
            aria-hidden="true"
            tabIndex={-1}
          />
          <div
            className={cn(
              "flex size-12 items-center justify-center rounded-full transition-colors [&_svg]:size-6",
              dragging ? "bg-primary text-primary-foreground" : "bg-primary/10 text-primary"
            )}
          >
            <CloudUpload />
          </div>
          <p className="text-sm font-medium">
            {dragging ? "Drop to upload" : "Drag & drop files here"}
          </p>
          <p className="text-xs text-muted-foreground">
            or <span className="font-medium text-primary underline-offset-2">browse your files</span>
          </p>
        </div>

        {/* Queue */}
        {items.length > 0 && (
          <ul className="-mx-1 max-h-64 space-y-1 overflow-y-auto px-1" aria-live="polite">
            {items.map((item) => {
              const doc = item.docId ? docById.get(item.docId) : undefined
              const rejected = item.phase === "rejected"
              const uploading = item.phase === "uploading"
              const status = doc?.processing_status
              const completed = status === "completed"
              const failed = status === "failed"
              const slowProcessing =
                item.phase === "active" &&
                item.slow &&
                (status === "pending" || status === "processing")

              return (
                <li
                  key={item.key}
                  className={cn(
                    "group/row relative flex items-start gap-3 rounded-xl border p-3 pr-8 animate-in fade-in slide-in-from-bottom-1 duration-200",
                    rejected
                      ? "border-destructive/30 bg-destructive/5"
                      : completed
                        ? "border-emerald-500/25 bg-emerald-500/5"
                        : failed
                          ? "border-destructive/25 bg-background"
                          : "border-border bg-background"
                  )}
                >
                  <FileIcon filename={item.filename} />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">{item.filename}</p>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      {formatBytes(item.size)}
                      {uploading && ` · Uploading ${item.progress}%`}
                      {item.phase === "active" && !status && " · Accepted, waiting for update…"}
                      {rejected && item.reason && ` · ${item.reason}`}
                      {completed && doc && ` · Indexed ${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}`}
                      {failed && doc?.error_message && ` · ${doc.error_message}`}
                      {failed && doc && !doc.error_message && " · Processing failed"}
                    </p>

                    {uploading && (
                      <div className="mt-2 h-1 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-primary transition-[width] duration-200"
                          style={{ width: `${item.progress}%` }}
                        />
                      </div>
                    )}

                    {item.phase === "active" && doc && !completed && (
                      <div className="mt-2">
                        <Pipeline status={doc.processing_status} chunkCount={doc.chunk_count} />
                      </div>
                    )}
                    {item.phase === "active" && !doc && (
                      <p className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
                        <LoaderCircle className="size-3 animate-spin" />
                        Finishing up…
                      </p>
                    )}
                    {slowProcessing && (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Still working — large documents can take a while.
                      </p>
                    )}
                  </div>

                  {completed && (
                    <span className="mt-1 flex size-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 [&_svg]:size-3">
                      <Check className="text-emerald-600" />
                    </span>
                  )}

                  {/* Dismiss (finished/rejected rows only) */}
                  {(rejected || completed || failed) && (
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      onClick={() => removeRow(item.key)}
                      aria-label={`Dismiss ${item.filename}`}
                      className={cn(
                        "absolute top-2 right-2 text-muted-foreground opacity-0 transition-opacity",
                        "group-hover/row:opacity-100 focus-visible:opacity-100",
                        rejected && "opacity-100",
                      )}
                    >
                      <X />
                    </Button>
                  )}
                </li>
              )
            })}
          </ul>
        )}

        {hasUnfinished && (
          <p className="text-center text-xs text-muted-foreground">
            You can close this dialog — indexing continues in the background and your sidebar stays up to date.
          </p>
        )}
      </DialogPopup>
    </Dialog>
  )
}
