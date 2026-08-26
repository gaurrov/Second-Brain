"use client"

import * as React from "react"
import { usePathname, useRouter } from "next/navigation"
import { Check, MessageSquare, Plus, Trash2, X } from "lucide-react"

import type { Conversation } from "@/types"
import { toast } from "@/components/ui/toast"
import { Button } from "@/components/ui/button"
import { useChatStore } from "@/stores/chat-store"
import { groupConversationsByDate, type ConversationGroup } from "@/lib/conversation-groups"
import { handleApiError } from "@/lib/errors"
import { cn } from "@/lib/utils"

/**
 * Sidebar conversation section — the backend is the source of truth.
 *
 * Lists past conversations grouped by recency (Today / Yesterday / date),
 * lets the user start a new one, switch between them (history + citations
 * are reloaded from GET /conversations/{id}), and delete them.
 */
export function ConversationsSidebar() {
  const conversations = useChatStore((state) => state.conversations)
  const status = useChatStore((state) => state.conversationsStatus)
  const activeId = useChatStore((state) => state.activeConversationId)
  const loadConversations = useChatStore((state) => state.loadConversations)
  const openConversation = useChatStore((state) => state.openConversation)
  const newConversation = useChatStore((state) => state.newConversation)
  const deleteConversation = useChatStore((state) => state.deleteConversation)

  const router = useRouter()
  const pathname = usePathname()

  const [deletingId, setDeletingId] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (status === "idle") void loadConversations()
  }, [status, loadConversations])

  // Group per render with a stable clock captured once per mount pass.
  const groups = React.useMemo(
    () => groupConversationsByDate(conversations),
    [conversations],
  )

  const handleSelect = (id: string) => {
    void openConversation(id)
    if (!pathname.startsWith("/dashboard")) {
      router.push("/dashboard")
    }
  }

  const handleDelete = async (conversation: Conversation) => {
    setDeletingId(null)
    try {
      await deleteConversation(conversation.id)
      toast.add({ type: "success", title: "Conversation deleted" })
    } catch (error) {
      const appError = handleApiError(error)
      if (appError.statusCode === 401) {
        toast.add({ type: "error", title: "Session expired", description: "Please sign in again." })
      } else {
        toast.add({
          type: "error",
          title: "Could not delete conversation",
          description: appError.message,
        })
      }
    }
  }

  return (
    <section aria-label="Conversations" className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between px-1 pb-1.5">
        <h2 className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          Conversations
        </h2>
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={newConversation}
          aria-label="Start a new conversation"
          title="New conversation"
        >
          <Plus />
        </Button>
      </div>

      {status === "loading" && conversations.length === 0 ? (
        <SkeletonGroups />
      ) : status === "error" && conversations.length === 0 ? (
        <LoadErrorState onRetry={() => void loadConversations()} />
      ) : conversations.length === 0 ? (
        <EmptyHistory onNew={newConversation} />
      ) : (
        <div className="-mx-1 min-h-0 flex-1 overflow-y-auto px-1 pb-1" role="list">
          {groups.map((group) => (
            <ConversationGroupSection
              key={group.label}
              group={group}
              activeId={activeId}
              deletingId={deletingId}
              onSelect={handleSelect}
              onConfirmDelete={(id) => setDeletingId(id)}
              onCancelDelete={() => setDeletingId(null)}
              onDelete={(conversation) => void handleDelete(conversation)}
            />
          ))}
        </div>
      )}
    </section>
  )
}

function ConversationGroupSection({
  group,
  activeId,
  deletingId,
  onSelect,
  onConfirmDelete,
  onCancelDelete,
  onDelete,
}: {
  group: ConversationGroup
  activeId: string | null
  deletingId: string | null
  onSelect: (id: string) => void
  onConfirmDelete: (id: string) => void
  onCancelDelete: () => void
  onDelete: (conversation: Conversation) => void
}) {
  return (
    <div className="mb-2">
      <p className="px-2 pb-1 text-[10px] font-medium uppercase tracking-widest text-muted-foreground/80 first:pt-1">
        {group.label}
      </p>
      <ul className="space-y-0.5" role="presentation">
        {group.conversations.map((conversation) => (
          <li key={conversation.id} className="group/conv relative">
            <button
              type="button"
              role="listitem"
              aria-current={activeId === conversation.id || undefined}
              title={conversation.title}
              onClick={() => onSelect(conversation.id)}
              className={cn(
                "flex w-full items-center gap-2 rounded-lg p-2 pr-8 text-left outline-none transition-colors",
                activeId === conversation.id
                  ? "bg-sidebar-accent ring-1 ring-ring/20"
                  : "hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-ring/50",
              )}
            >
              <MessageSquare
                className={cn(
                  "size-4 shrink-0",
                  activeId === conversation.id ? "text-primary" : "text-muted-foreground/70",
                )}
              />
              <span className="block min-w-0 flex-1 truncate text-sm">
                {conversation.title}
              </span>
            </button>

            {/* Delete affordance with inline confirm, mirroring documents */}
            <span
              className={cn(
                "absolute inset-y-0 right-1 z-10 flex items-center transition-opacity",
                deletingId === conversation.id
                  ? "opacity-100"
                  : "opacity-0 group-hover/conv:opacity-100 focus-within:opacity-100",
              )}
            >
              {deletingId === conversation.id ? (
                <>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    onClick={() => {
                      onCancelDelete()
                      onDelete(conversation)
                    }}
                    aria-label={`Confirm deleting "${conversation.title}"`}
                    className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                  >
                    <Check />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    onClick={onCancelDelete}
                    aria-label={`Cancel deleting "${conversation.title}"`}
                  >
                    <X />
                  </Button>
                </>
              ) : (
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={() => onConfirmDelete(conversation.id)}
                  onBlur={() => onCancelDelete()}
                  aria-label={`Delete "${conversation.title}"`}
                  className="text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                >
                  <Trash2 />
                </Button>
              )}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function SkeletonGroups() {
  return (
    <div className="-mx-1 space-y-2 px-1" aria-hidden="true">
      {[0, 1].map((groupIndex) => (
        <div key={groupIndex}>
          <span className="mb-1 block h-2 w-14 animate-pulse rounded bg-muted" />
          {[0, 1].map((row) => (
            <div key={row} className="flex items-center gap-2 rounded-lg p-2">
              <span className="size-4 shrink-0 animate-pulse rounded bg-muted" />
              <span
                className="block h-3.5 animate-pulse rounded bg-muted"
                style={{ width: `${68 - row * 18 - groupIndex * 8}%` }}
              />
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

function LoadErrorState({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="rounded-xl bg-muted/30 px-3 py-5 text-center">
      <p className="text-xs text-muted-foreground">Couldn&apos;t load conversations.</p>
      <Button variant="link" size="xs" className="mt-1" onClick={onRetry}>
        Try again
      </Button>
    </div>
  )
}

function EmptyHistory({ onNew }: { onNew: () => void }) {
  return (
    <div className="rounded-xl bg-muted/30 px-3 py-5 text-center">
      <p className="text-xs leading-relaxed text-muted-foreground">
        No conversations yet.
        <br />
        Ask something to start one.
      </p>
      <Button variant="outline" size="xs" className="mt-2" onClick={onNew}>
        New conversation
      </Button>
    </div>
  )
}
