import { create } from 'zustand';
import { apiClient } from '@/lib/api';
import { streamChat } from '@/lib/chat-stream';
import type { ChatSource, Conversation, ConversationMessage, SourceRef } from '@/types';

/**
 * Chat store — drives the Memora conversation against the real backend.
 *
 * Live exchange:
 *   POST /chat/stream  -> SSE token events appended to the assistant
 *                         message as they arrive, final "sources" event
 *                         attached as citations, "error" event surfaced
 *                         inline on the message.
 *
 * The stream does not carry the conversation id, so the first exchange of
 * a thread discovers it afterwards via GET /conversations (newest first by
 * created_at) and reuses it for follow-up turns — which is how the backend
 * attaches conversation history.
 *
 * Conversation management (sidebar):
 *   GET    /conversations            -> sidebar list (backend = truth)
 *   GET    /conversations/{id}       -> history incl. retrieval_metadata,
 *                                       mapped back into source cards
 *   DELETE /conversations/{id}       -> remove thread; resets an open one
 */

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /** Source citations from the final stream event (assistant only). */
  sources?: ChatSource[];
  /** Set when the backend reported a failure for this exchange. */
  failed?: boolean;
}

type ConversationsStatus = 'idle' | 'loading' | 'ready' | 'error';

interface ChatState {
  messages: ChatMessage[];
  /** True between sending a question and the stream finishing/failing. */
  isStreaming: boolean;
  /** The thread new sends attach to; null = a fresh conversation. */
  activeConversationId: string | null;

  /** Sidebar state (backend is the source of truth). */
  conversations: Conversation[];
  conversationsStatus: ConversationsStatus;

  /** True while a past conversation's messages are being fetched. */
  loadingHistory: boolean;

  loadConversations: (options?: { silent?: boolean }) => Promise<void>;
  openConversation: (id: string) => Promise<void>;
  newConversation: () => void;
  deleteConversation: (id: string) => Promise<void>;

  sendMessage: (text: string) => Promise<void>;
  stopStreaming: () => void;
  reset: () => void;
}

let abortController: AbortController | null = null;

/** Map a persisted backend message into the UI message shape. */
function messageFromHistory(message: ConversationMessage): ChatMessage | null {
  if (message.role !== 'user' && message.role !== 'assistant') return null;
  const sources = (message.retrieval_metadata ?? []).map((ref: SourceRef) => ({
    document_id: String(ref.document_id),
    filename: ref.filename,
    page: ref.page_number ?? null,
    chunk_index: ref.chunk_index,
  }));
  return {
    id: String(message.id),
    role: message.role,
    content: message.content,
    ...(message.role === 'assistant' && sources.length > 0 ? { sources } : {}),
  };
}

export const useChatStore = create<ChatState>()((set, get) => ({
  messages: [],
  isStreaming: false,
  activeConversationId: null,

  conversations: [],
  conversationsStatus: 'idle',

  loadingHistory: false,

  loadConversations: async (options) => {
    const silent = options?.silent ?? false;
    if (!silent || get().conversationsStatus !== 'loading') {
      set({ conversationsStatus: silent ? get().conversationsStatus : 'loading' });
    }
    try {
      const response = await apiClient.get<{
        total: number;
        conversations: Conversation[];
      }>('/conversations', { params: { limit: 100 } });
      set({ conversations: response.data.conversations, conversationsStatus: 'ready' });
    } catch {
      // A silent refresh keeps stale data rather than flashing an error.
      set({ conversationsStatus: get().conversations.length > 0 ? 'ready' : 'error' });
    }
  },

  openConversation: async (id) => {
    if (get().activeConversationId === id && get().messages.length > 0) return;
    abortController?.abort();
    abortController = null;
    set({ loadingHistory: true, isStreaming: false, activeConversationId: id });
    try {
      const response = await apiClient.get<{
        id: string;
        title: string;
        messages: ConversationMessage[];
      }>(`/conversations/${id}`, { params: { limit: 200 } });
      const messages = response.data.messages
        .map(messageFromHistory)
        .filter((message): message is ChatMessage => message !== null);
      set({ messages, loadingHistory: false });
    } catch {
      // Backend stays the source of truth: on failure leave nothing half-loaded.
      set({ messages: [], loadingHistory: false, activeConversationId: null });
    }
  },

  newConversation: () => {
    abortController?.abort();
    abortController = null;
    set({
      messages: [],
      isStreaming: false,
      loadingHistory: false,
      activeConversationId: null,
    });
  },

  deleteConversation: async (id) => {
    const wasActive = get().activeConversationId === id;
    if (wasActive) get().newConversation();
    set((state) => ({
      conversations: state.conversations.filter((conversation) => conversation.id !== id),
    }));
    try {
      await apiClient.delete(`/conversations/${id}`);
      void get().loadConversations({ silent: true });
    } catch {
      // Re-sync with the backend when the delete did not go through.
      void get().loadConversations({ silent: true });
    }
  },

  sendMessage: async (text) => {
    const content = text.trim();
    if (!content || get().isStreaming) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
    };
    // Placeholder filled in token-by-token by the stream.
    const assistantId = crypto.randomUUID();
    set((state) => ({
      messages: [
        ...state.messages,
        userMessage,
        { id: assistantId, role: 'assistant' as const, content: '' },
      ],
      isStreaming: true,
    }));

    abortController = new AbortController();
    const { signal } = abortController;

    const patchAssistant = (patch: Partial<ChatMessage>) =>
      set((state) => ({
        messages: state.messages.map((message) =>
          message.id === assistantId ? { ...message, ...patch } : message,
        ),
      }));

    await streamChat(
      content,
      get().activeConversationId,
      {
        onToken: (chunk) =>
          set((state) => ({
            messages: state.messages.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + chunk }
                : message,
            ),
          })),
        onDone: (sources: ChatSource[]) => {
          if (sources.length > 0) patchAssistant({ sources });
        },
        onError: (errorMessage: string) =>
          set((state) => ({
            messages: state.messages.map((message) =>
              message.id === assistantId && !message.content
                ? { ...message, failed: true, content: errorMessage }
                : message.id === assistantId
                  ? { ...message, failed: true }
                  : message,
            ),
          })),
      },
      signal,
    );
    abortController = null;

    // Post-stream reconciliation:
    //  - user stop with nothing received yet -> drop the empty placeholder;
    //  - clean close with nothing received   -> honest inline error (the
    //    frontend never fabricates an answer of its own);
    //  - otherwise leave the rendered backend text untouched.
    const finished = get().messages.find((message) => message.id === assistantId);
    if (finished) {
      if (!finished.content && signal.aborted) {
        set((state) => ({
          messages: state.messages.filter((message) => message.id !== assistantId),
        }));
      } else if (!finished.content && !finished.failed) {
        patchAssistant({
          failed: true,
          content: 'The assistant returned an empty response. Please try again.',
        });
      }
    }

    // Discover the thread id after the first exchange of a fresh thread so
    // follow-up turns share the same conversation (and its history). The
    // backend titles new threads with the question itself (first 80 chars),
    // so prefer that exact match over "newest" — robust even if another tab
    // created a thread in the meantime.
    if (!get().activeConversationId && !signal.aborted) {
      try {
        const response = await apiClient.get<{
          total: number;
          conversations: { id: string; title: string }[];
        }>('/conversations', { params: { limit: 5 } });
        const titlePrefix = content.slice(0, 80);
        const conversations = response.data.conversations;
        const match =
          conversations.find((conversation) => conversation.title === titlePrefix) ??
          conversations[0];
        if (match) set({ activeConversationId: match.id });
      } catch {
        // A missing id only means the next turn starts another thread.
      }
    }

    // Keep the sidebar in sync with the backend (new or updated thread).
    if (!signal.aborted) {
      void get().loadConversations({ silent: true });
    }

    set({ isStreaming: false });
  },

  stopStreaming: () => {
    abortController?.abort();
    abortController = null;
    set({ isStreaming: false });
  },

  reset: () => {
    abortController?.abort();
    abortController = null;
    set({
      messages: [],
      isStreaming: false,
      activeConversationId: null,
      conversations: [],
      conversationsStatus: 'idle',
      loadingHistory: false,
    });
  },
}));
