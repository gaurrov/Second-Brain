import { create } from 'zustand';
import { apiClient } from '@/lib/api';
import { handleApiError, AppError } from '@/lib/errors';
import { toast } from '@/components/ui/toast';
import type { Document, UploadConfig } from '@/types';

/**
 * Documents store — the signed-in user's personal knowledge library.
 *
 * Security: every request is authenticated with the JWT attached by the
 * axios request interceptor; the backend scopes all results to the token's
 * subject. The client never sends a user_id parameter.
 *
 * Mirrors the backend schemas exactly (src/api/v1/schemas/document_schema.py):
 *   DocumentListResponse   { total: number; documents: Document[] }
 *   DocumentResponse       { id, user_id, filename, file_type, upload_date,
 *                            processing_status, chunk_count, file_size_bytes,
 *                            error_message?, updated_at? }
 *   DocumentStatusResponse { id, processing_status, chunk_count,
 *                            error_message?, updated_at? }
 *   UploadConfigResponse   { max_upload_size_mb, allowed_extensions }  (public)
 *
 * Processing flow (real backend, never simulated):
 *   POST /documents/upload        -> 202 + DocumentResponse (status PENDING)
 *   GET  /documents/{id}/status   -> polled while PENDING/PROCESSING until
 *                                    COMPLETED or FAILED.
 */

export type LoadStatus = 'idle' | 'loading' | 'ready' | 'error';

/** Client-side task for a file currently being uploaded (not yet a Document). */
export interface UploadTask {
  id: string;
  filename: string;
  /** 0–100 */
  progress: number;
}

/** Product decision: this UI accepts the three core formats. */
const UI_ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt'];
/**
 * Fallback used only before GET /config/upload has responded; the real,
 * authoritative limit always comes from the backend configuration.
 */
export const FALLBACK_MAX_UPLOAD_SIZE_MB = 25;

/** Backend statuses that mean "keep polling /documents/{id}/status". */
const ACTIVE_STATUSES = new Set(['pending', 'processing']);
const STATUS_POLL_INTERVAL_MS = 2000;
/** Stop polling after this many consecutive failures (backend unreachable). */
const MAX_CONSECUTIVE_FAILURES = 3;

interface DocumentsState {
  documents: Document[];
  total: number;
  loadStatus: LoadStatus;
  error: string | null;
  uploads: UploadTask[];
  /** Live upload constraints fetched from GET /config/upload. */
  uploadConfig: UploadConfig | null;

  fetchDocuments: (options?: { silent?: boolean }) => Promise<void>;
  fetchUploadConfig: () => Promise<UploadConfig>;
  /** Configured max size in MB (backend value once known, fallback before). */
  getMaxUploadSizeMb: () => number;
  uploadDocument: (
    file: File,
    onProgress?: (percent: number) => void,
  ) => Promise<Document>;
  deleteDocument: (id: string) => Promise<void>;
  reset: () => void;
}

let pollTimer: ReturnType<typeof setTimeout> | null = null;
let consecutiveFailures = 0;
let statusPollSequence = 0;

function stopPolling() {
  if (pollTimer !== null) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

interface StatusPayload {
  id: string;
  processing_status: Document['processing_status'];
  chunk_count: number;
  error_message?: string | null;
  updated_at?: string | null;
}

export const useDocumentsStore = create<DocumentsState>()((set, get) => {
  /**
   * Poll `GET /documents/{id}/status` for every document still in a
   * PENDING/PROCESSING state, patching results into the list. Stops when
   * nothing is in flight.
   */
  function scheduleStatusPoll() {
    stopPolling();
    const activeIds = get()
      .documents.filter((doc) => ACTIVE_STATUSES.has(doc.processing_status))
      .map((doc) => doc.id);
    if (
      activeIds.length === 0 ||
      consecutiveFailures >= MAX_CONSECUTIVE_FAILURES
    ) {
      return;
    }
    pollTimer = setTimeout(async () => {
      const sequence = ++statusPollSequence;
      const responses = await Promise.allSettled(
        activeIds.map((id) =>
          apiClient.get<StatusPayload>(`/documents/${id}/status`),
        ),
      );
      if (sequence !== statusPollSequence) return; // superseded

      let failures = 0;
      let sawTransitionToDone = false;
      set((state) => {
        let documents = state.documents;
        for (const result of responses) {
          if (result.status === 'rejected') {
            failures += 1;
            continue;
          }
          const payload = result.value.data;
          const existing = documents.find((doc) => doc.id === payload.id);
          if (!existing) continue;
          if (existing.processing_status === payload.processing_status) {
            // No change — only refresh chunk/error metadata when present.
            continue;
          }
          sawTransitionToDone =
            sawTransitionToDone ||
            payload.processing_status === 'completed' ||
            payload.processing_status === 'failed';
          documents = documents.map((doc) =>
            doc.id === payload.id
              ? {
                  ...doc,
                  processing_status: payload.processing_status,
                  chunk_count: payload.chunk_count ?? doc.chunk_count,
                  error_message:
                    payload.error_message ?? doc.error_message ?? undefined,
                  updated_at: payload.updated_at ?? doc.updated_at,
                }
              : doc,
          );
          if (payload.processing_status === 'completed') {
            toast.add({
              type: 'success',
              title: `${existing.filename} ready`,
              description: `Indexed ${payload.chunk_count} chunk${payload.chunk_count === 1 ? '' : 's'}.`,
            });
          } else if (payload.processing_status === 'failed') {
            toast.add({
              type: 'error',
              title: `${existing.filename} failed to process`,
              description: payload.error_message ?? 'The pipeline hit an error.',
            });
          }
        }
        return { documents };
      });

      if (failures > 0 && failures === responses.length) {
        consecutiveFailures += 1;
      } else {
        consecutiveFailures = 0;
      }

      if (sawTransitionToDone || get().total !== get().documents.length) {
        // Refresh ordering/metadata once things settle.
        void get().fetchDocuments({ silent: true });
      }
      scheduleStatusPoll();
    }, STATUS_POLL_INTERVAL_MS);
  }

  return {
    documents: [],
    total: 0,
    loadStatus: 'idle',
    error: null,
    uploads: [],
    uploadConfig: null,

    fetchUploadConfig: async () => {
      const cached = get().uploadConfig;
      if (cached) return cached;
      const response = await apiClient.get<UploadConfig>('/config/upload');
      set({ uploadConfig: response.data });
      return response.data;
    },

    getMaxUploadSizeMb: () =>
      get().uploadConfig?.max_upload_size_mb ?? FALLBACK_MAX_UPLOAD_SIZE_MB,

    fetchDocuments: async ({ silent = false } = {}) => {
      if (!silent) set({ loadStatus: 'loading', error: null });
      try {
        const response = await apiClient.get<{
          total: number;
          documents: Document[];
        }>('/documents', { params: { limit: 200 } });
        consecutiveFailures = 0;
        set({
          documents: response.data.documents,
          total: response.data.total,
          loadStatus: 'ready',
          error: null,
        });
        scheduleStatusPoll();
      } catch (error) {
        const appError = handleApiError(error);
        consecutiveFailures += 1;
        if (silent && consecutiveFailures < MAX_CONSECUTIVE_FAILURES) {
          scheduleStatusPoll();
          return;
        }
        stopPolling();
        set({
          ...(get().documents.length === 0 || !silent
            ? { loadStatus: 'error' as const }
            : {}),
          error:
            appError.statusCode === 401
              ? 'Your session has expired. Please sign in again.'
              : appError.message,
        });
      }
    },

    uploadDocument: async (file, onProgress) => {
      const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
      if (!UI_ALLOWED_EXTENSIONS.includes(extension)) {
        throw new AppError(
          `“${extension || file.name}” files are not supported — PDF, DOCX or TXT only.`,
          415,
        );
      }
      const maxMb = get().getMaxUploadSizeMb();
      if (file.size > maxMb * 1024 * 1024) {
        throw new AppError(`${file.name} exceeds the ${maxMb} MB limit.`, 413);
      }

      const taskId = crypto.randomUUID();
      set((state) => ({
        uploads: [...state.uploads, { id: taskId, filename: file.name, progress: 0 }],
      }));

      try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await apiClient.post<Document>(
          '/documents/upload',
          formData,
          {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: (event) => {
              const progress = event.total
                ? Math.round((event.loaded / event.total) * 100)
                : 0;
              set((state) => ({
                uploads: state.uploads.map((task) =>
                  task.id === taskId ? { ...task, progress } : task,
                ),
              }));
              onProgress?.(progress);
            },
          },
        );
        const uploaded = response.data;
        set((state) => ({
          uploads: state.uploads.filter((task) => task.id !== taskId),
          documents: [uploaded, ...state.documents],
          total: state.total + 1,
          loadStatus: 'ready',
        }));
        scheduleStatusPoll();
        return uploaded;
      } catch (error) {
        set((state) => ({
          uploads: state.uploads.filter((task) => task.id !== taskId),
        }));
        const appError = handleApiError(error);
        if (appError.statusCode === 413) {
          // Server disagreed with our cached size limit — refresh it.
          void get()
            .fetchUploadConfig()
            .catch(() => undefined);
        }
        throw appError;
      }
    },

    deleteDocument: async (id: string) => {
      await apiClient.delete(`/documents/${id}`);
      set((state) => ({
        documents: state.documents.filter((doc) => doc.id !== id),
        total: Math.max(0, state.total - 1),
      }));
    },

    reset: () => {
      stopPolling();
      consecutiveFailures = 0;
      statusPollSequence += 1;
      set({
        documents: [],
        total: 0,
        loadStatus: 'idle',
        error: null,
        uploads: [],
        uploadConfig: null,
      });
    },
  };
});
