import axios from 'axios';
import { useAuthStore } from '@/stores/auth-store';
import type { ChatSource } from '@/types';

/**
 * Streaming chat client for POST /api/v1/chat/stream.
 *
 * The backend emits Server-Sent Events of the shape
 *   data: {"type":"token","content":"..."}\n\n
 *   data: {"type":"sources","sources":[...]}\n\n
 *   data: {"type":"error","content":"..."}\n\n
 * (see the docstring in src/api/v1/endpoints/chat.py).
 *
 * EventSource cannot send an Authorization header, so this uses fetch +
 * ReadableStream and parses SSE frames manually. A 401 triggers a single
 * token refresh (same contract as lib/api.ts) before failing.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export interface StreamChatCallbacks {
  onToken: (text: string) => void;
  onDone: (sources: ChatSource[]) => void;
  onError: (message: string) => void;
}

/** Refresh the auth tokens; returns true when new tokens were stored. */
async function refreshTokens(): Promise<boolean> {
  const refreshToken = useAuthStore.getState().refreshToken;
  if (!refreshToken) return false;
  try {
    const response = await axios.post(
      `${API_BASE}/auth/refresh`,
      { refresh_token: refreshToken },
      { headers: { 'Content-Type': 'application/json' } },
    );
    const tokens = response.data;
    useAuthStore.getState().setTokens(tokens.access_token, tokens.refresh_token);
    return true;
  } catch {
    return false;
  }
}

/**
 * Parse one complete SSE frame ("data: {...}") into a typed event object.
 * Returns null for frames that are not JSON data events (comments, etc.).
 */
function parseSseFrame(frame: string): Record<string, unknown> | null {
  const dataLines = frame
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart());
  if (dataLines.length === 0) return null;
  try {
    return JSON.parse(dataLines.join('\n'));
  } catch {
    return null;
  }
}

export async function streamChat(
  message: string,
  conversationId: string | null,
  callbacks: StreamChatCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  let attempt = 0;
  // Two attempts: initial + one retry after a token refresh.
  while (attempt < 2) {
    attempt += 1;
    const token = useAuthStore.getState().accessToken;

    let response: Response;
    try {
      response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          message,
          ...(conversationId ? { conversation_id: conversationId } : {}),
        }),
        signal,
      });
    } catch (error) {
      if (signal?.aborted) {
        callbacks.onDone([]);
        return;
      }
      callbacks.onError(
        error instanceof Error ? error.message : 'Could not reach the server.',
      );
      return;
    }

    if (response.status === 401 && attempt < 2 && (await refreshTokens())) {
      continue; // refreshed — retry with the new access token
    }

    if (!response.ok || !response.body) {
      let detail = `Request failed (${response.status}).`;
      try {
        const payload = await response.json();
        if (typeof payload?.detail === 'string') detail = payload.detail;
      } catch {
        // keep generic message
      }
      callbacks.onError(detail);
      return;
    }

    await readStream(response.body, callbacks, signal);
    return;
  }
}

async function readStream(
  body: ReadableStream<Uint8Array>,
  callbacks: StreamChatCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let sources: ChatSource[] = [];
  let streamFailed = false;

  const handleFrame = (frame: string) => {
    const event = parseSseFrame(frame);
    if (!event) return;
    switch (event.type) {
      case 'token':
        if (typeof event.content === 'string') callbacks.onToken(event.content);
        break;
      case 'sources':
        sources = Array.isArray(event.sources) ? (event.sources as ChatSource[]) : [];
        break;
      case 'error':
        streamFailed = true;
        callbacks.onError(
          typeof event.content === 'string'
            ? event.content
            : 'The assistant hit an unexpected error.',
        );
        break;
      default:
        break;
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let separatorIndex: number;
      while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        handleFrame(frame);
      }
      if (signal?.aborted) {
        reader.cancel().catch(() => undefined);
        break;
      }
    }
    // Flush any trailing frame not terminated by a blank line.
    const rest = buffer.trim();
    if (rest) handleFrame(rest);
  } catch {
    // Reader failures (network drop / abort). Only surface real errors:
    // an abort after tokens were received is a normal user stop.
    if (!signal?.aborted && !streamFailed) {
      callbacks.onError('The connection was interrupted.');
      return;
    }
  }

  if (!streamFailed) callbacks.onDone(sources);
}
