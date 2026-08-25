// Global types for the frontend

export interface User {
  id: string;
  username: string;
  email: string;
  is_active: boolean;
  role: string;
  created_at: string;
  updated_at: string;
}

export interface Document {
  id: string;
  user_id: string;
  filename: string;
  file_type: string;
  upload_date: string;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  chunk_count: number;
  file_size_bytes: number;
  error_message?: string;
  updated_at?: string;
}

/** Mirrors GET /api/v1/config/upload (src/api/v1/endpoints/config.py). */
export interface UploadConfig {
  max_upload_size_mb: number;
  allowed_extensions: string[];
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

/** Mirrors the backend SourceSchema (src/api/v1/schemas/chat_schema.py). */
export interface ChatSource {
  document_id: string;
  filename: string;
  page: number | null;
  chunk_index: number;
}

/**
 * Full provenance persisted on assistant messages and returned by
 * GET /conversations/{id} as retrieval_metadata (SourceRefSchema).
 */
export interface SourceRef {
  document_id: string;
  filename: string;
  page_number: number | null;
  chunk_index: number;
  score: number;
  snippet: string;
}

/** Mirrors MessageResponse from GET /conversations/{id}. */
export interface ConversationMessage {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  retrieval_metadata?: SourceRef[] | null;
}
