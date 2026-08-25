import { AxiosError } from 'axios';

export class AppError extends Error {
  public statusCode: number;
  public data: unknown;

  constructor(message: string, statusCode: number = 500, data: unknown = null) {
    super(message);
    this.name = 'AppError';
    this.statusCode = statusCode;
    this.data = data;
  }
}

export function handleApiError(error: unknown): AppError {
  if (error instanceof AxiosError) {
    const status = error.response?.status || 500;
    // Fastapi returns details in 'detail' field
    const detail = error.response?.data?.detail;
    const message = typeof detail === 'string' 
      ? detail 
      : error.message || 'An unexpected API error occurred';
      
    return new AppError(message, status, error.response?.data);
  }

  if (error instanceof Error) {
    return new AppError(error.message, 500);
  }

  return new AppError('An unknown error occurred', 500);
}
