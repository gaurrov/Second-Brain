import type { Conversation } from '@/types';

/**
 * Groups conversations into sidebar buckets by recency:
 * Today / Yesterday / "Aug 22" (year appended when not current).
 *
 * Pure and timezone-honest: bucketing uses the caller's `now`, so it is
 * trivially unit-testable and consistent within a render pass.
 */

export interface ConversationGroup {
  label: string;
  conversations: Conversation[];
}

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

const DAY_MS = 24 * 60 * 60 * 1000;

export function groupConversationsByDate(
  conversations: Conversation[],
  now: Date = new Date(),
): ConversationGroup[] {
  const todayStart = startOfDay(now);
  const yesterdayStart = todayStart - DAY_MS;

  const groups = new Map<string, Conversation[]>();
  for (const conversation of conversations) {
    const updated = new Date(conversation.updated_at);
    const dayStart = startOfDay(updated);
    let label: string;
    if (dayStart >= todayStart) {
      label = 'Today';
    } else if (dayStart >= yesterdayStart) {
      label = 'Yesterday';
    } else {
      const sameYear = updated.getFullYear() === now.getFullYear();
      label = updated.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        ...(sameYear ? {} : { year: 'numeric' }),
      });
    }
    const bucket = groups.get(label);
    if (bucket) {
      bucket.push(conversation);
    } else {
      groups.set(label, [conversation]);
    }
  }
  return [...groups.entries()].map(([label, list]) => ({ label, conversations: list }));
}
