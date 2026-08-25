"use client"

import { useEffect, useSyncExternalStore } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuthStore } from '@/stores/auth-store';

const emptySubscribe = () => () => {};

/** True once the component has hydrated on the client, false during SSR/hydration. */
function useHydrated() {
  return useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false
  );
}

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { accessToken } = useAuthStore();
  const router = useRouter();
  const pathname = usePathname();
  const isMounted = useHydrated();

  useEffect(() => {
    if (!isMounted) return;

    const isAuthRoute = pathname.startsWith('/login') || pathname.startsWith('/signup');
    const isPublicRoute = pathname === '/';

    if (!accessToken && !isAuthRoute && !isPublicRoute) {
      router.push('/login');
    } else if (accessToken && isAuthRoute) {
      router.push('/dashboard'); // Main application entry after login
    }
  }, [accessToken, pathname, router, isMounted]);

  // Prevent hydration errors by not rendering children immediately if state might mismatch
  if (!isMounted) {
    return null;
  }

  return <>{children}</>;
}
