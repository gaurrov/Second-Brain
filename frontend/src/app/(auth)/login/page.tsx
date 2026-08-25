"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import * as z from "zod"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Brain, Eye, EyeOff, Loader2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { apiClient } from "@/lib/api"
import { handleApiError } from "@/lib/errors"
import { useAuthStore } from "@/stores/auth-store"
import type { User } from "@/types"

const loginSchema = z.object({
  email: z.string().email({ message: "Please enter a valid email address." }),
  password: z.string().min(1, { message: "Password is required." }),
})

type LoginFormValues = z.infer<typeof loginSchema>

export default function LoginPage() {
  const router = useRouter()
  const setTokens = useAuthStore((state) => state.setTokens)
  const setUser = useAuthStore((state) => state.setUser)

  const [showPassword, setShowPassword] = useState(false)
  const [globalError, setGlobalError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: { email: "", password: "" },
  })

  const onSubmit = async (data: LoginFormValues) => {
    setGlobalError(null)
    try {
      const response = await apiClient.post("/auth/login", data)
      const { access_token, refresh_token } = response.data

      setTokens(access_token, refresh_token)

      const profileResponse = await apiClient.get<User>("/users/profile")
      setUser(profileResponse.data)

      router.push("/dashboard")
    } catch (error) {
      const appError = handleApiError(error)
      setGlobalError(appError.message)
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background p-4">
      {/* Theme-aware dot grid */}
      <div className="absolute inset-0 bg-[radial-gradient(circle,var(--foreground)_0.5px,transparent_0.5px)] bg-[size:24px_24px] opacity-[0.04] pointer-events-none dark:opacity-[0.06]" />

      <Card className="z-10 w-full max-w-md border-border/50 bg-card/80 shadow-lg backdrop-blur-md">
        <CardHeader className="items-center space-y-3 pb-6">
          <div className="flex size-11 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm">
            <Brain className="size-5" />
          </div>
          <div className="space-y-1.5 text-center">
            <CardTitle className="font-display text-[1.75rem] leading-tight tracking-tight">
              Welcome back
            </CardTitle>
            <CardDescription className="text-muted-foreground">
              Sign in to access your knowledge base
            </CardDescription>
          </div>
        </CardHeader>

        <div className="px-6 pb-6">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            {globalError && (
              <div className="flex items-start gap-2 rounded-lg border border-destructive/25 bg-destructive/5 px-3.5 py-2.5 text-sm text-destructive">
                <span className="mt-0.5 block size-1.5 shrink-0 rounded-full bg-destructive" />
                <span>{globalError}</span>
              </div>
            )}

            <div className="space-y-2">
              <label
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                htmlFor="email"
              >
                Email
              </label>
              <Input
                id="email"
                type="email"
                placeholder="name@example.com"
                autoFocus
                {...register("email")}
                className={errors.email ? "border-destructive focus-visible:ring-destructive" : ""}
              />
              {errors.email && (
                <p className="text-[0.8rem] text-destructive">{errors.email.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <label
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                htmlFor="password"
              >
                Password
              </label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  {...register("password")}
                  className={`pr-10 ${errors.password ? "border-destructive focus-visible:ring-destructive" : ""}`}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground transition-colors hover:text-foreground"
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  <span className="sr-only">{showPassword ? "Hide password" : "Show password"}</span>
                </button>
              </div>
              {errors.password && (
                <p className="text-[0.8rem] text-destructive">{errors.password.message}</p>
              )}
            </div>

            <Button type="submit" className="mt-6 h-11 w-full text-sm font-medium" disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Signing in…
                </>
              ) : (
                "Sign In"
              )}
            </Button>
          </form>

          <div className="mt-6 text-center text-sm text-muted-foreground">
            Don&apos;t have an account?{" "}
            <Link href="/signup" className="font-medium text-primary underline underline-offset-2 transition-colors hover:opacity-80">
              Sign up
            </Link>
          </div>
        </div>
      </Card>
    </div>
  )
}
