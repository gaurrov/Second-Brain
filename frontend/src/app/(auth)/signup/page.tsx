"use client"

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Eye, EyeOff, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { apiClient } from "@/lib/api";
import { handleApiError } from "@/lib/errors";

const signupSchema = z.object({
  username: z
    .string()
    .min(3, { message: "Username must be at least 3 characters." })
    .max(50, { message: "Username cannot exceed 50 characters." })
    .regex(/^[a-z][a-z0-9_]{2,49}$/, {
      message: "Must start with a letter and contain only lowercase letters, numbers, and underscores.",
    })
    .refine((val) => !val.includes("__"), {
      message: "Cannot contain consecutive underscores.",
    })
    .refine((val) => !val.endsWith("_"), {
      message: "Cannot end with an underscore.",
    }),
  email: z.string().email({ message: "Please enter a valid email address." }),
  password: z
    .string()
    .min(12, { message: "Password must be at least 12 characters." })
    .max(72, { message: "Password cannot exceed 72 bytes." })
    .refine((val) => !/\s/.test(val), {
      message: "Password cannot contain whitespace.",
    })
    .refine((val) => /[a-z]/.test(val), {
      message: "Must contain at least one lowercase letter.",
    })
    .refine((val) => /[A-Z]/.test(val), {
      message: "Must contain at least one uppercase letter.",
    })
    .refine((val) => /[0-9]/.test(val), {
      message: "Must contain at least one number.",
    })
    .refine((val) => /[^A-Za-z0-9]/.test(val), {
      message: "Must contain at least one special character.",
    }),
}).superRefine((data, ctx) => {
  const passwordLower = data.password.toLowerCase();
  const emailLocal = data.email.split("@")[0]?.toLowerCase() || "";
  
  if (data.username && passwordLower.includes(data.username.toLowerCase())) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "Password cannot contain your username.",
      path: ["password"],
    });
  }
  
  if (emailLocal && passwordLower.includes(emailLocal)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "Password cannot contain your email name.",
      path: ["password"],
    });
  }
});

type SignupFormValues = z.infer<typeof signupSchema>;

export default function SignupPage() {
  const router = useRouter();
  
  const [showPassword, setShowPassword] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);
  
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<SignupFormValues>({
    resolver: zodResolver(signupSchema),
    defaultValues: {
      username: "",
      email: "",
      password: "",
    },
  });

  const onSubmit = async (data: SignupFormValues) => {
    setGlobalError(null);
    try {
      await apiClient.post('/auth/register', data);
      
      // Auto-login after successful registration
      const loginResponse = await apiClient.post('/auth/login', {
        email: data.email,
        password: data.password,
      });
      
      const { access_token, refresh_token } = loginResponse.data;
      
      // We dynamically import the store so we don't cause circular dependency issues here, 
      // or we can just import it normally.
      const { useAuthStore } = await import("@/stores/auth-store");
      useAuthStore.getState().setTokens(access_token, refresh_token);
      
      // Fetch user profile
      const profileResponse = await apiClient.get('/users/profile');
      useAuthStore.getState().setUser(profileResponse.data);
      
      router.push('/dashboard');
    } catch (error) {
      const appError = handleApiError(error);
      setGlobalError(appError.message);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4 relative overflow-hidden">
      {/* Subtle Background Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />

      <Card className="w-full max-w-md z-10 bg-card/80 backdrop-blur-md shadow-lg border-border/50">
        <CardHeader className="space-y-1 pb-6">
          <CardTitle className="text-3xl font-semibold tracking-tight text-center">
            Create an account
          </CardTitle>
          <CardDescription className="text-center text-muted-foreground">
            Join Memora to start building your Second Brain
          </CardDescription>
        </CardHeader>
        
        <div className="px-6 pb-6">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            
            {globalError && (
              <div className="p-3 text-sm text-destructive bg-destructive/10 rounded-md border border-destructive/20 text-center">
                {globalError}
              </div>
            )}

            <div className="space-y-2">
              <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70" htmlFor="username">
                Username
              </label>
              <Input
                id="username"
                placeholder="johndoe"
                {...register("username")}
                className={errors.username ? "border-destructive focus-visible:ring-destructive" : ""}
              />
              {errors.username && (
                <p className="text-[0.8rem] text-destructive">{errors.username.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70" htmlFor="email">
                Email
              </label>
              <Input
                id="email"
                type="email"
                placeholder="name@example.com"
                {...register("email")}
                className={errors.email ? "border-destructive focus-visible:ring-destructive" : ""}
              />
              {errors.email && (
                <p className="text-[0.8rem] text-destructive">{errors.email.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70" htmlFor="password">
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
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  tabIndex={-1}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                  <span className="sr-only">{showPassword ? "Hide password" : "Show password"}</span>
                </button>
              </div>
              {errors.password && (
                <p className="text-[0.8rem] text-destructive">{errors.password.message}</p>
              )}
              <p className="text-xs text-muted-foreground mt-1">
                At least 12 characters, including uppercase, lowercase, number, and special character.
              </p>
            </div>

            <Button type="submit" className="w-full mt-6" disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating account...
                </>
              ) : (
                "Sign Up"
              )}
            </Button>
          </form>

          <div className="mt-6 text-center text-sm text-muted-foreground">
            Already have an account?{" "}
            <Link href="/login" className="text-primary hover:underline font-medium transition-colors">
              Sign in
            </Link>
          </div>
        </div>
      </Card>
    </div>
  );
}
