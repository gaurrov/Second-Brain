import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, FileText, MessageSquare, Sparkles } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"

export default function Home() {
  return (
    <main className="flex-1 flex flex-col items-center justify-center p-6 md:p-24 relative overflow-hidden">
      {/* Subtle Background Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />

      <div className="absolute top-4 right-6 z-10">
        <ThemeToggle />
      </div>

      <div className="z-10 w-full max-w-5xl flex flex-col items-center text-center space-y-8">
        
        <div className="inline-flex items-center rounded-full border border-border px-3 py-1 text-sm text-muted-foreground bg-background/50 backdrop-blur-sm shadow-sm transition-colors hover:bg-muted/50 cursor-default">
          <Sparkles className="mr-2 h-4 w-4 text-primary" />
          <span>Memora v1.0 Foundation</span>
        </div>

        <h1 className="text-5xl md:text-7xl font-semibold tracking-tight text-foreground">
          Your Intelligent <br className="hidden md:block" />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-foreground to-muted-foreground">
            Second Brain
          </span>
        </h1>
        
        <p className="max-w-[42rem] leading-normal text-muted-foreground sm:text-xl sm:leading-8">
          Upload documents and converse with your private knowledge base seamlessly. 
          A sophisticated blend of ChatGPT-like interaction and Notion-like organization.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 mt-8 w-full justify-center">
          <Link href="/signup">
            <Button size="lg" className="rounded-full shadow-md h-12 px-8">
              Get Started
            </Button>
          </Link>
          <Link href="/login">
            <Button size="lg" variant="outline" className="rounded-full h-12 px-8 bg-background/50 backdrop-blur-sm border-border/50">
              View Documentation
            </Button>
          </Link>
        </div>
      </div>

      <div className="mt-24 z-10 w-full max-w-5xl grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-card/50 backdrop-blur-sm shadow-sm border-border/50 hover:shadow-md transition-all duration-300">
          <CardHeader>
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
              <FileText className="h-5 w-5 text-primary" />
            </div>
            <CardTitle>Document Ingestion</CardTitle>
            <CardDescription>Upload PDF, DOCX, and TXT files. Automatic chunking and vector embeddings via Qdrant.</CardDescription>
          </CardHeader>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm shadow-sm border-border/50 hover:shadow-md transition-all duration-300">
          <CardHeader>
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
              <Brain className="h-5 w-5 text-primary" />
            </div>
            <CardTitle>RAG Architecture</CardTitle>
            <CardDescription>Retrieval-Augmented Generation using local BAAI embeddings and Groq LLMs.</CardDescription>
          </CardHeader>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm shadow-sm border-border/50 hover:shadow-md transition-all duration-300">
          <CardHeader>
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
              <MessageSquare className="h-5 w-5 text-primary" />
            </div>
            <CardTitle>Streaming Chat</CardTitle>
            <CardDescription>Real-time token streaming with precise source citations and conversation history.</CardDescription>
          </CardHeader>
        </Card>
      </div>

      {/* Decorative Blur Orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute bottom-1/4 right-1/4 w-[30rem] h-[30rem] bg-secondary/5 rounded-full blur-3xl pointer-events-none -z-10" />
    </main>
  );
}
