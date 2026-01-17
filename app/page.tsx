"use client"
import ChatInterface from "@/components/chat-interface"

export default function Home() {
  return (
    <main className="h-screen w-full bg-gradient-to-br from-background via-background to-muted">
      <ChatInterface />
    </main>
  )
}
