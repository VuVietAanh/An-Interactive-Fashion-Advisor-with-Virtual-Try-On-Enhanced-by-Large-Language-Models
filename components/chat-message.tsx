import { Card } from "@/components/ui/card"

interface Message {
  type: "user" | "assistant"
  content: string
  timestamp: Date
}

export default function ChatMessage({ message }: { message: Message }) {
  const isUser = message.type === "user"

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <Card
        className={`max-w-xs rounded-2xl px-4 py-3 sm:max-w-sm lg:max-w-md ${
          isUser ? "bg-gradient-to-r from-primary to-accent text-primary-foreground" : "bg-card text-card-foreground"
        }`}
      >
        <p className="text-sm leading-relaxed">{message.content}</p>
        <p className={`mt-1 text-xs ${isUser ? "text-primary-foreground/70" : "text-muted-foreground"}`}>
          {message.timestamp.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })}
        </p>
      </Card>
    </div>
  )
}
