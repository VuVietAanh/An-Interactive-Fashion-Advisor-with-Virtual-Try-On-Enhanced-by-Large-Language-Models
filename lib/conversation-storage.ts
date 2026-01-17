// Conversation Storage Utilities for LocalStorage

export interface Conversation {
  id: string
  title: string
  createdAt: Date
  updatedAt: Date
  messages: Array<{
    id: string
    type: "user" | "assistant"
    content: string
    products?: any[]
    timestamp: Date
  }>
  context?: {
    baseQuery?: string | null
    filters: Record<string, any>
  }
  candidateUniverse?: any[]
}

const STORAGE_KEY = "fashion_chat_conversations"
const MAX_CONVERSATIONS = 50 // Limit to prevent storage overflow

// Generate a simple title from the first user query
export function generateConversationTitle(firstUserMessage: string): string {
  if (!firstUserMessage || firstUserMessage.trim().length === 0) {
    return "New Conversation"
  }

  // Clean up the message
  let title = firstUserMessage.trim()

  // Limit length
  if (title.length > 50) {
    title = title.substring(0, 47) + "..."
  }

  // Capitalize first letter
  title = title.charAt(0).toUpperCase() + title.slice(1)

  return title
}

// Save conversation to LocalStorage
export function saveConversation(conversation: Conversation): void {
  try {
    const conversations = getAllConversations()
    
    // Update existing or add new
    const index = conversations.findIndex(c => c.id === conversation.id)
    if (index >= 0) {
      conversations[index] = conversation
    } else {
      conversations.unshift(conversation) // Add to beginning
    }

    // Sort by updatedAt (newest first)
    conversations.sort((a, b) => 
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    )

    // Limit to MAX_CONVERSATIONS
    const limited = conversations.slice(0, MAX_CONVERSATIONS)

    // Save to LocalStorage
    localStorage.setItem(STORAGE_KEY, JSON.stringify(limited))
    
    // Dispatch custom event to notify other components in the same tab
    window.dispatchEvent(new CustomEvent("conversationUpdated", {
      detail: { conversationId: conversation.id }
    }))
  } catch (error) {
    console.error("Failed to save conversation:", error)
  }
}

// Get all conversations from LocalStorage
export function getAllConversations(): Conversation[] {
  try {
    const data = localStorage.getItem(STORAGE_KEY)
    if (!data) return []

    const conversations = JSON.parse(data) as Conversation[]
    
    // Convert date strings back to Date objects
    return conversations.map(conv => ({
      ...conv,
      createdAt: new Date(conv.createdAt),
      updatedAt: new Date(conv.updatedAt),
      messages: conv.messages.map(msg => ({
        ...msg,
        timestamp: new Date(msg.timestamp),
      })),
    }))
  } catch (error) {
    console.error("Failed to load conversations:", error)
    return []
  }
}

// Get a specific conversation by ID
export function getConversation(id: string): Conversation | null {
  const conversations = getAllConversations()
  return conversations.find(c => c.id === id) || null
}

// Delete a conversation
export function deleteConversation(id: string): void {
  try {
    const conversations = getAllConversations()
    const filtered = conversations.filter(c => c.id !== id)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered))
    
    // Dispatch custom event to notify other components
    window.dispatchEvent(new CustomEvent("conversationUpdated", {
      detail: { conversationId: id, action: "deleted" }
    }))
  } catch (error) {
    console.error("Failed to delete conversation:", error)
  }
}

// Delete all conversations
export function deleteAllConversations(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
    
    // Dispatch custom event to notify other components
    window.dispatchEvent(new CustomEvent("conversationUpdated", {
      detail: { action: "deletedAll" }
    }))
  } catch (error) {
    console.error("Failed to delete all conversations:", error)
  }
}

// Create a new conversation
export function createNewConversation(): Conversation {
  return {
    id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    title: "New Conversation",
    createdAt: new Date(),
    updatedAt: new Date(),
    messages: [],
    context: {
      baseQuery: null,
      filters: {},
    },
    candidateUniverse: [],
  }
}

// Format relative time (e.g., "2 min ago", "yesterday")
export function formatRelativeTime(date: Date): string {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return "Just now"
  if (diffMins < 60) return `${diffMins} min ago`
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`
  if (diffDays === 1) return "Yesterday"
  if (diffDays < 7) return `${diffDays} days ago`
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} week${Math.floor(diffDays / 7) > 1 ? "s" : ""} ago`
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} month${Math.floor(diffDays / 30) > 1 ? "s" : ""} ago`
  return date.toLocaleDateString()
}

