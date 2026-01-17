"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
} from "@/components/ui/sidebar"
import { Plus, Trash2, MessageSquare } from "lucide-react"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import {
  getAllConversations,
  deleteConversation,
  deleteAllConversations,
  formatRelativeTime,
  type Conversation,
} from "@/lib/conversation-storage"

interface ConversationSidebarProps {
  currentConversationId: string | null
  onSelectConversation: (conversationId: string | null) => void
  onNewConversation: () => void
}

export default function ConversationSidebar({
  currentConversationId,
  onSelectConversation,
  onNewConversation,
}: ConversationSidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [conversationToDelete, setConversationToDelete] = useState<string | null>(null)

  // Load conversations from LocalStorage
  useEffect(() => {
    loadConversations()
    
    // Listen for storage changes (from other tabs/windows)
    const handleStorageChange = () => {
      loadConversations()
    }
    
    // Listen for custom events (from same tab)
    const handleConversationUpdated = () => {
      loadConversations()
    }
    
    window.addEventListener("storage", handleStorageChange)
    window.addEventListener("conversationUpdated", handleConversationUpdated)
    
    return () => {
      window.removeEventListener("storage", handleStorageChange)
      window.removeEventListener("conversationUpdated", handleConversationUpdated)
    }
  }, [])

  const loadConversations = () => {
    const all = getAllConversations()
    setConversations(all)
  }

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setConversationToDelete(id)
    setDeleteDialogOpen(true)
  }

  const confirmDelete = () => {
    if (conversationToDelete) {
      deleteConversation(conversationToDelete)
      loadConversations()
      
      // If deleting current conversation, switch to new
      if (conversationToDelete === currentConversationId) {
        onNewConversation()
      }
    }
    setDeleteDialogOpen(false)
    setConversationToDelete(null)
  }

  const handleDeleteAll = () => {
    if (confirm("Are you sure you want to delete all conversations? This cannot be undone.")) {
      deleteAllConversations()
      loadConversations()
      onNewConversation()
    }
  }

  return (
    <>
      <Sidebar>
        <SidebarHeader className="border-b border-border">
          <div className="flex items-center gap-2 px-2 py-2">
            <MessageSquare className="h-5 w-5 text-primary" />
            <h2 className="font-semibold text-lg">Chat History</h2>
          </div>
        </SidebarHeader>

        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Conversations</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {/* New Conversation Button */}
                <SidebarMenuItem>
                  <SidebarMenuButton
                    onClick={onNewConversation}
                    className="w-full justify-start gap-2"
                  >
                    <Plus className="h-4 w-4" />
                    <span>New Conversation</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>

                <SidebarSeparator />

                {/* Conversation List */}
                {conversations.length === 0 ? (
                  <div className="px-2 py-8 text-center text-sm text-muted-foreground">
                    No conversations yet.
                    <br />
                    Start a new chat to begin!
                  </div>
                ) : (
                  conversations.map((conv) => (
                    <SidebarMenuItem key={conv.id}>
                      <div className="group relative flex w-full items-center">
                        <SidebarMenuButton
                          onClick={() => onSelectConversation(conv.id)}
                          isActive={conv.id === currentConversationId}
                          className="flex-1 justify-start gap-2 truncate pr-8"
                          tooltip={conv.title}
                        >
                          <MessageSquare className="h-4 w-4 shrink-0" />
                          <span className="truncate">{conv.title}</span>
                        </SidebarMenuButton>
                        
                        {/* Delete button (show on hover) */}
                        <Button
                          variant="ghost"
                          size="icon"
                          className="absolute right-1 h-7 w-7 opacity-0 transition-opacity group-hover:opacity-100"
                          onClick={(e) => handleDelete(conv.id, e)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                      
                      {/* Timestamp */}
                      <div className="px-2 pb-1 text-xs text-muted-foreground">
                        {formatRelativeTime(conv.updatedAt)}
                      </div>
                    </SidebarMenuItem>
                  ))
                )}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>

        <SidebarFooter className="border-t border-border">
          <div className="px-2 py-2">
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start text-destructive hover:text-destructive"
              onClick={handleDeleteAll}
              disabled={conversations.length === 0}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete All
            </Button>
          </div>
        </SidebarFooter>
      </Sidebar>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Conversation?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete this conversation. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDelete} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}

