"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import ChatMessage from "./chat-message"
import ProductCard from "./product-card"
import ConversationSidebar from "./conversation-sidebar"
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { Send } from "lucide-react"
import {
  saveConversation,
  getConversation,
  createNewConversation,
  generateConversationTitle,
  type Conversation,
} from "@/lib/conversation-storage"

interface Message {
  id: string
  type: "user" | "assistant"
  content: string
  products?: Product[]
  timestamp: Date
}

interface Product {
  id: string
  name: string
  category: string
  price: number
  image: string
  description: string
  rating: number
  link?: string
  color?: string
  sex?: string
  material?: string
  theme?: string
}

export default function ChatInterface() {
  // ---------- CONVERSATION MANAGEMENT ----------
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(() => {
    // Try to get last active conversation from localStorage on mount
    if (typeof window !== "undefined") {
      const lastActive = localStorage.getItem("fashion_chat_last_active")
      return lastActive || null
    }
    return null
  })
  const [isInitialized, setIsInitialized] = useState(false)

  // ---------- EXISTING STATES ----------
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "0",
      type: "assistant",
      content: "Hi there! I’m your fashion assistant. What are you looking for?",
      timestamp: new Date(),
    },
  ])

  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const [pagesByMessage, setPagesByMessage] = useState<Record<string, number>>({})

  // ---------- NEW: GLOBAL CANDIDATE UNIVERSE ----------
  const [candidateUniverse, setCandidateUniverse] = useState<Product[]>([])

  // ---------- CONTEXT STATE ----------
  type Filters = {
    color?: string
    priceMax?: number
    priceMin?: number
    sex?: string
    category?: string
    material?: string
    theme?: string
    ratingMin?: number
    ratingMax?: number
    // NEW: Negative constraints
    excludeColor?: string | string[]
    excludeMaterial?: string | string[]
    excludeTheme?: string | string[]
  }

  // NEW: Conversation history for context reference resolution
  type ConversationHistoryItem = {
    query: string
    timestamp: Date
    extractedConstraints: {
      positive: {
        color?: string[]
        material?: string[]
        theme?: string[]
        category?: string
        sex?: string
      }
      negative: {
        exclude_color?: string[]
        exclude_material?: string[]
        exclude_theme?: string[]
      }
    }
    results?: {
      displayedAttributes: {
        colors: string[]
        materials: string[]
        themes: string[]
      }
    }
  }

  const [context, setContext] = useState<{ baseQuery: string | null; filters: Filters }>({
    baseQuery: null,
    filters: {},
  })

  // NEW: Track which attribute is being asked (for replacement logic)
  const [pendingAttributeQuestion, setPendingAttributeQuestion] = useState<"color" | "material" | "theme" | null>(null)

  // NEW: Conversation history state
  const [conversationHistory, setConversationHistory] = useState<ConversationHistoryItem[]>([])

  const scrollToBottom = (instant = false) => {
    // Use setTimeout to ensure DOM is updated
    setTimeout(() => {
      if (messagesContainerRef.current) {
        // Scroll the container directly to bottom
        messagesContainerRef.current.scrollTo({
          top: messagesContainerRef.current.scrollHeight,
          behavior: instant ? "auto" : "smooth"
        })
      } else {
        // Fallback to scrollIntoView
        messagesEndRef.current?.scrollIntoView({ 
          behavior: instant ? "auto" : "smooth",
          block: "end"
        })
      }
    }, instant ? 50 : 100)
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Initialize: Create new conversation only if no active conversation exists
  useEffect(() => {
    if (isInitialized) return
    
    // Check if we have a valid active conversation
    if (currentConversationId) {
      const conv = getConversation(currentConversationId)
      if (conv) {
        // Load existing conversation
        loadConversation(currentConversationId)
        setIsInitialized(true)
        return
      }
    }
    
    // No valid conversation found, create new one
    createNewConversationHandler()
    setIsInitialized(true)
  }, [isInitialized, currentConversationId])

  // Auto-save conversation after messages change
  useEffect(() => {
    if (!currentConversationId || !isInitialized) return
    
    // Debounce auto-save
    const timeoutId = setTimeout(() => {
      saveCurrentConversation()
    }, 1000) // Save 1 second after last change

    return () => clearTimeout(timeoutId)
  }, [messages, context, candidateUniverse, currentConversationId, isInitialized])

  // =====================================================
  //  CONVERSATION MANAGEMENT FUNCTIONS
  // =====================================================
  function createNewConversationHandler() {
    const newConv = createNewConversation()
    setCurrentConversationId(newConv.id)
    
    // Save last active conversation ID
    if (typeof window !== "undefined") {
      localStorage.setItem("fashion_chat_last_active", newConv.id)
    }
    
    setMessages([
      {
        id: "0",
        type: "assistant",
        content: "Hi there! I’m your fashion assistant. What are you looking for?",
        timestamp: new Date(),
      },
    ])
    setContext({ baseQuery: null, filters: {} })
    setCandidateUniverse([])
    setConversationHistory([])
    setPendingAttributeQuestion(null)
    setPagesByMessage({})
    saveConversation(newConv)
    
    // Scroll to bottom after creating new conversation
    scrollToBottom(true) // Use instant scroll for better UX
  }

  function loadConversation(conversationId: string) {
    const conv = getConversation(conversationId)
    if (!conv) {
      console.error("Conversation not found:", conversationId)
      return
    }

    setCurrentConversationId(conv.id)
    
    // Save last active conversation ID
    if (typeof window !== "undefined") {
      localStorage.setItem("fashion_chat_last_active", conv.id)
    }
    
    setMessages(conv.messages.length > 0 ? conv.messages : [
      {
        id: "0",
        type: "assistant",
        content: "Hi there! I’m your fashion assistant. What are you looking for?",
        timestamp: new Date(),
      },
    ])
    setContext(conv.context ? { 
      baseQuery: conv.context.baseQuery ?? null, 
      filters: conv.context.filters || {} 
    } : { baseQuery: null, filters: {} })
    setCandidateUniverse(conv.candidateUniverse || [])
    setConversationHistory([]) // Reset conversation history for reference resolution
    setPendingAttributeQuestion(null)
    setPagesByMessage({})
    
    // Scroll to bottom after loading conversation
    scrollToBottom(true) // Use instant scroll for better UX
  }

  function saveCurrentConversation() {
    if (!currentConversationId) return

    // Generate title from first user message if not set
    const firstUserMessage = messages.find(m => m.type === "user")
    const title = firstUserMessage 
      ? generateConversationTitle(firstUserMessage.content)
      : "New Conversation"

    const conversation: Conversation = {
      id: currentConversationId,
      title,
      createdAt: new Date(), // Will be preserved from original if exists
      updatedAt: new Date(),
      messages,
      context,
      candidateUniverse,
    }

    saveConversation(conversation)
  }

  // =====================================================
  //  API CALL
  // =====================================================
  const fetchRecommendations = async (
    query: string,
    slots?: any,
    k: number = 300,
    n: number = 300
  ): Promise<{ products: Product[]; assistantMessage?: string; intent?: string }> => {
    try {
      const payload: any = slots ? { q: query, k, n, slots } : { q: query, k, n }

      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const data = await res.json()
      const items = Array.isArray(data?.items) ? data.items : []

      const mapped: Product[] = items.map((it: any, i: number) => {
        const m = it?.meta ?? {}
        const name = String(m.Name ?? m.Title ?? "Product")
        const descParts = [m.Sex, m.Color, m.Material, m.Theme].filter(Boolean)
        const description =
          descParts.length > 0 ? String(descParts.join(" • ")) : String(m.Link ?? "")

        return {
          id: String(m.id ?? m.product_id ?? m.Link ?? m.Title ?? i + 1),
          name,
          category: String(m.Category ?? "Unknown"),
          price: Number(m.PriceNum ?? 0),
          image: String(m.Image ?? "/placeholder.jpg"),
          description,
          rating: Number(m.RatingNum ?? 4.5),
          link: m.Link ? String(m.Link) : undefined,
          color: m.Color ? String(m.Color) : undefined,
          sex: m.Sex ? String(m.Sex) : undefined,
          material: m.Material ? String(m.Material) : undefined,
          theme: m.Theme ? String(m.Theme) : undefined,
        }
      })

      return {
        products: mapped,
        assistantMessage:
          typeof data?.assistant_message === "string" ? data.assistant_message : undefined,
        intent: typeof data?.intent === "string" ? data.intent : undefined,
      }
    } catch {
      return { products: [], assistantMessage: undefined, intent: "unknown" }
    }
  }


  // =====================================================
  //  NEW: Reference Resolution
  // =====================================================
  function resolveReference(
    query: string,
    attribute: "color" | "material" | "theme",
    history: ConversationHistoryItem[]
  ): string[] {
    const lowerQuery = query.toLowerCase()
    
    // Pattern: "this [attribute]"
    if (lowerQuery.match(new RegExp(`this ${attribute}`, "i"))) {
      const lastQuery = history[history.length - 1]
      if (lastQuery) {
        return lastQuery.extractedConstraints.positive[attribute] || 
               lastQuery.results?.displayedAttributes[attribute === "color" ? "colors" : attribute === "material" ? "materials" : "themes"] || 
               []
      }
    }
    
    // Pattern: "that [attribute]"
    if (lowerQuery.match(new RegExp(`that ${attribute}`, "i"))) {
      const prevQuery = history[history.length - 2]
      if (prevQuery) {
        return prevQuery.extractedConstraints.positive[attribute] || 
               prevQuery.results?.displayedAttributes[attribute === "color" ? "colors" : attribute === "material" ? "materials" : "themes"] || 
               []
      }
    }
    
    // Pattern: "these [attributes]" (plural)
    const pluralAttr = attribute === "color" ? "colors" : attribute === "material" ? "materials" : "themes"
    if (lowerQuery.match(new RegExp(`these ${pluralAttr}?`, "i"))) {
      const lastQuery = history[history.length - 1]
      if (lastQuery) {
        return lastQuery.results?.displayedAttributes[pluralAttr] || []
      }
    }
    
    // Pattern: "the [attribute]"
    if (lowerQuery.match(new RegExp(`the ${attribute}`, "i"))) {
      const lastQuery = history[history.length - 1]
      if (lastQuery) {
        return lastQuery.extractedConstraints.positive[attribute] || 
               lastQuery.results?.displayedAttributes[attribute === "color" ? "colors" : attribute === "material" ? "materials" : "themes"] || 
               []
      }
    }
    
    return []
  }

  // =====================================================
  //  NEW: Detect Attribute Change Intent using LLM (Option B)
  // =====================================================
  async function detectAttributeChangeIntent(
    text: string,
    history: ConversationHistoryItem[]
  ): Promise<{
    isAttributeChange: boolean
    attributeType?: "color" | "material" | "theme"
    action?: "replace" | "exclude"
  }> {
    try {
      // Prepare conversation history for context
      const historyForApi = history.slice(-5).map(item => ({
        query: item.query,
        timestamp: item.timestamp.toISOString(),
      }))

      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          action: "classify_attribute_change",
          query: text,
          conversation_history: historyForApi,
        }),
      })

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }

      const data = await res.json()
      
      return {
        isAttributeChange: data?.is_attribute_change === true,
        attributeType: data?.attribute_type || undefined,
        action: data?.action || undefined,
      }
    } catch (error) {
      console.warn("[ATTRIBUTE_CHANGE] Failed to classify, returning default:", error)
      // Fallback: return no attribute change
      return {
        isAttributeChange: false,
        attributeType: undefined,
        action: undefined,
      }
    }
  }

  // =====================================================
  //  NEW: Generate Question Based on Negative Intent
  // =====================================================
  function generatePreferenceQuestion(attributeType: "color" | "material" | "theme"): string {
    const questions = {
      color: [
        "Which color do you prefer?",
        "Any specific color you want?",
        "What color are you looking for?",
        "Tell me the color you have in mind.",
      ],
      material: [
        "What material do you prefer?",
        "Any specific material you want?",
        "Which fabric are you looking for?",
        "Tell me the material you prefer.",
      ],
      theme: [
        "Which style do you like?",
        "Any style you want to focus on?",
        "What style are you aiming for?",
        "Tell me the style you prefer.",
      ],
    }
    
    const questionList = questions[attributeType]
    return questionList[Math.floor(Math.random() * questionList.length)]
  }

  // =====================================================
  //  CONSTRAINT PARSER (giữ nguyên)
  // =====================================================
  function parseConstraints(text: string): {
    base?: string
    filters: Filters
    newCategory?: string
  } {
    const t = text.toLowerCase()
    const filters: Filters = {}

    const colorMatch = t.match(/\b(black|white|red|blue|green|brown|grey|gray|yellow|pink|purple|beige|navy)\b/)
    if (colorMatch) filters.color = colorMatch[1]
    
    // Extract material (cotton, silk, leather, denim, wool, polyester, etc.)
    const materialPatterns = [
      /\b(cotton|silk|leather|denim|wool|polyester|linen|cashmere|nylon|spandex|viscose|rayon|chiffon|satin|velvet|fleece|canvas|suede|mesh|jersey|twill|corduroy)\b/i
    ]
    for (const pattern of materialPatterns) {
      const materialMatch = text.match(pattern)
      if (materialMatch) {
        filters.material = materialMatch[1].toLowerCase()
        break
      }
    }
    
    // Extract theme (casual, formal, sporty, elegant, vintage, modern, classic, etc.)
    const themePatterns = [
      /\b(casual|formal|sporty|elegant|vintage|modern|classic|bohemian|minimalist|streetwear|business|party|beach|winter|summer|spring|fall|autumn)\b/i
    ]
    for (const pattern of themePatterns) {
      const themeMatch = text.match(pattern)
      if (themeMatch) {
        filters.theme = themeMatch[1].toLowerCase()
        break
      }
    }

    const under = t.match(/under\s*\$?(\d+(?:\.\d+)?)/) || t.match(/less than\s*\$?(\d+(?:\.\d+)?)/)
    if (under) filters.priceMax = Number(under[1])

    const over = t.match(/over\s*\$?(\d+(?:\.\d+)?)/) || t.match(/more than\s*\$?(\d+(?:\.\d+)?)/)
    if (over) filters.priceMin = Number(over[1])

    const between = t.match(/(?:between|from|about)\s*\$?(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*\$?(\d+(?:\.\d+)?)/)
    if (between) {
      const a = Number(between[1])
      const b = Number(between[2])
      filters.priceMin = Math.min(a, b)
      filters.priceMax = Math.max(a, b)
    }

    const ratingOver = t.match(/rating\s*(?:over|>=?|at least)\s*(\d+(?:\.\d+)?)/) || t.match(/\b(\d+(?:\.\d+)?)\+\s*stars?\b/)
    if (ratingOver) filters.ratingMin = Number(ratingOver[1])

    const ratingBetween = t.match(/rating\s*(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)/)
    if (ratingBetween) {
      const a = Number(ratingBetween[1])
      const b = Number(ratingBetween[2])
      filters.ratingMin = Math.min(a, b)
      filters.ratingMax = Math.max(a, b)
    }

    if (/(for )?(men|male|boys)\b/.test(t)) filters.sex = "men"
    if (/(for )?(women|female|girls)\b/.test(t)) filters.sex = "women"

    const productTypePattern = /\b(jean|jeans|trouser|trousers|pant|pants|shirt|polo|dress|jacket|coat|sweater|hoodie|sneaker|sneakers|shoe|shoes|boot|boots|belt|belts|hat|hats|cap|caps|bag|bags|watch|watches|sunglass|sunglasses)\b/i
    const categoryMatch = text.match(productTypePattern)
    if (categoryMatch) {
      let cat = categoryMatch[1].toLowerCase()
      if (cat.endsWith("s") && cat !== "jeans" && cat !== "trousers") {
        cat = cat.slice(0, -1)
      }
      filters.category = cat
    }

    const newTopicKeywords = /\b(I want|show me|find|search|look for|I need|get me)\b/i.test(text)
    const hasProductKeywords =
      /\b(for men|for women|for|men|women|male|female)\b/i.test(text) ||
      /\b(under \$|over \$|price|color|material|rating)\b/i.test(text) ||
      categoryMatch !== null

    const newCategory =
      newTopicKeywords && hasProductKeywords ? "new_topic" : categoryMatch ? "new_topic" : undefined

    const base = newTopicKeywords ? text : undefined
    return { base, filters, newCategory }
  }


  // =====================================================
  //  CLIENT FILTER (giữ nguyên + thêm negative filtering)
  // =====================================================
  function applyClientFilters(items: Product[], filters: Filters): Product[] {
    return items.filter((p) => {
      // === EXISTING POSITIVE FILTERS (giữ nguyên) ===
      if (filters.category) {
        const need = filters.category
          .toLowerCase()
          .replace(/s$/, "")       // remove plural
          .replace(/[_\-]/g, " ")
          .trim()
      
        // Normalize product category
        const catRaw = (p.category || "").toLowerCase()
        const catNorm = catRaw
          .replace(/s$/, "")         // polos -> polo
          .replace(/[_\-]/g, " ")
          .trim()
      
        // Normalize name + desc
        const text = `${p.name} ${p.description}`.toLowerCase()
        const textNorm = text
          .replace(/[_\-]/g, " ")
          .replace(/s\b/g, "")       // jeans -> jean
          .trim()
      
        // Flexible matching: exact, substring, or word boundary
        const needWords = need.split(" ")
        const catWords = catNorm.split(" ")
        const needFirstWord = needWords.length > 0 ? needWords[0] : ""
        const catFirstWord = catWords.length > 0 ? catWords[0] : ""
        const match =
          catNorm === need ||
          catNorm.includes(need) ||
          need.includes(catNorm) ||
          textNorm.includes(need) ||
          // Thêm: match nếu từ đầu tiên của category match
          (needFirstWord && catFirstWord && catFirstWord === needFirstWord)
      
        if (!match) return false
      }
      if (filters.color) {
        const color = filters.color.toLowerCase()
        const combined = `${p.color ?? ""} ${p.description} ${p.name}`.toLowerCase()
        if (!combined.includes(color)) return false
      }
      if (typeof filters.priceMax === "number" && p.price > filters.priceMax) return false
      if (typeof filters.priceMin === "number" && p.price < filters.priceMin) return false
      if (typeof filters.ratingMin === "number" && p.rating < filters.ratingMin) return false
      if (typeof filters.ratingMax === "number" && p.rating > filters.ratingMax) return false

      if (filters.sex) {
        const target = filters.sex.toLowerCase()
        const sx = (p.sex || p.description || "").toLowerCase()
        const re = new RegExp(`\\b${target}\\b`)
        if (!re.test(sx)) return false
      }

      if (filters.material) {
        const m = filters.material.toLowerCase()
        const combined = `${p.material ?? ""} ${p.description} ${p.name}`.toLowerCase()
        if (!combined.includes(m)) return false
      }
      if (filters.theme) {
        const th = filters.theme.toLowerCase()
        const combined = `${p.theme ?? ""} ${p.description} ${p.name}`.toLowerCase()
        if (!combined.includes(th)) return false
      }
      
      // === NEW: NEGATIVE FILTERS (exclusion) ===
      
      // Exclude Color
      if (filters.excludeColor) {
        const excludeColors = Array.isArray(filters.excludeColor) 
          ? filters.excludeColor 
          : [filters.excludeColor]
        const productColor = (p.color ?? "").toLowerCase()
        const productText = `${p.color ?? ""} ${p.description} ${p.name}`.toLowerCase()
        if (excludeColors.some(exclude => {
          const excludeLower = exclude.toLowerCase()
          return productColor.includes(excludeLower) || 
                 excludeLower.includes(productColor) ||
                 productText.includes(excludeLower)
        })) {
          return false // Excluded!
        }
      }
      
      // Exclude Material
      if (filters.excludeMaterial) {
        const excludeMaterials = Array.isArray(filters.excludeMaterial) 
          ? filters.excludeMaterial 
          : [filters.excludeMaterial]
        const productMaterial = (p.material ?? "").toLowerCase()
        const productText = `${p.material ?? ""} ${p.description} ${p.name}`.toLowerCase()
        if (excludeMaterials.some(exclude => {
          const excludeLower = exclude.toLowerCase()
          return productMaterial.includes(excludeLower) || 
                 excludeLower.includes(productMaterial) ||
                 productText.includes(excludeLower)
        })) {
          return false // Excluded!
        }
      }
      
      // Exclude Theme
      if (filters.excludeTheme) {
        const excludeThemes = Array.isArray(filters.excludeTheme) 
          ? filters.excludeTheme 
          : [filters.excludeTheme]
        const productTheme = (p.theme ?? "").toLowerCase()
        const productText = `${p.theme ?? ""} ${p.description} ${p.name}`.toLowerCase()
        if (excludeThemes.some(exclude => {
          const excludeLower = exclude.toLowerCase()
          return productTheme.includes(excludeLower) || 
                 excludeLower.includes(productTheme) ||
                 productText.includes(excludeLower)
        })) {
          return false // Excluded!
        }
      }
      
      return true
    })
  }


  // =====================================================
  //  ASSISTANT MESSAGE GENERATOR (giữ nguyên)
  // =====================================================
  const generateAssistantResponse = (userMessage: string, products: Product[]): string => {
    if (!products || products.length === 0) {
      return "I couldn’t find matching products yet. Please describe the item, color, and price range you prefer."
    }

    const first = products[0]
    const total = products.length
    const priceText = first.price > 0 ? ` around $${first.price.toFixed(2)}` : ""

    const templates = [
      `I found ${total} products that match your request. A highlighted option is "${first.name}" (${first.category})${priceText}. Browse the list below and tell me if you want to narrow by color or price.`,
      `Based on "${userMessage}", I suggest ${total} products. A standout choice is "${first.name}", which fits the style you’re after. Check the options and tell me which you like.`,
      `I filtered ${total} matching products. "${first.name}" is a notable pick for both style and price${priceText}. See the list below and let me know if you want to refine by color or budget.`,
    ]

    return templates[Math.floor(Math.random() * templates.length)]
  }


  // =====================================================
  //  HANDLE SEND MESSAGE — MAIN LOGIC
  // =====================================================
  const handleSendMessage = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    const parsed = parseConstraints(input)
    const isNewTopic = parsed.newCategory === "new_topic"

    // NEW: Detect attribute change intent using LLM (Option B)
    const attributeChangeResult = await detectAttributeChangeIntent(input, conversationHistory)
    
    // If user wants to change an attribute, ask them what they prefer instead
    if (attributeChangeResult.isAttributeChange && attributeChangeResult.attributeType) {
      const question = generatePreferenceQuestion(attributeChangeResult.attributeType)
      
      // Save which attribute we're asking about
      setPendingAttributeQuestion(attributeChangeResult.attributeType)
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: question,
        timestamp: new Date(),
      }
      
      setMessages((prev) => [...prev, assistantMessage])
      setIsLoading(false)
      return // Don't proceed with search, just ask the question
    }
    
    // NEW: If user is answering a pending attribute question, replace that attribute
    // Check if parsed input contains the attribute we're asking about
    let shouldReplaceAttribute = false
    let attributeToReplace: "color" | "material" | "theme" | null = null
    
    if (pendingAttributeQuestion) {
      // Check if user's response contains the attribute we asked about
      // Also check if user mentions the attribute keyword (e.g., "pink color", "cotton material")
      const lowerInput = input.toLowerCase()
      const mentionsColor = lowerInput.includes("color") || lowerInput.includes("màu")
      const mentionsMaterial = lowerInput.includes("material") || lowerInput.includes("chất liệu")
      const mentionsTheme = lowerInput.includes("theme") || lowerInput.includes("phong cách") || lowerInput.includes("style")
      
      if (pendingAttributeQuestion === "color" && (parsed.filters.color || mentionsColor)) {
        shouldReplaceAttribute = true
        attributeToReplace = "color"
        // If user says "pink color" but parseConstraints didn't extract it, try to extract manually
        if (!parsed.filters.color && mentionsColor) {
          const colorMatch = lowerInput.match(/\b(black|white|red|blue|green|brown|grey|gray|yellow|pink|purple|beige|navy)\b/)
          if (colorMatch) {
            parsed.filters.color = colorMatch[1]
          }
        }
      } else if (pendingAttributeQuestion === "material" && (parsed.filters.material || mentionsMaterial)) {
        shouldReplaceAttribute = true
        attributeToReplace = "material"
        // If user says "cotton material" but parseConstraints didn't extract it, try to extract manually
        if (!parsed.filters.material && mentionsMaterial) {
          const materialMatch = lowerInput.match(/\b(cotton|silk|leather|denim|wool|polyester|linen|cashmere|nylon|spandex|viscose|rayon|chiffon|satin|velvet|fleece|canvas|suede|mesh|jersey|twill|corduroy)\b/i)
          if (materialMatch) {
            parsed.filters.material = materialMatch[1].toLowerCase()
          }
        }
      } else if (pendingAttributeQuestion === "theme" && (parsed.filters.theme || mentionsTheme)) {
        shouldReplaceAttribute = true
        attributeToReplace = "theme"
        // If user says "casual theme" but parseConstraints didn't extract it, try to extract manually
        if (!parsed.filters.theme && mentionsTheme) {
          const themeMatch = lowerInput.match(/\b(casual|formal|sporty|elegant|vintage|modern|classic|bohemian|minimalist|streetwear|business|party|beach|winter|summer|spring|fall|autumn)\b/i)
          if (themeMatch) {
            parsed.filters.theme = themeMatch[1].toLowerCase()
          }
        }
      }
      
      // Clear pending question after processing (even if we didn't find the attribute, to avoid stuck state)
      if (shouldReplaceAttribute || parsed.filters.color || parsed.filters.material || parsed.filters.theme) {
        setPendingAttributeQuestion(null)
      }
    }
    
    // Build extracted constraints for history (only positive constraints now)
    const extractedConstraints = {
      positive: {
        color: parsed.filters.color ? [parsed.filters.color] : undefined,
        material: parsed.filters.material ? [parsed.filters.material] : undefined,
        theme: parsed.filters.theme ? [parsed.filters.theme] : undefined,
        category: parsed.filters.category,
        sex: parsed.filters.sex,
      },
      negative: {},
    }

    // Update context
    // If replacing an attribute (after answering a question), replace instead of merge
    const nextContext = isNewTopic
      ? { 
          baseQuery: input, 
          filters: {
            ...parsed.filters,
          }
        }
      : shouldReplaceAttribute && attributeToReplace
      ? {
          // REPLACE MODE: Keep category and other filters, but replace the specific attribute
          baseQuery: context.baseQuery ?? input,
          filters: {
            // Keep all existing filters from context
            ...context.filters,
            // Replace the specific attribute with new value from parsed input
            ...(attributeToReplace === "color" ? { color: parsed.filters.color } : {}),
            ...(attributeToReplace === "material" ? { material: parsed.filters.material } : {}),
            ...(attributeToReplace === "theme" ? { theme: parsed.filters.theme } : {}),
            // Also allow adding new material/theme if user mentions them (even if not the one being asked)
            // But only if they're not being replaced
            ...(attributeToReplace !== "material" && parsed.filters.material ? { material: parsed.filters.material } : {}),
            ...(attributeToReplace !== "theme" && parsed.filters.theme ? { theme: parsed.filters.theme } : {}),
            // Keep category from context (important: don't lose it!)
            category: context.filters.category || parsed.filters.category,
            // Keep sex from context
            sex: context.filters.sex || parsed.filters.sex,
            // Keep price and rating from context
            priceMin: context.filters.priceMin ?? parsed.filters.priceMin,
            priceMax: context.filters.priceMax ?? parsed.filters.priceMax,
            ratingMin: context.filters.ratingMin ?? parsed.filters.ratingMin,
            ratingMax: context.filters.ratingMax ?? parsed.filters.ratingMax,
          },
        }
      : {
          // NORMAL MERGE MODE: Merge all filters
          baseQuery: context.baseQuery ?? input,
          filters: { 
            ...context.filters, 
            ...parsed.filters,
          },
        }

    setContext(nextContext)

    // Build effective query for ANN if needed
    const constraintParts: string[] = []
    if (nextContext.filters.category) constraintParts.push(`category ${nextContext.filters.category}`)
    if (nextContext.filters.color) constraintParts.push(`color ${nextContext.filters.color}`)
    if (typeof nextContext.filters.priceMax === "number")
      constraintParts.push(`price under ${nextContext.filters.priceMax}`)
    if (typeof nextContext.filters.priceMin === "number")
      constraintParts.push(`price over ${nextContext.filters.priceMin}`)
    if (nextContext.filters.sex) constraintParts.push(`for ${nextContext.filters.sex}`)
    const constraintPhrase = constraintParts.join(", ")

    const effectiveQuery = [nextContext.baseQuery, constraintPhrase].filter(Boolean).join(" ; ")

    // Build slots (include negative constraints)
    const slots = {
      category: nextContext.filters.category || undefined,
      sex: nextContext.filters.sex || undefined,
      color: nextContext.filters.color ? [nextContext.filters.color] : undefined,
      material: nextContext.filters.material ? [nextContext.filters.material] : undefined,
      theme: nextContext.filters.theme ? [nextContext.filters.theme] : undefined,
      price:
        typeof nextContext.filters.priceMin === "number" ||
        typeof nextContext.filters.priceMax === "number"
          ? { min: nextContext.filters.priceMin, max: nextContext.filters.priceMax }
          : undefined,
      rating:
        typeof nextContext.filters.ratingMin === "number" ||
        typeof nextContext.filters.ratingMax === "number"
          ? { min: nextContext.filters.ratingMin, max: nextContext.filters.ratingMax }
          : undefined,
    }

    // Dynamic N & K
    const numFilters = Object.values(nextContext.filters).filter((v) => v !== undefined).length
    // Nếu chỉ có category (ví dụ: "Polo for men") → cần lấy thật nhiều products để user lọc tiếp
    const isCategoryOnly =
      !!nextContext.filters.category &&
      !nextContext.filters.color &&
      typeof nextContext.filters.priceMax !== "number" &&
      typeof nextContext.filters.priceMin !== "number" &&
      !nextContext.filters.material &&
      !nextContext.filters.theme &&
      !nextContext.filters.ratingMin &&
      !nextContext.filters.ratingMax

    const serverN = (() => {
      if (isCategoryOnly) return 400
      if (numFilters <= 1) return 200
      if (numFilters === 0) return 250
      if (numFilters >= 3) return 100
      return 150
    })()

    const serverK = Math.max(200, Math.min(500, serverN + 100))

    // -------------------------------------------
    // NEW LOGIC — KEEP CANDIDATE UNIVERSE
    // -------------------------------------------
    let products: Product[] = []
    let apiAssistantMsg: string | undefined = undefined

    if (isNewTopic || candidateUniverse.length === 0) {
      // Call backend ANN
      const result = await fetchRecommendations(effectiveQuery, slots, serverK, serverN)
      apiAssistantMsg = result.assistantMessage

      // Save candidate universe
      setCandidateUniverse(result.products)

      // Apply local filters
      products = applyClientFilters(result.products, nextContext.filters)
    } else {
      // Only refine locally
      products = applyClientFilters(candidateUniverse, nextContext.filters)
    }

    // Assistant message
    let finalContent = ""
    if (apiAssistantMsg?.trim()) finalContent = apiAssistantMsg
    else if (products.length > 0) finalContent = generateAssistantResponse(input, products)
    else
      finalContent =
        "I couldn’t find matching products yet. Please describe the item, color, and price range you have in mind."

    // NEW: Extract displayed attributes from products for history
    const displayedAttributes = {
      colors: Array.from(new Set(products.map(p => p.color).filter(Boolean) as string[])),
      materials: Array.from(new Set(products.map(p => p.material).filter(Boolean) as string[])),
      themes: Array.from(new Set(products.map(p => p.theme).filter(Boolean) as string[])),
    }

    // NEW: Update conversation history
    const historyItem: ConversationHistoryItem = {
      query: input,
      timestamp: new Date(),
      extractedConstraints: extractedConstraints,
      results: {
        displayedAttributes: displayedAttributes,
      },
    }
    setConversationHistory((prev) => {
      const newHistory = [...prev, historyItem]
      // Keep only last 10 items to avoid memory issues
      return newHistory.slice(-10)
    })

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: "assistant",
      content: finalContent,
      products: products.length > 0 ? products : undefined,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, assistantMessage])
    setIsLoading(false)
  }

  // =====================================================
  // RENDER UI WITH SIDEBAR
  // =====================================================
  return (
    <SidebarProvider>
      <div className="flex h-full w-full">
        {/* Conversation Sidebar */}
        <ConversationSidebar
          currentConversationId={currentConversationId}
          onSelectConversation={(id) => {
            if (id) {
              loadConversation(id)
            }
          }}
          onNewConversation={createNewConversationHandler}
        />

        {/* Main Chat Interface */}
        <SidebarInset className="flex-1">
    <div className="flex h-full flex-col bg-gradient-to-br from-background via-background to-muted">
      {/* Header */}
      <div className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="mx-auto max-w-4xl px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-primary to-accent">
              <span className="text-sm font-bold text-primary-foreground">👗</span>
            </div>
            <div>
              <h1 className="text-lg font-semibold text-foreground">Fashion Advisor</h1>
              <p className="text-xs text-muted-foreground">Personal fashion advisor</p>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Container */}
            <div ref={messagesContainerRef} className="flex-1 overflow-y-auto flex flex-col">
              <div className="flex-1 flex flex-col justify-end min-h-full">
                <div className="mx-auto max-w-4xl w-full space-y-4 px-4 py-6">
          {messages.map((message) => (
            <div key={message.id}>
              <ChatMessage message={message} />
              {message.products && message.products.length > 0 && (
                <div className="mt-4">
                  {(() => {
                    const perPage = 9
                    const current = pagesByMessage[message.id] ?? 1
                    const totalPages = Math.max(1, Math.ceil(message.products!.length / perPage))
                    const safePage = Math.min(Math.max(current, 1), totalPages)
                    const start = (safePage - 1) * perPage
                    const end = start + perPage
                    const pageItems = message.products!.slice(start, end)

                    const goTo = (p: number) =>
                      setPagesByMessage((prev) => ({ ...prev, [message.id]: p }))
                    const prev = () => goTo(Math.max(1, safePage - 1))
                    const next = () => goTo(Math.min(totalPages, safePage + 1))

                    return (
                      <>
                        <div className="mb-3 flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">{`Showing ${
                            start + 1
                          }-${Math.min(end, message.products!.length)} of ${
                            message.products!.length
                          }`}</span>
                          <div className="flex items-center gap-2">
                            <Button size="sm" variant="outline" onClick={prev} disabled={safePage === 1}>
                              Previous
                            </Button>
                            <span className="text-xs">{safePage} / {totalPages}</span>
                            <Button size="sm" variant="outline" onClick={next} disabled={safePage === totalPages}>
                              Next
                            </Button>
                          </div>
                        </div>
                        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                          {pageItems.map((product) => (
                            <ProductCard key={product.id} product={product} />
                          ))}
                        </div>
                      </>
                    )
                  })()}
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="flex gap-2 rounded-2xl bg-card px-4 py-3">
                <div className="h-2 w-2 animate-bounce rounded-full bg-primary" />
                <div className="animation-delay-200 h-2 w-2 animate-bounce rounded-full bg-primary" />
                <div className="animation-delay-400 h-2 w-2 animate-bounce rounded-full bg-primary" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
                </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card/50 backdrop-blur-sm">
        <div className="mx-auto max-w-4xl px-4 py-4">
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
              placeholder="What are you looking for? (e.g., white shirt, jeans, sneakers...)"
              className="flex-1 rounded-full border-border bg-background/50 text-foreground placeholder:text-muted-foreground focus:bg-background"
              disabled={isLoading}
            />
            <Button
              onClick={handleSendMessage}
              disabled={isLoading || !input.trim()}
              className="rounded-full bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90"
              size="icon"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
}
