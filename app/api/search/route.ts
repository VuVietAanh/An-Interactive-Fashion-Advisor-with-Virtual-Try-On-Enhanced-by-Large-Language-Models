export const runtime = "nodejs";

type SearchReq = {
  q: string;
  k?: number;
  n?: number;
  slots?: unknown;
};

type Intent = "fashion" | "chitchat" | "off_topic" | "unknown";

const OPENAI_URL = "https://api.openai.com/v1/chat/completions";

function getApiKey(): string | null {
  return process.env.OPENAI_API_KEY || null;
}

// --- 1) Intent classification with Qwen (fallback to GPT) ---
async function classifyIntent(q: string): Promise<Intent> {
  // Try Qwen first (with 3s timeout for fast fallback)
  try {
    const baseUrl = process.env.RETRIEVAL_API_URL ?? "http://127.0.0.1:8000";
    const url = `${baseUrl.replace(/\/$/, "")}/qwen/intent`;
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(),8000); // 8s timeout (increased for Qwen)
    
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: q }),
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = (await res.json()) as any;
      const intent = data?.intent;
      if (intent === "fashion" || intent === "chitchat" || intent === "off_topic") {
        return intent as Intent;
      }
    }
  } catch (error) {
    console.warn("[INTENT] Qwen failed or timeout, falling back to GPT:", error);
  }

  // Fallback to GPT if Qwen fails or not available
  const apiKey = getApiKey();
  if (!apiKey) return "unknown";

  try {
    const res = await fetch(OPENAI_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content:
              "You are the intent classifier for a FASHION SHOPPING ASSISTANT. Classify the user message into exactly one of: " +
              "fashion (searching for / asking about clothing, shoes, or fashion accessories), " +
              "chitchat (small talk, greetings, casual chat without product intent), " +
              "off_topic (anything not about fashion, e.g., phones, laptops, food, politics). " +
              "Reply with only one token in {fashion, chitchat, off_topic}.",
          },
          { role: "user", content: q },
        ],
        max_tokens: 4,
        temperature: 0,
      }),
    });

    if (!res.ok) return "unknown";
    const data = (await res.json()) as any;
    const content: string | undefined =
      data?.choices?.[0]?.message?.content?.trim().toLowerCase();

    if (content === "fashion") return "fashion";
    if (content === "chitchat") return "chitchat";
    if (content === "off_topic" || content === "offtopic") return "off_topic";
    return "unknown";
  } catch {
    return "unknown";
  }
}

// --- 2) Classify attribute change intent with Qwen (fallback to GPT) ---
type AttributeChangeResult = {
  is_attribute_change: boolean;
  attribute_type: "color" | "material" | "theme" | null;
  action: "replace" | "exclude" | null;
};

async function classifyAttributeChange(
  q: string,
  conversationHistory?: Array<{ query: string; timestamp: string }>
): Promise<AttributeChangeResult> {
  // Try Qwen first (with 3s timeout)
  try {
    const baseUrl = process.env.RETRIEVAL_API_URL ?? "http://127.0.0.1:8000";
    const url = `${baseUrl.replace(/\/$/, "")}/qwen/attribute-change`;
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000); // 8s timeout (increased for Qwen)
    
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ 
        query: q,
        conversation_history: conversationHistory || []
      }),
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = (await res.json()) as any;
      return {
        is_attribute_change: data?.is_attribute_change === true,
        attribute_type: data?.attribute_type || null,
        action: data?.action || null,
      };
    }
  } catch (error) {
    console.warn("[ATTRIBUTE_CHANGE] Qwen failed or timeout, falling back to GPT:", error);
  }

  // Fallback to GPT if Qwen fails or not available
  const apiKey = getApiKey();
  if (!apiKey) {
    return { is_attribute_change: false, attribute_type: null, action: null };
  }

  try {
    const historyContext = conversationHistory && conversationHistory.length > 0
      ? `\n\nRecent conversation:\n${conversationHistory.slice(-3).map(h => `- ${h.query}`).join("\n")}`
      : "";

    const res = await fetch(OPENAI_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content:
              "You are the attribute-change classifier for a FASHION ASSISTANT.\n" +
              "Decide if the user wants to CHANGE an attribute (color, material, theme).\n\n" +
              "Detect these cases:\n" +
              "1) User WANTS TO CHANGE: 'I want to change material', 'Can I switch color?', 'I want a different theme'\n" +
              "2) User DISLIKES an attribute: 'I dislike this color', 'I hate this material', 'Don't like this theme'\n" +
              "3) User is NOT changing (normal search): 'Show me pink hats', 'I want black shirts'\n\n" +
              "Return JSON only: {\"is_attribute_change\": true/false, \"attribute_type\": \"color\"|\"material\"|\"theme\"|null, \"action\": \"replace\"|\"exclude\"|null}\n" +
              "Return ONLY JSON. No extra text.",
          },
          { 
            role: "user", 
            content: `User message: "${q}"${historyContext}` 
          },
        ],
        max_tokens: 50,
        temperature: 0,
        response_format: { type: "json_object" },
      }),
    });

    if (!res.ok) {
      return { is_attribute_change: false, attribute_type: null, action: null };
    }
    
    const data = (await res.json()) as any;
    const content: string | undefined = data?.choices?.[0]?.message?.content?.trim();
    
    if (content) {
      try {
        const parsed = JSON.parse(content);
        return {
          is_attribute_change: parsed.is_attribute_change === true,
          attribute_type: parsed.attribute_type || null,
          action: parsed.action || null,
        };
      } catch {
        // If JSON parse fails, return default
      }
    }
  } catch (error) {
    console.warn("[ATTRIBUTE_CHANGE] GPT fallback failed:", error);
  }

  return { is_attribute_change: false, attribute_type: null, action: null };
}

// --- 3) Generate assistant reply with Qwen (fallback to GPT) ---
async function generateAssistantMessage(
  q: string,
  intent: Intent,
  hasProducts: boolean
): Promise<string | undefined> {
  // Try Qwen first (with 10s timeout for generation - longer than intent)
  try {
    const baseUrl = process.env.RETRIEVAL_API_URL ?? "http://127.0.0.1:8000";
    const url = `${baseUrl.replace(/\/$/, "")}/qwen/generate`;
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 20000); // 20s timeout (increased for Qwen generation)
    
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        query: q,
        intent: intent,
        has_products: hasProducts,
      }),
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = (await res.json()) as any;
      const message = data?.message;
      if (message && typeof message === "string" && message.trim()) {
        return message.trim();
      }
    }
  } catch (error) {
    console.warn("[GENERATE] Qwen failed or timeout, falling back to GPT:", error);
  }

  // Fallback to GPT if Qwen fails or not available
  const apiKey = getApiKey();
  if (!apiKey) {
    // Fallback rule-based if no API key
    if (intent === "chitchat") {
      return "Hi there! I’m your fashion assistant. What clothing, shoes, or accessories are you looking for?";
    }
    if (intent === "off_topic") {
      return "I can help with fashion items only (clothing, shoes, accessories). Tell me what you’d like to find.";
    }
    if (!hasProducts && intent === "fashion") {
      return "I couldn’t find matching products yet. Could you describe the item, color, and price range you prefer?";
    }
    return undefined;
  }

  try {
    const res = await fetch(OPENAI_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content:
              "You are a FASHION SHOPPING ASSISTANT who responds in English.\n" +
              "IMPORTANT: Only discuss products that exist in the system database.\n" +
              "NEVER mention store names, brands, or any info not in the database (e.g., H&M, Zara, Uniqlo, or any other store/brand).\n" +
              "If no products are found, say that nothing was found and ask the user to restate their request (type of item, color, price range). DO NOT suggest where to buy or any brand.\n" +
              "Only talk about clothing, shoes, and fashion accessories that could appear in the product metadata (jeans, trousers, shirts, dresses, jackets, coats, hoodies, sweaters, shoes, sneakers, boots, hats, caps, bags, belts, watches, sunglasses, etc.).\n" +
              "If the user asks about anything outside that (e.g., phones, laptops, food, politics), politely refuse and redirect them back to fashion products.\n" +
              "Always respond concisely and naturally, based only on products in the system.",
          },
          {
            role: "user",
            content: `Intent: ${intent}\nProducts found? ${
              hasProducts ? "Yes" : "No"
            }\nUser message: "${q}"`,
          },
        ],
        max_tokens: 120,
        temperature: 0.5,
      }),
    });

    if (!res.ok) return undefined;
    const data = (await res.json()) as any;
    const content: string | undefined =
      data?.choices?.[0]?.message?.content?.trim();
    return content || undefined;
  } catch {
    return undefined;
  }
}

// --- 3) Handler chính /api/search ---
export async function POST(req: Request) {
  const body = await req.json();
  
  // Handle attribute change classification request
  if (body.action === "classify_attribute_change") {
    const { query, conversation_history } = body;
    if (!query || typeof query !== "string") {
      return Response.json(
        { error: "Missing or invalid query" },
        { status: 400 }
      );
    }
    
    const result = await classifyAttributeChange(query, conversation_history);
    return Response.json(result);
  }
  
  // Original search request handling
  const payload: SearchReq = body;

  if (!payload.q || typeof payload.q !== "string") {
    return new Response(JSON.stringify({ error: "q is required" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }

  const q = payload.q;

  // Step 1: classify intent (if no API key → 'unknown')
  const intent = await classifyIntent(q);

  // If chitchat or off_topic → do NOT call retrieval, just respond
  if (intent === "chitchat" || intent === "off_topic") {
    const assistantMsg =
      (await generateAssistantMessage(q, intent, false)) ??
      (intent === "chitchat"
        ? "Hi there! I’m your fashion assistant. What clothing, shoes, or accessories are you looking for?"
        : "I can help with fashion items only. Tell me what kind of item you want to find.");

    return new Response(
      JSON.stringify({
        intent,
        items: [],
        assistant_message: assistantMsg,
      }),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      }
    );
  }

  // Step 2: intent is fashion or unknown → call retrieval service (Python)
  const baseUrl = process.env.RETRIEVAL_API_URL ?? "http://127.0.0.1:8000";
  const url = `${baseUrl.replace(/\/$/, "")}/search`;

  const upstream = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      q,
      k: payload.k ?? 5,
      n: payload.n,
      slots: payload.slots,
    }),
  });

  const status = upstream.status;
  let upstreamJson: any = null;
  try {
    upstreamJson = await upstream.json();
  } catch {
    upstreamJson = null;
  }

  const items = Array.isArray(upstreamJson?.items) ? upstreamJson.items : [];
  let assistantMsg: string | undefined = undefined;

  // If intent is fashion but no products → ask OpenAI to craft a response
  if ((intent === "fashion" || intent === "unknown") && items.length === 0) {
    const generatedMsg = await generateAssistantMessage(q, "fashion", false);
    
    // Validate: ensure message does not contain store/brand names
    const forbiddenBrands = ["h&m", "zara", "uniqlo", "nike", "adidas", "gucci", "prada", "lv", "louis vuitton", "chanel", "dior", "versace", "puma", "reebok", "converse", "vans", "gap", "old navy", "forever 21", "hollister", "abercrombie", "calvin klein", "tommy hilfiger", "ralph lauren", "levi's", "levis"];
    const hasForbiddenBrand = generatedMsg && forbiddenBrands.some(brand => 
      generatedMsg.toLowerCase().includes(brand.toLowerCase())
    );
    
    // If message contains a brand → use safe fallback
    if (hasForbiddenBrand) {
      assistantMsg = "I couldn’t find matching products yet. Please describe the item, color, and price range you prefer.";
    } else {
      assistantMsg = generatedMsg ?? "I couldn’t find matching products yet. Please describe the item, color, and price range you prefer.";
    }
  }

  const bodyOut = {
    ...(upstreamJson || {}),
    items,
    intent,
    assistant_message: assistantMsg,
  };

  return new Response(JSON.stringify(bodyOut), {
    status,
    headers: { "content-type": "application/json" },
  });
}
