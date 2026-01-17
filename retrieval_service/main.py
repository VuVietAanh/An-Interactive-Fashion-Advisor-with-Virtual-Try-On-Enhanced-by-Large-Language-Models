


import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix OpenMP conflict on Windows (must be before importing torch/faiss)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# =======================================================================
# Logging
# =======================================================================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retrieval")

# =======================================================================
# Pydantic models
# =======================================================================

class PriceRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class RatingRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class Slots(BaseModel):
    category: Optional[str] = None
    sex: Optional[str] = None
    color: Optional[List[str]] = None
    material: Optional[List[str]] = None
    theme: Optional[List[str]] = None
    price: Optional[PriceRange] = None
    rating: Optional[RatingRange] = None
    # NEW: Negative constraints (exclusion)
    exclude_color: Optional[List[str]] = None
    exclude_material: Optional[List[str]] = None
    exclude_theme: Optional[List[str]] = None


class SearchRequest(BaseModel):
    q: Optional[str] = None
    k: int | None = 5
    n: int | None = None
    slots: Optional[Slots] = None


class SearchResponseItem(BaseModel):
    score: float
    meta: Dict[str, Any]


class SearchResponse(BaseModel):
    items: List[SearchResponseItem]


# =======================================================================
# Qwen LLM Models
# =======================================================================

class QwenIntentRequest(BaseModel):
    query: str


class QwenIntentResponse(BaseModel):
    intent: str  # "fashion" | "chitchat" | "off_topic" | "unknown"


class QwenGenerateRequest(BaseModel):
    query: str
    intent: str
    has_products: bool


class QwenGenerateResponse(BaseModel):
    message: str


class QwenAttributeChangeRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, Any]]] = None  # Optional context


class QwenAttributeChangeResponse(BaseModel):
    is_attribute_change: bool  # True if user wants to change an attribute
    attribute_type: Optional[str] = None  # "color" | "material" | "theme" | null
    action: Optional[str] = None  # "replace" | "exclude" | null


# =======================================================================
# Helpers
# =======================================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _index_dir() -> Path:
    return _project_root() / "index_retrival"


def _get_adapter_path() -> Path:
    adapter_path = _project_root() / "epoch_2_final"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing epoch_2_final at {adapter_path}")
    return adapter_path


def _safe_float(x: Any) -> float:
    """Convert to safe float for JSON (no NaN/inf)."""
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return 9999.0
        return v
    except Exception:
        return 9999.0


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vectors row-wise."""
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def _normalize_text(x: Any) -> str:
    """Normalize string for category/sex matching."""
    if not x:
        return ""
    s = str(x).strip().lower()
    # remove trailing s (polo / polos / jeans / jean)
    if len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    return s


# =======================================================================
# Model loading
# =======================================================================

def _load_metadata() -> List[Dict[str, Any]]:
    meta_path = _index_dir() / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.pkl at {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, list):
        raise ValueError("meta.pkl must be a list")
    return meta


def _load_faiss_index() -> faiss.Index:
    idx_path = _index_dir() / "products.faiss"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing products.faiss at {idx_path}")
    return faiss.read_index(str(idx_path))


def _load_embedding_model() -> Any:
    """Load embedding model (intfloat/e5-large-v2)."""
    from sentence_transformers import SentenceTransformer
    
    model_name = "intfloat/e5-large-v2"
    logger.info(f"[MODEL] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"[MODEL] Model loaded successfully! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def _load_qwen_llm() -> Tuple[Any, Any]:
    """Load Qwen 3B for LLM tasks (intent classification, response generation)."""
    adapter_path = _get_adapter_path()
    
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")
    
    logger.info(f"[QWEN] Loading base model: {base}")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    
    # Determine dtype and device settings
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    
    logger.info(f"[QWEN] Loading model with dtype={dtype}, use_cuda={use_cuda}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base,
            trust_remote_code=True,
            dtype=dtype,  # Use dtype instead of deprecated torch_dtype
            device_map="auto" if use_cuda else None,
            low_cpu_mem_usage=True,  # Reduce memory usage on CPU
        )
    except Exception as e:
        logger.error(f"[QWEN] Error loading model: {e}")
        # Fallback: try without device_map on CPU
        if not use_cuda:
            logger.info("[QWEN] Retrying without device_map...")
            model = AutoModelForCausalLM.from_pretrained(
                base,
                trust_remote_code=True,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
        else:
            raise
    
    logger.info(f"[QWEN] Attaching LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    
    logger.info(f"[QWEN] Qwen model loaded successfully!")
    return model, tokenizer


# =======================================================================
# Encoder
# =======================================================================

def _encode_text(model: Any, texts: List[str],
                 normalize: bool = True) -> np.ndarray:
    """Encode text using SentenceTransformer model."""
    # SentenceTransformer.encode() returns numpy array directly
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False
    )
    
    # Ensure float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    return embeddings


# =======================================================================
# FastAPI App
# =======================================================================

app = FastAPI(title="Retrieval Service", version="1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Any | None = None
_index: faiss.Index | None = None
_meta: List[Dict[str, Any]] | None = None
_qwen_model: Any | None = None
_qwen_tokenizer: Any | None = None


# =======================================================================
# Startup (lazy load)
# =======================================================================

@app.on_event("startup")
def _startup():
    global _model, _index, _meta, _qwen_model, _qwen_tokenizer
    logger.info("[STARTUP] Loading FAISS + metadata + embedding model...")
    _model = _load_embedding_model()
    _index = _load_faiss_index()
    _meta = _load_metadata()

    logger.info("[STARTUP] Loading Qwen LLM model...")
    try:
        import gc
        gc.collect()  # Clean up memory before loading large model
        _qwen_model, _qwen_tokenizer = _load_qwen_llm()
        logger.info("[STARTUP] Qwen LLM loaded successfully!")
    except Exception as e:
        import traceback
        logger.error(f"[STARTUP] Failed to load Qwen LLM: {e}")
        logger.error(traceback.format_exc())
        logger.warning("[STARTUP] LLM features will be unavailable.")
        _qwen_model = None
        _qwen_tokenizer = None
    
    logger.info("[STARTUP] Service Ready")


# =======================================================================
# SEARCH API
# =======================================================================

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:

    if not _model or not _index or _meta is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # == Pure text search (no slots) =======================================
    if req.slots is None:
        if not req.q:
            raise HTTPException(status_code=400,
                                detail="q is required if no slots provided")
        qtext = req.q.strip()
        emb = _encode_text(_model, [qtext])
        # Ensure embedding has correct shape: (1, embedding_dim)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim == 2 and emb.shape[0] != 1:
            emb = emb.reshape(1, -1)
        # Ensure float32 and correct shape
        emb = emb.astype("float32")
        # Validate dimension matches index
        if emb.shape[1] != _index.d:
            raise ValueError(
                f"Embedding dimension mismatch: embedding has {emb.shape[1]} dims, "
                f"but index expects {_index.d} dims"
            )
        D, I = _index.search(emb, req.k or 5)

        items = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0 or idx >= len(_meta):
                continue
            meta = _meta[idx]
            items.append(SearchResponseItem(
                score=_safe_float(dist),
                meta=meta
            ))
        return SearchResponse(items=items[:req.n or 12])

    # ===================================================================
    # SLOTS SEARCH (category/sex/color/price...)
    # ===================================================================

    sl = req.slots
    want_cat = _normalize_text(sl.category) if sl.category else ""
    want_sex = _normalize_text(sl.sex) if sl.sex else ""

    want_colors = [_normalize_text(c) for c in (sl.color or []) if c]
    want_mats = [_normalize_text(m) for m in (sl.material or []) if m]
    want_themes = [_normalize_text(t) for t in (sl.theme or []) if t]
    # NEW: Negative constraints (exclusion)
    exclude_colors = [_normalize_text(c) for c in (sl.exclude_color or []) if c]
    exclude_mats = [_normalize_text(m) for m in (sl.exclude_material or []) if m]
    exclude_themes = [_normalize_text(t) for t in (sl.exclude_theme or []) if t]

    pmin = sl.price.min if (sl.price and sl.price.min is not None) else None
    pmax = sl.price.max if (sl.price and sl.price.max is not None) else None

    # ---- Build embedding query -----------------------------------------
    parts = []
    if want_sex:
        parts.append(want_sex)
    if want_colors:
        parts.append(" ".join(want_colors))
    if want_cat:
        parts.append(want_cat)

    query_text = " ".join(parts) or (req.q or "clothing")
    
    # CRITICAL FIX: Nếu query quá ngắn (chỉ có category), thêm context từ original query
    if len(query_text.split()) < 2 and req.q:
        query_text = req.q.strip()
        logger.info(f"[SEARCH] Query too short, using original query: '{query_text}'")
    
    logger.info(f"[SEARCH] Query text = '{query_text}' | Slots: cat={want_cat}, sex={want_sex}")

    # ---- Filtering function (define before use) ------------------------
    def pass_filters(idx: int) -> bool:
        m = _meta[idx]

        # === EXISTING POSITIVE FILTERS (giữ nguyên) ===
        # sex - flexible matching (men/male/man, women/female/woman)
        if want_sex:
            product_sex = _normalize_text(m.get("Sex", ""))
            # Map common variations
            sex_variations = {
                "men": ["men", "male", "man"],
                "women": ["women", "female", "woman", "womens"],
                "man": ["men", "male", "man"],
                "woman": ["women", "female", "woman", "womens"],
            }
            want_sex_variations = sex_variations.get(want_sex, [want_sex])
            if product_sex not in want_sex_variations:
                return False

        # category - improved matching: check both directions
        if want_cat:
            cat = _normalize_text(m.get("Category"))
            # Check if want_cat is in cat OR cat is in want_cat (for partial matches)
            # Examples:
            # - want_cat="polo", cat="polo shirt" → "polo" in "polo shirt" ✓
            # - want_cat="polo", cat="polo" → cat == want_cat ✓
            # - want_cat="shirt", cat="polo shirt" → "shirt" in "polo shirt" ✓
            cat_match = (cat == want_cat) or (want_cat in cat) or (cat in want_cat)
            if not cat_match:
                return False

        # color
        if want_colors:
            cpool = " ".join(_normalize_text(x) for x in str(m.get("Color", "")).split("|"))
            for need in want_colors:
                if need not in cpool:
                    return False

        # price
        price = m.get("PriceNum")
        if price is not None:
            if pmin is not None and price < pmin:
                return False
            if pmax is not None and price > pmax:
                return False

        # === NEW: NEGATIVE FILTERS (exclusion) ===
        
        # Exclude Color
        if exclude_colors:
            product_color = _normalize_text(m.get("Color", ""))
            product_color_pool = " ".join(_normalize_text(x) for x in str(m.get("Color", "")).split("|"))
            for exclude_color in exclude_colors:
                exclude_color_norm = _normalize_text(exclude_color)
                if exclude_color_norm in product_color_pool or product_color_pool in exclude_color_norm:
                    return False  # Excluded!
        
        # Exclude Material
        if exclude_mats:
            product_material = _normalize_text(m.get("Material", ""))
            for exclude_mat in exclude_mats:
                exclude_mat_norm = _normalize_text(exclude_mat)
                if exclude_mat_norm in product_material or product_material in exclude_mat_norm:
                    return False  # Excluded!
        
        # Exclude Theme
        if exclude_themes:
            product_theme = _normalize_text(m.get("Theme", ""))
            for exclude_theme in exclude_themes:
                exclude_theme_norm = _normalize_text(exclude_theme)
                if exclude_theme_norm in product_theme or product_theme in exclude_theme_norm:
                    return False  # Excluded!

        # All filters passed
        return True

    # ---- CRITICAL: Nếu chỉ có category (có thể có sex) → dùng direct search ngay
    # Vì ANN với query ngắn "polo" hoặc "women polo" có thể không tìm được đủ sản phẩm
    is_simple_category_search = (
        want_cat and 
        not want_colors and 
        not want_mats and 
        not want_themes and 
        pmin is None and 
        pmax is None
    )
    # Bao gồm cả trường hợp: chỉ category, hoặc category + sex
    
    if is_simple_category_search:
        logger.info(f"[SEARCH] Simple category search (category='{want_cat}', sex='{want_sex or 'any'}') → using direct search for full coverage")
        # Debug: Check first few products to see why they're filtered out
        if len(_meta) > 0:
            sample_size = min(10, len(_meta))
            logger.debug(f"[SEARCH] Debug: Checking first {sample_size} products...")
            for i in range(sample_size):
                m = _meta[i]
                cat_raw = m.get("Category", "")
                cat_norm = _normalize_text(cat_raw)
                sex_raw = m.get("Sex", "")
                sex_norm = _normalize_text(sex_raw)
                passes = pass_filters(i)
                logger.debug(f"[SEARCH] Product {i}: cat='{cat_raw}' (norm: '{cat_norm}'), sex='{sex_raw}' (norm: '{sex_norm}'), passes={passes}")
        kept = [i for i in range(len(_meta)) if pass_filters(i)]
        logger.info(f"[SEARCH] Direct search found {len(kept)} products")
        # Sort by index để giữ thứ tự ổn định (hoặc có thể sort theo price nếu muốn)
        kept_with_dist = [(idx, 0.0) for idx in kept]  # Distance = 0 vì không dùng ANN
    else:
        # ---- ANN search -----------------------------------------------------
        base_k = req.k or 200
        if want_cat:
            # Có category + filters khác → tăng k để coverage tốt hơn
            k = min(base_k * 2, 500)
            logger.info(f"[SEARCH] Category with filters, using k={k}")
        else:
            k = base_k
        
        emb = _encode_text(_model, [query_text])
        # Ensure embedding has correct shape: (1, embedding_dim)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim == 2 and emb.shape[0] != 1:
            emb = emb.reshape(1, -1)
        # Ensure float32 and correct shape
        emb = emb.astype("float32")
        # Validate dimension matches index
        if emb.shape[1] != _index.d:
            raise ValueError(
                f"Embedding dimension mismatch: embedding has {emb.shape[1]} dims, "
                f"but index expects {_index.d} dims"
            )
        logger.debug(f"[SEARCH] Embedding shape: {emb.shape}, index dimension: {_index.d}")
        D, I = _index.search(emb, k)

        cand = list(I[0])
        cand_dist = list(D[0])

        logger.info(f"[SEARCH] ANN got {len(cand)} candidates")

        kept = [i for i in cand if i >= 0 and i < len(_meta) and pass_filters(i)]
        logger.info(f"[SEARCH] After filtering: {len(kept)}/{len(cand)} candidates passed filters")

        # ---- Fallback: direct search if ANN results too few ------------------------
        # Nếu có category và kết quả quá ít (< 20% số lượng expected), thử direct search
        if want_cat:
            # Ước tính: nếu chỉ có category, có thể có 200-400 sản phẩm
            # Nếu có category + sex, có thể có 100-200 sản phẩm
            expected_min = 50 if want_sex else 100
            if len(kept) < expected_min:
                logger.warning(f"[SEARCH] Only {len(kept)} results from ANN (expected at least {expected_min}), trying direct search...")
                direct_kept = [i for i in range(len(_meta)) if pass_filters(i)]
                if len(direct_kept) > len(kept):
                    logger.info(f"[SEARCH] Direct search found {len(direct_kept)} results (vs {len(kept)} from ANN) → using direct search")
                    kept = direct_kept
        elif not kept:
            logger.warning("[SEARCH] ANN returned 0 after filters → fallback direct-search")
            kept = [i for i in range(len(_meta)) if pass_filters(i)]
        
        # ---- Sort by ANN distance ------------------------------------------
        kept_with_dist = []
        for idx in kept:
            try:
                pos = cand.index(idx)
                dist = cand_dist[pos]
            except Exception:
                dist = 9999.0
            kept_with_dist.append((idx, _safe_float(dist)))

        kept_with_dist.sort(key=lambda x: x[1])
        
    # ---- Build response -------------------------------------------------
    n = req.n or 12
    items = [
        SearchResponseItem(
            score=_safe_float(dist),
            meta=_meta[idx]
        )
        for idx, dist in kept_with_dist[:n]
    ]

    return SearchResponse(items=items)


# =======================================================================
# Qwen LLM Endpoints
# =======================================================================

def _qwen_classify_intent(query: str) -> str:
    """Classify intent using Qwen model."""
    if not _qwen_model or not _qwen_tokenizer:
        return "unknown"
    
    try:
        prompt = f"""<|im_start|>system
You are the intent classifier for a FASHION SHOPPING ASSISTANT.
Classify the user message into exactly one of:
- fashion (searching for / asking about clothing, shoes, or fashion accessories)
- chitchat (small talk, greetings, casual chat without product intent)
- off_topic (anything not about fashion, e.g., phones, laptops, food, politics)

Reply with exactly one token: fashion, chitchat, or off_topic.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
        
        inputs = _qwen_tokenizer(prompt, return_tensors="pt")
        device = next(_qwen_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _qwen_model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
            )
        
        response = _qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        intent = response.split("assistant")[-1].strip().lower()
        
        if "fashion" in intent:
            return "fashion"
        elif "chitchat" in intent:
            return "chitchat"
        elif "off_topic" in intent or "offtopic" in intent:
            return "off_topic"
        return "unknown"
    except Exception as e:
        logger.error(f"[QWEN] Error classifying intent: {e}")
        return "unknown"


def _qwen_generate_response(query: str, intent: str, has_products: bool) -> str:
    """Generate assistant message using Qwen model."""
    if not _qwen_model or not _qwen_tokenizer:
        return ""
    
    try:
        prompt = f"""<|im_start|>system
You are a FASHION SHOPPING ASSISTANT who responds in English.
IMPORTANT: Only talk about products that exist in the system database.
NEVER mention store names, brands, or any information not in the database (e.g., H&M, Zara, Uniqlo, or any other store/brand).
If no products are found, say nothing was found and ask the user to restate their request (item type, color, price range). DO NOT suggest where to buy or any brand.
Only talk about clothing, shoes, and fashion accessories that could appear in the product metadata (jeans, trousers, shirts, dresses, jackets, coats, hoodies, sweaters, shoes, sneakers, boots, hats, caps, bags, belts, watches, sunglasses, etc.).
If the user asks about anything outside that (e.g., phones, laptops, food, politics), politely refuse and redirect them back to fashion products.
Always respond concisely and naturally, based only on products in the system.
<|im_end|>
<|im_start|>user
Intent: {intent}
Products found? {"Yes" if has_products else "No"}
User message: "{query}"
<|im_end|>
<|im_start|>assistant
"""
        
        inputs = _qwen_tokenizer(prompt, return_tensors="pt")
        device = next(_qwen_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Optimize for speed: use greedy decoding (faster than sampling on CPU)
            # If GPU available, can use sampling for more variety
            use_sampling = torch.cuda.is_available()
            gen_kwargs = {
                **inputs,
                "max_new_tokens": 80,  # Reduced for faster generation
                "do_sample": use_sampling,
            }
            if use_sampling:
                gen_kwargs.update({
                    "temperature": 0.5,
                    "top_p": 0.9,
                })
            
            outputs = _qwen_model.generate(**gen_kwargs)
        
        response = _qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        message = response.split("assistant")[-1].strip()
        return message
    except Exception as e:
        logger.error(f"[QWEN] Error generating response: {e}")
        return ""


@app.post("/qwen/intent", response_model=QwenIntentResponse)
def qwen_classify_intent(req: QwenIntentRequest) -> QwenIntentResponse:
    """Classify intent using Qwen model."""
    if not _qwen_model or not _qwen_tokenizer:
        raise HTTPException(status_code=503, detail="Qwen model not loaded")
    
    intent = _qwen_classify_intent(req.query)
    return QwenIntentResponse(intent=intent)


@app.post("/qwen/generate", response_model=QwenGenerateResponse)
def qwen_generate_response(req: QwenGenerateRequest) -> QwenGenerateResponse:
    """Generate assistant message using Qwen model."""
    if not _qwen_model or not _qwen_tokenizer:
        raise HTTPException(status_code=503, detail="Qwen model not loaded")
    
    message = _qwen_generate_response(req.query, req.intent, req.has_products)
    if not message:
        message = "Mình chưa tìm được sản phẩm phù hợp. Bạn có thể mô tả rõ hơn loại trang phục (loại, màu sắc, mức giá) được không?"
    
    return QwenGenerateResponse(message=message)


def _qwen_classify_attribute_change(query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Classify if user wants to change an attribute (color, material, theme) using Qwen model."""
    if not _qwen_model or not _qwen_tokenizer:
        return {"is_attribute_change": False, "attribute_type": None, "action": None}
    
    try:
        # Build context from conversation history if available
        context_str = ""
        if conversation_history and len(conversation_history) > 0:
            recent_queries = [item.get("query", "") for item in conversation_history[-3:]]  # Last 3 queries
            context_str = "\n".join([f"- {q}" for q in recent_queries if q])
            context_str = f"\n\nLịch sử hội thoại gần đây:\n{context_str}" if context_str else ""
        
        prompt = f"""<|im_start|>system
Bạn là bộ phân loại intent cho một TRỢ LÝ TƯ VẤN THỜI TRANG.
Nhiệm vụ của bạn: Phân tích câu của user và xác định xem họ có muốn THAY ĐỔI một thuộc tính (màu sắc, chất liệu, phong cách) hay không.

Các trường hợp CẦN phát hiện:
1. User muốn THAY ĐỔI thuộc tính:
   - "I want change Material" → attribute_change, material, replace
   - "Can I switch color?" → attribute_change, color, replace
   - "I want different theme" → attribute_change, theme, replace
   - "Show me something else" → attribute_change, null, replace (không rõ attribute)

2. User KHÔNG THÍCH thuộc tính (cần hỏi lại):
   - "I dislike this color" → attribute_change, color, exclude
   - "I hate this material" → attribute_change, material, exclude
   - "Don't like this theme" → attribute_change, theme, exclude

3. User KHÔNG muốn thay đổi (tìm kiếm bình thường):
   - "Show me pink hats" → not attribute_change
   - "I want black shirts" → not attribute_change
   - "Find me cotton dresses" → not attribute_change

Trả về JSON với format:
{{
  "is_attribute_change": true/false,
  "attribute_type": "color" | "material" | "theme" | null,
  "action": "replace" | "exclude" | null
}}

Chỉ trả về JSON, không có text thêm.
<|im_end|>
<|im_start|>user
Câu của user: "{query}"{context_str}
<|im_end|>
<|im_start|>assistant
"""
        
        inputs = _qwen_tokenizer(prompt, return_tensors="pt")
        device = next(_qwen_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _qwen_model.generate(
                **inputs,
                max_new_tokens=50,  # Enough for JSON response
                do_sample=False,
            )
        
        response = _qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_str = response.split("assistant")[-1].strip()
        
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_str)
            
            # Validate and normalize
            is_change = bool(result.get("is_attribute_change", False))
            attr_type = result.get("attribute_type")
            if attr_type not in ["color", "material", "theme", None]:
                attr_type = None
            action = result.get("action")
            if action not in ["replace", "exclude", None]:
                action = None
            
            return {
                "is_attribute_change": is_change,
                "attribute_type": attr_type,
                "action": action
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[QWEN] Failed to parse attribute change response: {e}, raw: {json_str}")
            # Fallback: try to detect from keywords
            query_lower = query.lower()
            if any(word in query_lower for word in ["change", "switch", "different", "other", "dislike", "hate", "don't like"]):
                # Try to detect attribute
                if "color" in query_lower or "màu" in query_lower:
                    return {"is_attribute_change": True, "attribute_type": "color", "action": "replace"}
                elif "material" in query_lower or "chất liệu" in query_lower:
                    return {"is_attribute_change": True, "attribute_type": "material", "action": "replace"}
                elif "theme" in query_lower or "phong cách" in query_lower:
                    return {"is_attribute_change": True, "attribute_type": "theme", "action": "replace"}
            return {"is_attribute_change": False, "attribute_type": None, "action": None}
    except Exception as e:
        logger.error(f"[QWEN] Error classifying attribute change: {e}")
        return {"is_attribute_change": False, "attribute_type": None, "action": None}


@app.post("/qwen/attribute-change", response_model=QwenAttributeChangeResponse)
def qwen_classify_attribute_change(req: QwenAttributeChangeRequest) -> QwenAttributeChangeResponse:
    """Classify if user wants to change an attribute using Qwen model."""
    if not _qwen_model or not _qwen_tokenizer:
        raise HTTPException(status_code=503, detail="Qwen model not loaded")
    
    result = _qwen_classify_attribute_change(req.query, req.conversation_history)
    return QwenAttributeChangeResponse(
        is_attribute_change=result["is_attribute_change"],
        attribute_type=result["attribute_type"],
        action=result["action"]
    )


# =======================================================================
# VTO (Virtual Try-On) Endpoints
# =======================================================================

class VTORequest(BaseModel):
    person_image: str  # Base64 encoded image
    cloth_image: str  # Base64 encoded image or product image URL
    product_id: Optional[str] = None  # Optional product ID to fetch image


class VTOResponse(BaseModel):
    result_image: str  # Base64 encoded result image
    status: str  # "success" | "error"
    message: Optional[str] = None


@app.post("/vto", response_model=VTOResponse)
def virtual_tryon(req: VTORequest) -> VTOResponse:
    """Run Virtual Try-On with person pose and clothing item."""
    try:
        import sys
        from pathlib import Path
        # Add current directory to path to ensure vto_service can be imported
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        from vto_service import run_virtual_tryon, base64_to_image, image_to_base64
        import requests
        from io import BytesIO
        from PIL import Image
        
        # Load person image
        try:
            person_img = base64_to_image(req.person_image)
        except Exception as e:
            logger.error(f"[VTO] Error loading person image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid person image: {str(e)}")
        
        # Load cloth image
        try:
            if req.cloth_image.startswith("http://") or req.cloth_image.startswith("https://"):
                # Fetch image from URL
                response = requests.get(req.cloth_image, timeout=10)
                response.raise_for_status()
                cloth_img = Image.open(BytesIO(response.content)).convert("RGB")
            elif req.cloth_image.startswith("data:image"):
                # Base64 encoded image
                cloth_img = base64_to_image(req.cloth_image)
            else:
                # Assume base64 without prefix
                cloth_img = base64_to_image(req.cloth_image)
        except Exception as e:
            logger.error(f"[VTO] Error loading cloth image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid cloth image: {str(e)}")
        
        # Run VTO pipeline via Custom HuggingFace Space
        logger.info("[VTO] Running virtual try-on via HuggingFace Space...")
        # Generate garment description from product info if available
        garment_description = None
        if req.product_id:
            # Could fetch product details here if needed
            garment_description = "a photo of clothing item"
        
        result_img = run_virtual_tryon(
            person_image=person_img,
            cloth_image=cloth_img,
            num_inference_steps=30,  # Default inference steps
            guidance_scale=2.5,
            garment_description=garment_description
        )
        
        # Convert result to base64
        result_base64 = image_to_base64(result_img)
        
        return VTOResponse(
            result_image=result_base64,
            status="success",
            message="Virtual try-on completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VTO] Error in virtual try-on: {e}")
        return VTOResponse(
            result_image="",
            status="error",
            message=f"VTO processing failed: {str(e)}"
        )


# =======================================================================
# Main (dev only)
# =======================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
