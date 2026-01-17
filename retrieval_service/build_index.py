"""
Script để build FAISS index mới với Qwen 3B model
Sử dụng model đã fine-tune từ epoch_2_final
"""
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

# Fix OpenMP conflict on Windows (must be before importing torch/faiss)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _index_dir() -> Path:
    return _project_root() / "index_retrival"


def _get_adapter_path() -> Path:
    """Get path to the Qwen LoRA adapter (epoch_2_final)"""
    adapter_path = _project_root() / "epoch_2_final"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing epoch_2_final directory at {adapter_path}")
    return adapter_path


def _load_qwen_model() -> tuple[Any, Any]:
    """Load Qwen 3B model with LoRA adapter"""
    adapter_path = _get_adapter_path()

    # Read base model from adapter config
    config_path = adapter_path / "adapter_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")

    # Load tokenizer
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load model FOR CAUSAL LM (not AutoModel!!!)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    return model, tokenizer


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vectors"""
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def _encode_text_batch(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    batch_size: int = 16,
    normalize: bool = True
) -> np.ndarray:

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Lấy hidden states từ mô hình causal LM
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,   # bật hidden states
                return_dict=True,
                use_cache=False              # tắt KV-cache
            )

            # hidden_states là tuple: (layer0, layer1, ..., last_layer)
            last_hidden_state = outputs.hidden_states[-1]

            attention_mask = inputs["attention_mask"]
            mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

            sum_embeddings = torch.sum(last_hidden_state * mask_exp, dim=1)
            sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        embeddings = embeddings.cpu().numpy().astype(np.float32)
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)

    if normalize:
        all_embeddings = _normalize(all_embeddings)

    return all_embeddings



def _build_product_text(meta: Dict[str, Any]) -> str:
    parts = []

    if meta.get("Category"):
        parts.append(str(meta["Category"]))

    if meta.get("Sex"):
        parts.append(str(meta["Sex"]))

    if meta.get("Color"):
        parts.append(str(meta["Color"]))

    if meta.get("Material"):
        parts.append(str(meta["Material"]))

    if meta.get("Theme"):
        parts.append(str(meta["Theme"]))

    if meta.get("Title"):
        parts.append(str(meta["Title"]))
    elif meta.get("Name"):
        parts.append(str(meta["Name"]))

    if meta.get("Description"):
        parts.append(str(meta["Description"]))

    return " ".join(parts)


def main():
    print("=" * 60)
    print("Building FAISS index with Qwen 3B model")
    print("=" * 60)

    meta_path = _index_dir() / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.pkl at {meta_path}")

    print(f"\nLoading metadata from: {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    print(f"Loaded {len(meta)} products")

    print("\nLoading Qwen 3B model...")
    model, tokenizer = _load_qwen_model()
    print("Model loaded successfully!")

    print("\nBuilding text representations...")
    product_texts = [
        _build_product_text(m)
        for m in tqdm(meta, desc="Processing products")
    ]

    print("\nEncoding products with Qwen 3B...")
    embeddings = _encode_text_batch(model, tokenizer, product_texts, batch_size=16)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    print("\nBuilding FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    out_path = _index_dir() / "products.faiss"
    print(f"\nSaving index to: {out_path}")
    faiss.write_index(index, str(out_path))

    print("\n" + "=" * 60)
    print("Index build completed successfully!")
    print(f"Total vectors: {index.ntotal}")
    print("=" * 60)


if __name__ == "__main__":
    main()
