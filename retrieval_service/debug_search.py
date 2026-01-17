"""
Script để debug search - kiểm tra dữ liệu và test search logic
"""
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _index_dir() -> Path:
    return _project_root() / "index_retrival"


def _get_adapter_path() -> Path:
    adapter_path = _project_root() / "epoch_2_final"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing epoch_2_final directory at {adapter_path}")
    return adapter_path


def check_data():
    """Kiểm tra dữ liệu trong database"""
    print("=" * 60)
    print("KIỂM TRA DỮ LIỆU TRONG DATABASE")
    print("=" * 60)
    
    # Load metadata
    meta_path = _index_dir() / "meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    print(f"\nTổng số sản phẩm: {len(meta)}")
    
    # Tìm tất cả categories
    categories = set()
    for m in meta:
        cat = m.get("Category", "").strip()
        if cat:
            categories.add(cat.lower())
    
    print(f"\nTất cả categories trong database:")
    for cat in sorted(categories):
        count = sum(1 for m in meta if m.get("Category", "").strip().lower() == cat)
        print(f"  - {cat}: {count} sản phẩm")
    
    # Tìm sản phẩm có chứa "polo"
    print("\n" + "=" * 60)
    print("TÌM SẢN PHẨM CÓ CHỨA 'POLO'")
    print("=" * 60)
    
    polo_products = []
    for i, m in enumerate(meta):
        cat = str(m.get("Category", "")).lower()
        title = str(m.get("Title", "")).lower()
        name = str(m.get("Name", "")).lower()
        desc = str(m.get("Description", "")).lower()
        
        if "polo" in cat or "polo" in title or "polo" in name or "polo" in desc:
            polo_products.append((i, m))
    
    print(f"\nTìm thấy {len(polo_products)} sản phẩm có chứa 'polo':")
    for idx, m in polo_products[:10]:  # Hiển thị 10 sản phẩm đầu
        cat = m.get("Category", "N/A")
        sex = m.get("Sex", "N/A")
        title = m.get("Title") or m.get("Name", "N/A")
        price = m.get("PriceNum", "N/A")
        print(f"\n  [{idx}] {title}")
        print(f"      Category: {cat}")
        print(f"      Sex: {sex}")
        print(f"      Price: {price}")
    
    # Tìm sản phẩm Polo cho Men
    print("\n" + "=" * 60)
    print("TÌM SẢN PHẨM POLO CHO MEN")
    print("=" * 60)
    
    polo_men = []
    for i, m in enumerate(meta):
        cat = str(m.get("Category", "")).lower()
        sex = str(m.get("Sex", "")).lower()
        
        # Check category có chứa polo
        cat_has_polo = "polo" in cat
        # Check sex
        sex_is_men = sex in ["men", "male", "man"]
        
        if cat_has_polo and sex_is_men:
            polo_men.append((i, m))
    
    print(f"\nTìm thấy {len(polo_men)} sản phẩm Polo cho Men:")
    for idx, m in polo_men[:10]:
        cat = m.get("Category", "N/A")
        sex = m.get("Sex", "N/A")
        title = m.get("Title") or m.get("Name", "N/A")
        price = m.get("PriceNum", "N/A")
        print(f"\n  [{idx}] {title}")
        print(f"      Category: {cat}")
        print(f"      Sex: {sex}")
        print(f"      Price: {price}")


def test_category_matching():
    """Test category matching logic"""
    print("\n" + "=" * 60)
    print("TEST CATEGORY MATCHING LOGIC")
    print("=" * 60)
    
    def norm(s: Any) -> str:
        return (str(s).strip().lower()) if s is not None else ""
    
    test_cases = [
        ("polo", "polo"),
        ("polo", "polo shirt"),
        ("polo", "shirt"),
        ("polo", "Polo Shirt"),
        ("polo", "POLO"),
    ]
    
    print("\nTest cases:")
    for want_cat, db_cat in test_cases:
        want_cat_norm = want_cat.rstrip('s') if want_cat.endswith('s') else want_cat
        cat_val_norm = db_cat.rstrip('s') if db_cat.endswith('s') else db_cat
        want_cat_norm = want_cat_norm.lower()
        cat_val_norm = cat_val_norm.lower()
        
        # Current logic
        match = (cat_val_norm == want_cat_norm) or (want_cat_norm in cat_val_norm) or (cat_val_norm in want_cat_norm)
        
        print(f"  want='{want_cat}' vs db='{db_cat}' -> {match}")
        print(f"    normalized: '{want_cat_norm}' vs '{cat_val_norm}'")
        print(f"    exact: {cat_val_norm == want_cat_norm}")
        print(f"    want in db: {want_cat_norm in cat_val_norm}")
        print(f"    db in want: {cat_val_norm in want_cat_norm}")


def test_search_simulation():
    """Simulate search với query 'polo men'"""
    print("\n" + "=" * 60)
    print("SIMULATE SEARCH: 'polo men'")
    print("=" * 60)
    
    # Load data
    meta_path = _index_dir() / "meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    # Simulate filtering
    def norm(s: Any) -> str:
        return (str(s).strip().lower()) if s is not None else ""
    
    want_cat = "polo"
    want_sex = "men"
    
    want_cat_norm = want_cat.rstrip('s') if want_cat.endswith('s') else want_cat
    want_sex_norm = want_sex.lower()
    
    print(f"\nFilters:")
    print(f"  Category: '{want_cat}' (normalized: '{want_cat_norm}')")
    print(f"  Sex: '{want_sex}' (normalized: '{want_sex_norm}')")
    
    # Test filtering
    matched = []
    for i, m in enumerate(meta):
        cat_val = norm(m.get("Category"))
        sex_val = norm(m.get("Sex"))
        
        cat_val_norm = cat_val.rstrip('s') if cat_val.endswith('s') else cat_val
        
        # Category matching
        cat_ok = (cat_val_norm == want_cat_norm) or (want_cat_norm in cat_val_norm) or (cat_val_norm in want_cat_norm)
        
        # Sex matching
        sex_ok = (sex_val == want_sex_norm)
        
        if cat_ok and sex_ok:
            matched.append((i, m, cat_val, sex_val))
    
    print(f"\nKết quả filtering: {len(matched)} sản phẩm match")
    for idx, m, cat, sex in matched[:10]:
        title = m.get("Title") or m.get("Name", "N/A")
        print(f"  [{idx}] {title} | Category: {cat} | Sex: {sex}")


if __name__ == "__main__":
    try:
        check_data()
        test_category_matching()
        test_search_simulation()
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()

