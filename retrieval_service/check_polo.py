"""Quick check for Polo products"""
import pickle
from pathlib import Path

def _project_root():
    return Path(__file__).resolve().parents[1]

def _index_dir():
    return _project_root() / "index_retrival"

# Load metadata
meta_path = _index_dir() / "meta.pkl"
print(f"Loading from: {meta_path}")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

print(f"Total products: {len(meta)}")

# Find Polo products
polo_all = []
polo_men = []

for i, m in enumerate(meta):
    cat = str(m.get("Category", "")).lower()
    sex = str(m.get("Sex", "")).lower()
    title = str(m.get("Title", "") or m.get("Name", "")).lower()
    
    if "polo" in cat or "polo" in title:
        polo_all.append((i, m))
        if sex in ["men", "male", "man"]:
            polo_men.append((i, m))

print(f"\nProducts with 'polo' in category or title: {len(polo_all)}")
print(f"Polo products for men: {len(polo_men)}")

if polo_men:
    print("\nFirst 5 Polo products for men:")
    for idx, m in polo_men[:5]:
        print(f"\n[{idx}]")
        print(f"  Title: {m.get('Title') or m.get('Name', 'N/A')}")
        print(f"  Category: {m.get('Category', 'N/A')}")
        print(f"  Sex: {m.get('Sex', 'N/A')}")
        print(f"  Price: {m.get('PriceNum', 'N/A')}")
else:
    print("\nNo Polo products for men found!")
    if polo_all:
        print("\nBut found Polo products (all genders):")
        for idx, m in polo_all[:5]:
            print(f"\n[{idx}]")
            print(f"  Title: {m.get('Title') or m.get('Name', 'N/A')}")
            print(f"  Category: {m.get('Category', 'N/A')}")
            print(f"  Sex: {m.get('Sex', 'N/A')}")

# Check all unique categories
categories = {}
for m in meta:
    cat = m.get("Category", "").strip()
    if cat:
        cat_lower = cat.lower()
        if cat_lower not in categories:
            categories[cat_lower] = []
        categories[cat_lower].append(cat)

print("\n" + "="*60)
print("All unique categories (case-insensitive):")
for cat_lower in sorted(categories.keys()):
    if "polo" in cat_lower or "shirt" in cat_lower:
        count = sum(1 for m in meta if m.get("Category", "").strip().lower() == cat_lower)
        print(f"  {cat_lower}: {count} products")


