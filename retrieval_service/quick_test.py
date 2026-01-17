"""
Quick test script - Test nhanh service với một vài query đơn giản
Chạy script này sau khi service đã start
"""
import json
import sys

try:
    import requests
except ImportError:
    print("Cần cài đặt requests: pip install requests")
    sys.exit(1)


def quick_test():
    base_url = "http://127.0.0.1:8000"
    
    print("=" * 60)
    print("QUICK TEST - Retrieval Service")
    print("=" * 60)
    
    # Test 1: Kiểm tra service
    print("\n[1] Kiểm tra service...")
    try:
        r = requests.get(f"{base_url}/docs", timeout=5)
        if r.status_code == 200:
            print("✓ Service đang chạy")
        else:
            print(f"✗ Service trả về: {r.status_code}")
            return
    except Exception as e:
        print(f"✗ Không thể kết nối: {e}")
        print("\n→ Hãy đảm bảo service đã được start:")
        print("  python -m retrieval_service.main")
        return
    
    # Test 2: Simple text query
    print("\n[2] Test text query: 'jeans'")
    try:
        r = requests.post(
            f"{base_url}/search",
            json={"q": "jeans", "k": 3},
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            items = data.get("items", [])
            print(f"✓ Tìm thấy {len(items)} kết quả")
            if items:
                print(f"  Top 1: Score = {items[0]['score']:.4f}")
                meta = items[0].get("meta", {})
                title = meta.get("Title") or meta.get("Name", "N/A")
                print(f"  Product: {title[:50]}")
        elif r.status_code == 503:
            print("⚠ Service đang load model, vui lòng đợi...")
        else:
            print(f"✗ Lỗi: {r.status_code}")
            print(f"  {r.text[:200]}")
    except Exception as e:
        print(f"✗ Lỗi: {e}")
    
    # Test 3: Slots query
    print("\n[3] Test slots query: category='jeans', sex='Men'")
    try:
        r = requests.post(
            f"{base_url}/search",
            json={
                "slots": {"category": "jeans", "sex": "Men"},
                "k": 5,
                "n": 3
            },
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            items = data.get("items", [])
            print(f"✓ Tìm thấy {len(items)} kết quả")
            if items:
                print(f"  Top 1: Score = {items[0]['score']:.4f}")
        elif r.status_code == 503:
            print("⚠ Service đang load model, vui lòng đợi...")
        else:
            print(f"✗ Lỗi: {r.status_code}")
    except Exception as e:
        print(f"✗ Lỗi: {e}")
    
    print("\n" + "=" * 60)
    print("Test hoàn tất!")
    print("=" * 60)


if __name__ == "__main__":
    quick_test()

