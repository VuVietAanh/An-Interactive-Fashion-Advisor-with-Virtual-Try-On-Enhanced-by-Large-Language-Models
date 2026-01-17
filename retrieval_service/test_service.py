"""
Script để test retrieval service với Qwen 3B
"""
import json
import time
from pathlib import Path

import requests


def test_service(base_url: str = "http://127.0.0.1:8000"):
    """Test các endpoint của service"""
    
    print("=" * 60)
    print("Testing Retrieval Service với Qwen 3B")
    print("=" * 60)
    
    # Test 1: Health check (kiểm tra service có chạy không)
    print("\n[Test 1] Kiểm tra service có chạy...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("✓ Service đang chạy")
        else:
            print(f"✗ Service trả về status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Không thể kết nối đến service. Hãy đảm bảo service đã được start:")
        print(f"  python -m retrieval_service.main")
        return False
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        return False
    
    # Test 2: Text query search (không có slots)
    print("\n[Test 2] Test text query search...")
    test_cases = [
        {"q": "jeans", "k": 5},
        {"q": "red dress", "k": 5},
        {"q": "leather jacket", "k": 3},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test 2.{i}: Query = '{test_case['q']}'")
        try:
            response = requests.post(
                f"{base_url}/search",
                json=test_case,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                print(f"    ✓ Tìm thấy {len(items)} kết quả")
                if items:
                    print(f"    ✓ Top result score: {items[0]['score']:.4f}")
                    if "meta" in items[0]:
                        meta = items[0]["meta"]
                        print(f"    ✓ Sample product: {meta.get('Title', meta.get('Name', 'N/A'))}")
            elif response.status_code == 503:
                print(f"    ⚠ Service chưa sẵn sàng (503). Đang load model...")
                print(f"    ⚠ Vui lòng đợi thêm vài giây để model load xong")
            else:
                print(f"    ✗ Lỗi: {response.status_code}")
                print(f"    Response: {response.text[:200]}")
        except Exception as e:
            print(f"    ✗ Lỗi: {e}")
    
    # Test 3: Slots-based search
    print("\n[Test 3] Test slots-based search...")
    slots_tests = [
        {
            "slots": {
                "category": "jeans",
                "sex": "Men"
            },
            "k": 5,
            "n": 3
        },
        {
            "slots": {
                "category": "dress",
                "color": ["red", "blue"]
            },
            "k": 10,
            "n": 5
        },
        {
            "slots": {
                "price": {"min": 50, "max": 200}
            },
            "k": 5
        }
    ]
    
    for i, test_case in enumerate(slots_tests, 1):
        print(f"\n  Test 3.{i}: Slots = {json.dumps(test_case['slots'], indent=6)}")
        try:
            response = requests.post(
                f"{base_url}/search",
                json=test_case,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                print(f"    ✓ Tìm thấy {len(items)} kết quả")
                if items:
                    print(f"    ✓ Top result score: {items[0]['score']:.4f}")
            elif response.status_code == 503:
                print(f"    ⚠ Service chưa sẵn sàng (503)")
            else:
                print(f"    ✗ Lỗi: {response.status_code}")
                print(f"    Response: {response.text[:200]}")
        except Exception as e:
            print(f"    ✗ Lỗi: {e}")
    
    # Test 4: Edge cases
    print("\n[Test 4] Test edge cases...")
    
    # Empty query
    print("\n  Test 4.1: Empty query (should fail)")
    try:
        response = requests.post(
            f"{base_url}/search",
            json={"q": ""},
            timeout=10
        )
        if response.status_code == 400:
            print("    ✓ Đúng: Trả về 400 cho empty query")
        else:
            print(f"    ⚠ Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"    ✗ Lỗi: {e}")
    
    # No query and no slots
    print("\n  Test 4.2: No query and no slots (should fail)")
    try:
        response = requests.post(
            f"{base_url}/search",
            json={},
            timeout=10
        )
        if response.status_code == 400:
            print("    ✓ Đúng: Trả về 400 khi không có query và slots")
        else:
            print(f"    ⚠ Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"    ✗ Lỗi: {e}")
    
    print("\n" + "=" * 60)
    print("Test hoàn tất!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    test_service(base_url)

