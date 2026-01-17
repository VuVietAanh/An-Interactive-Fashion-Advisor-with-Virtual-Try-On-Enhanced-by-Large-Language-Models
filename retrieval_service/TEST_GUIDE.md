# Hướng dẫn Test Retrieval Service

## Bước 1: Cài đặt dependencies

```bash
cd retrieval_service
pip install -r requirements.txt
```

## Bước 2: Start service

Trong terminal thứ nhất:

```bash
cd retrieval_service
python -m retrieval_service.main
```

Hoặc:

```bash
cd retrieval_service
python main.py
```

Service sẽ chạy tại `http://127.0.0.1:8000`

**Lưu ý**: Lần đầu tiên start, service cần download Qwen model từ HuggingFace và load LoRA adapter. Quá trình này có thể mất vài phút.

Bạn sẽ thấy các thông báo:
- "Loading base model: Qwen/Qwen2.5-3B-Instruct"
- "Loading LoRA adapter from: ..."
- "INFO:     Uvicorn running on http://127.0.0.1:8000"

## Bước 3: Test service

Trong terminal thứ hai (sau khi service đã start):

```bash
cd retrieval_service
python test_service.py
```

Script sẽ test:
1. ✅ Service có chạy không
2. ✅ Text query search (không có slots)
3. ✅ Slots-based search
4. ✅ Edge cases (empty query, no query)

## Bước 4: Test thủ công với curl

### Test text query:
```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"q": "jeans", "k": 5}'
```

### Test slots search:
```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "slots": {
      "category": "jeans",
      "sex": "Men"
    },
    "k": 10,
    "n": 5
  }'
```

## Bước 5: Xem API documentation

Mở trình duyệt và truy cập:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Troubleshooting

### Service trả về 503 (Service Unavailable)
- Model đang được load, đợi thêm vài giây
- Kiểm tra xem có đủ RAM/VRAM không

### Lỗi "Missing epoch_2_final directory"
- Đảm bảo thư mục `epoch_2_final` có trong project root
- Kiểm tra đường dẫn: `E:\CURSOR\code\epoch_2_final`

### Lỗi "Missing products.faiss"
- Đảm bảo file `index_retrival/products.faiss` tồn tại
- Nếu chưa có, chạy `python build_index.py` để build index

### Lỗi "Missing meta.pkl"
- Đảm bảo file `index_retrival/meta.pkl` tồn tại

