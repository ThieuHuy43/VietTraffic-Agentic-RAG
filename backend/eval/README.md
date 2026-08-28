# Eval pipeline

`golden_dataset.json` — bộ câu hỏi đã verify thật qua `/chat`, mỗi item gồm câu hỏi + citation
đúng mong đợi (`expected_doc_id`, `expected_dieu`, `expected_khoan` — 2 field sau có thể `null`
nếu không cần khớp tới mức đó, ví dụ chunk bảng không có khoản).

## `eval_retrieval.py`
Test riêng tầng retriever (Qdrant hybrid search + reranker), **không** qua `expand_query`/LLM.
Nhanh, không tốn API, dùng để regression-test mỗi khi sửa `ingest.py` hoặc
`retriever/qdrant_retriever.py`. Chạy trong container (cần Qdrant thật đã ingest dữ liệu):

```
docker compose exec backend_api python eval/eval_retrieval.py
```

Vì bỏ qua `expand_query`, hit-rate ở đây **thấp hơn** `eval_e2e.py` với các câu hỏi dùng từ
ngữ thông tục (VD "vượt đèn đỏ" chưa được dịch sang "không chấp hành hiệu lệnh của đèn tín
hiệu giao thông") — đó là tín hiệu bình thường cho thấy `expand_query` đang phát huy tác dụng,
không phải lỗi harness.

## `eval_e2e.py`
Test toàn bộ pipeline qua `/chat` thật (router → expand_query → retrieve → grade → generate).
Chậm hơn (mỗi câu ~15-60s tùy độ dài context) và tốn LLM call thật. Yêu cầu server đang chạy:

```
docker compose exec backend_api python eval/eval_e2e.py
# hoặc chạy ngoài container nếu đã map port 8000 ra host:
EVAL_BASE_URL=http://localhost:8000 python backend/eval/eval_e2e.py
```

Câu hỏi rơi vào HITL/web_search fallback được tính là **FAIL** (vì đây là câu hỏi golden dataset
lẽ ra phải trả lời được từ dữ liệu đã ingest).

## Thêm câu hỏi mới vào golden_dataset.json
Không tự đoán citation — luôn chạy thật qua `/chat` trước, đọc citation LLM trả về, verify đúng
với văn bản luật gốc rồi mới thêm vào dataset. Dataset sai sẽ khiến eval báo fail giả (false
negative) hoặc pass giả (false positive) cho các lần sửa code sau này.
