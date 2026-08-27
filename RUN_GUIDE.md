# 🚀 Hướng dẫn Vận hành VietTraffic Agentic RAG

Tài liệu này tổng hợp toàn bộ các câu lệnh cần thiết để bạn có thể Deploy, quản trị dữ liệu và bảo trì dự án một cách dễ dàng nhất trên môi trường Windows (PowerShell).

---

## 1. Cấu hình Môi trường
Trước khi chạy bất kỳ lệnh nào, hãy đảm bảo bạn đã điền đầy đủ các API Key vào file `.env` tại thư mục gốc:
```env
GEMINI_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
TAVILY_API_KEY=your_key
```

---

## 2. Khởi chạy / Triển khai Dự án (Deploy)
Bất cứ khi nào bạn **sửa code (Python, HTML, CSS)** hoặc **lần đầu tiên chạy dự án**, bạn cần chạy lệnh Deploy. Lệnh này sẽ build lại Docker Image và khởi động các container.

Mở **PowerShell** tại thư mục `viet-traffic-rag` và chạy:
```powershell
.\deploy.ps1
```
*Lưu ý: Script này mặc định sẽ KHÔNG nạp lại dữ liệu (Ingest) vào Qdrant để tiết kiệm thời gian và chi phí API.*

Sau khi Deploy thành công, hệ thống có thể truy cập tại:
- **Chatbot UI:** http://localhost:8000
- **Admin Dashboard (HITL):** http://localhost:8000/admin

---

## 3. Quản trị Dữ liệu (Re-ingest / Vector Database)

Khi bạn **thêm/xóa/sửa các file Luật (PDF, Word, HTML)** trong thư mục `data/raw_laws/`, bạn BẮT BUỘC phải nạp lại dữ liệu vào Database. Có 2 cách để làm việc này:

### CÁCH 1: Nạp lại cùng lúc với Deploy
Khởi tạo biến môi trường `FORCE_REINGEST` và chạy file deploy:
```powershell
$env:FORCE_REINGEST="1"; .\deploy.ps1
```

### CÁCH 2: Nạp lại thủ công khi Container đang chạy (Khuyên dùng)
Nếu hệ thống đang chạy ổn định và bạn chỉ muốn cập nhật dữ liệu, hãy chạy lệnh Docker này:
```bash
docker compose exec -e FORCE_REINGEST=1 backend_api python ingest.py
```
Hệ thống sẽ tự động xóa Collection cũ (`viet_traffic_laws`) và tạo Embedding mới.

---

## 4. Xem Log & Theo dõi Lỗi (Monitor)

Để biết hệ thống (LLM, LangGraph) đang xử lý luồng suy nghĩ (thought) như thế nào, hoặc để debug lỗi:

**Xem Log của Backend (FastAPI):**
```bash
docker compose logs -f backend_api
```
*(Bấm `Ctrl + C` để thoát chế độ xem log)*

**Xem Log của Database (Qdrant):**
```bash
docker compose logs -f qdrant
```

---

## 5. Quản lý Vòng đời Container

**Dừng toàn bộ hệ thống (Không xóa dữ liệu Qdrant vì đã có Volume):**
```bash
docker compose down
```

**Khởi động lại (Restart) hệ thống nhanh (Không build lại code):**
```bash
docker compose restart
```

**Xóa tận gốc toàn bộ Database Qdrant (Trường hợp dữ liệu bị hỏng nặng):**
```bash
docker compose down -v
```
*(Lệnh này sẽ xóa cả Volume `./data/qdrant_data`. Lần sau khởi động bắt buộc phải Re-ingest lại dữ liệu).*
