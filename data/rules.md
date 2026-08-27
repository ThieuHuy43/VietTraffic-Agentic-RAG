# Quy tắc Xử lý Dữ liệu (Data Processing Rules) - VietTraffic Agentic RAG

Dựa trên tài liệu mô tả dữ liệu và cấu trúc phức tạp của các văn bản pháp luật, dưới đây là bộ 5 Quy tắc Xử lý dữ liệu nòng cốt mà hệ thống Agentic RAG tuân thủ để đảm bảo tính chính xác và không sinh ra ảo giác (hallucination).

## 📜 Rule 1: Xử lý Xung đột và Khấu trừ Hiệu lực (Conflict Resolution)
*Đây là rule quan trọng nhất để Chatbot không tư vấn sai luật hết hạn.*

*   **Luật Đường bộ (35/2024) & Luật TTATGTĐB (36/2024):** Gắn thẻ `status: active` và `supersedes: Luat_23_2008`.
*   **Luật Giao thông đường bộ 2008:** Phải bị gắn thẻ `status: superseded` (Hết hiệu lực). 
    👉 *Hành động của Bot:* Nếu Qdrant vô tình trích xuất lên Luật 2008, Bot sẽ tự động block chunk này lại hoặc bắt buộc phải đối chiếu xem Luật 2024 có quy định khác không trước khi trả lời.
*   **Xử lý Nghị định (NĐ 100/2019 & NĐ 123/2021):** NĐ 123 không thay thế toàn bộ NĐ 100 mà chỉ **sửa đổi, bổ sung**.
    👉 *Hành động của Bot:* NĐ 123 phải được ưu tiên trọng số cao hơn. Nếu câu hỏi chạm vào một "Điều" bị sửa đổi (vd: Điều 16 về phạt xe kinh doanh), Bot phải gộp ngữ cảnh của cả NĐ 100 và 123 để đưa ra mức phạt tổng hợp chính xác nhất.

## 🏷️ Rule 2: Phân luồng Ngữ cảnh theo Intent (Metadata Routing)
Hệ thống đánh Tag Metadata cho từng văn bản theo 3 nhóm cốt lõi:

*   **Tag `doc_type: nghi_dinh` (Nhóm 2):** Kích hoạt trọng số (boost) tối đa nếu câu hỏi của người dùng có chứa từ khóa: *"phạt bao nhiêu", "lỗi", "bị giam xe không", "tước bằng"*.
*   **Tag `doc_type: quy_chuan` / `thong_tu` (Nhóm 3):** Kích hoạt trọng số tối đa nếu câu hỏi chứa từ khóa mang tính kỹ thuật: *"biển báo", "vạch kẻ đường", "tốc độ tối đa", "đèn tín hiệu", "bằng B2"*.
*   **Tag `doc_type: luat` (Nhóm 1):** Dùng để trả lời các quy tắc nền tảng: *"Được phép rẽ phải khi đèn đỏ không?", "độ tuổi chạy xe"*.

## 🧩 Rule 3: Chiến lược Băm nhỏ Dữ liệu (Hierarchical Chunking)
*Pháp luật không thể cắt ngang câu. Phải bảo toàn cấu trúc cây.*

*   **Cắt theo node:** `Chương` ➡️ `Điều` ➡️ `Khoản` ➡️ `Điểm`.
*   **Nguyên tắc Parent-Child Context:** Mỗi khi tách một "Khoản" (vd: *Khoản 1. Phạt tiền từ 2-3 triệu*), hệ thống **bắt buộc** phải kẹp thêm tiêu đề của "Điều" vào ngay phía trước (vd: *[Điều 5. Lỗi vượt đèn đỏ] Khoản 1. Phạt tiền từ 2-3 triệu*). Điều này giúp Vector Database không bị lạc trôi ngữ cảnh khi tính toán khoảng cách vector.

## 🔍 Rule 4: Chiến lược Truy xuất Kết hợp (Hybrid RRF Retrieval)
Vì văn bản luật VN có cách hành văn rất khuôn mẫu, nếu chỉ dùng Semantic Search (Tìm theo nghĩa) sẽ dễ bị sai lệch.

*   **Dense Vector (Semantic):** Bắt ý định người dùng (VD: User hỏi *"kẹp 3"* ➡️ Dense Vector hiểu là *"chở quá số người quy định"*).
*   **Sparse Vector (Lexical/BM25):** Bắt buộc đối chiếu chính xác từ khóa (VD: Phải khớp đúng chữ *"Nghị định 100"*, *"Điều 16"*, *"100 km/h"*).
*   **RRF (Reciprocal Rank Fusion):** Gộp điểm của 2 loại Vector trên để chốt ra Top 6 đoạn luật chuẩn nhất.

## 📝 Rule 5: Nguyên tắc Ép buộc Sinh văn bản (Strict Generation Guidelines)
Khi cung cấp ngữ cảnh cho LLM (DeepSeek / Gemini), Prompt hệ thống ép buộc LLM phải tuân thủ 3 lệnh cấm:

1.  **Cấm bịa đặt:** Chỉ được phép lấy thông tin nằm trong Context. Nếu không có thông tin, phải báo "Không tìm thấy" và đẩy cờ cho luồng `web_search_node` hoặc Admin xử lý (HITL).
2.  **Cấm nói chung chung:** Bắt buộc trích dẫn nguồn gốc ở cuối mỗi lập luận theo format `[Tên văn bản, Điều X, Khoản Y]`.
3.  **Văn phong:** Nghiêm túc, súc tích, dễ hiểu, tránh trích dẫn dài dòng nguyên cả một Điều luật nếu không cần thiết, chỉ đưa ra thông tin cốt lõi (Mức phạt, hình phạt bổ sung).
