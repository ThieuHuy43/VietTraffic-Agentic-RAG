# VietTraffic Agentic RAG 🚦🇻🇳

Hệ thống **VietTraffic-Agentic-RAG** là một ứng dụng Hỏi-Đáp (Q&A) thông minh chuyên biệt về Luật Giao thông đường bộ Việt Nam. Hệ thống sử dụng kiến trúc **Agentic RAG** (Retrieval-Augmented Generation kết hợp với AI Agents), cho phép tra cứu thông tin nhanh chóng, chính xác dựa trên cơ sở dữ liệu luật pháp kết hợp với khả năng tìm kiếm thông tin theo thời gian thực.

## 🌟 Tính năng nổi bật

- **Giao diện trực quan**: Ứng dụng web được xây dựng bằng [Streamlit](https://streamlit.io/) thân thiện và dễ sử dụng.
- **Truy xuất thông tin mạnh mẽ**: Tích hợp **Qdrant** làm Vector Database để lưu trữ và tìm kiếm các điều luật giao thông ngữ nghĩa.
- **Tích hợp Agentic LLM**: Sử dụng sức mạnh của **Google Gemini** làm ngôn ngữ mô hình lõi và **Tavily** để Agent có thể tra cứu thêm thông tin thực tế từ Internet khi cần.
- **Kiến trúc Microservices**: Tách biệt hoàn toàn Frontend và Backend (API), dễ dàng scale và bảo trì.
- **Triển khai cực nhanh**: Hỗ trợ Docker & Docker Compose để chạy ứng dụng ở bất kỳ đâu chỉ bằng một dòng lệnh.

## 🛠️ Công nghệ sử dụng

- **Frontend:** Streamlit, Python
- **Backend API:** FastAPI / LangChain / LlamaIndex (Python)
- **Vector Database:** Qdrant
- **LLM & Tools:** Google Gemini API, Tavily Search API
- **Deployment:** Docker, Docker Compose, Makefile, Shell/PowerShell scripts

## 🚀 Hướng dẫn cài đặt và chạy ứng dụng

### 1. Yêu cầu hệ thống
- Đã cài đặt [Docker](https://docs.docker.com/get-docker/) và [Docker Compose](https://docs.docker.com/compose/install/).
- Cần có các API keys: `GEMINI_API_KEY` và `TAVILY_API_KEY`.

### 2. Thiết lập môi trường

Đầu tiên, hãy clone repository về máy:
```bash
git clone https://github.com/ThieuHuy43/VietTraffic-Agentic-RAG.git
cd VietTraffic-Agentic-RAG
```

Tạo một file `.env` ở thư mục gốc (nơi chứa file `docker-compose.yml`) và điền các khóa API của bạn vào:

```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Khởi chạy ứng dụng

Dự án đã hỗ trợ sẵn các script để tự động khởi chạy các containers một cách nhanh chóng. 

**Dành cho Linux / macOS:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Dành cho Windows (PowerShell):**
```powershell
.\deploy.ps1
```

**Hoặc khởi chạy trực tiếp thông qua Docker Compose hoặc Makefile:**
```bash
# Bằng Makefile
make up

# Bằng Docker Compose
docker-compose up -d --build
```

### 4. Truy cập ứng dụng

Sau khi các container khởi động thành công (bạn có thể dùng `docker-compose ps` để kiểm tra):
- **Web UI (Streamlit):** Truy cập [http://localhost:8501](http://localhost:8501) để chat với hệ thống.
- **Backend API Docs (Swagger):** Truy cập [http://localhost:8000/docs](http://localhost:8000/docs).
- **Qdrant DB API:** Đang chạy tại cổng `6333` (Dashboard tại http://localhost:6333/dashboard).

## 📂 Cấu trúc thư mục

```text
VietTraffic-Agentic-RAG/
│
├── backend/            # Chứa mã nguồn cho FastAPI Backend và hệ thống Agent/RAG
├── frontend/           # Chứa mã nguồn cho Streamlit UI
├── data/               # Nơi lưu trữ tài liệu dữ liệu thô và Qdrant storage
├── checkpoints/        # Lưu trữ checkpoint/state của Agent
├── docker-compose.yml  # File cấu hình để dựng toàn bộ các dịch vụ
├── Makefile            # Chứa các lệnh tắt để deploy
├── deploy.sh           # Script khởi chạy cho Linux/macOS
└── deploy.ps1          # Script khởi chạy cho Windows
```

## 🤝 Đóng góp

Mọi đóng góp (Pull Requests, Issues) để cải thiện dự án luôn được hoan nghênh! Nếu bạn thấy repo này hữu ích, đừng quên cho một ⭐ nhé!
