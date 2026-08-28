import os
import sqlite3
import json
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

import sys
# Thêm đường dẫn để nhận diện module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from agent import build_graph

app = FastAPI(title="Viet Traffic Legal Assistant API")

# Cho phép CORS để Frontend gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo LangGraph
agent_app = build_graph()

class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None

class ResumeRequest(BaseModel):
    thread_id: str
    action: str  # "approve", "edit", "reject"
    edited_content: Optional[str] = None

# Mô tả (tiếng Việt, thân thiện) cho từng bước thật trong LangGraph, để hiện tiến trình "thinking"
# cho người dùng qua SSE — phản ánh đúng công việc backend đang làm, không phải giả lập.
STEP_LABELS = {
    "router_node": "Đang xác định loại câu hỏi...",
    "retrieve_node": "Đang tìm kiếm trong cơ sở dữ liệu luật...",
    "grade_node": "Đang đánh giá mức độ liên quan của dữ liệu tìm được...",
    "web_search_node": "Không đủ dữ liệu nội bộ, đang tìm kiếm thêm từ nguồn uy tín trên mạng...",
    "generate_node": "Đang soạn câu trả lời...",
}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Endpoint Chat.
    Thực thi LangGraph và stream SSE. Nếu bị ngắt bởi HITL, trả về status="pending".
    Mỗi khi một node trong graph hoàn tất, gửi thêm event status="step" kèm mô tả bước đó
    để frontend hiện tiến trình xử lý thật (thinking) cho người dùng.
    Riêng generate_node dùng llm.stream() (xem nodes.py) nên LangGraph phát được token thật qua
    stream_mode="messages" — forward trực tiếp thành event status="token" để frontend hiện chữ
    xuất hiện dần thay vì đợi cả câu trả lời xong (không giảm tổng thời gian xử lý, nhưng cải
    thiện rõ cảm nhận tốc độ vì DeepSeek generate_node hay mất 10-30s+ cho câu trả lời dài).
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        try:
            for stream_type, event in agent_app.stream(
                {"question": req.question}, config=config, stream_mode=["updates", "messages"]
            ):
                if stream_type == "messages":
                    chunk, metadata = event
                    if metadata.get("langgraph_node") == "generate_node" and chunk.content:
                        yield f"data: {json.dumps({'status': 'token', 'content': chunk.content, 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
                    continue

                state = agent_app.get_state(config)
                # Kiểm tra nếu đồ thị đang bị đóng băng tại process_hitl_node
                if state.next and "process_hitl_node" in state.next:
                    yield f"data: {json.dumps({'status': 'pending', 'message': 'Đang chờ Admin duyệt kết quả tìm kiếm ngoài (HITL)...', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
                    break

                for node, update in event.items():
                    label = STEP_LABELS.get(node)
                    if label:
                        yield f"data: {json.dumps({'status': 'step', 'step': node, 'message': label, 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
                    if "final_answer" in update:
                        yield f"data: {json.dumps({'status': 'done', 'answer': update['final_answer'], 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': f'Lỗi hệ thống: {str(e)}', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/pending")
def pending_endpoint():
    """Lấy danh sách các câu hỏi đang chờ Admin duyệt (HITL)."""
    db_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "langgraph.sqlite")
    if not os.path.exists(db_path):
        return {"pending": []}
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Truy vấn các thread có trong DB
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        return {"pending": []}
    finally:
        conn.close()
        
    pending = []
    for (thread_id,) in rows:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = agent_app.get_state(config)
            # Lọc những state có điểm neo tiếp theo là process_hitl_node
            if state.next and "process_hitl_node" in state.next:
                pending.append({
                    "thread_id": thread_id,
                    "question": state.values.get("question"),
                    "draft_answer": state.values.get("draft_answer")
                })
        except Exception:
            pass
            
    return {"pending": pending}

@app.get("/status/{thread_id}")
def status_endpoint(thread_id: str):
    """API kiểm tra trạng thái của một thread (đã được duyệt hay chưa)."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent_app.get_state(config)
        if state.next and "process_hitl_node" in state.next:
            return {"status": "pending"}
        if state.values and "final_answer" in state.values:
            return {"status": "done", "answer": state.values["final_answer"]}
        return {"status": "error", "message": "Chưa có kết quả."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/resume")
def resume_endpoint(req: ResumeRequest):
    """
    Endpoint cho Admin duyệt (approve), sửa (edit) hoặc từ chối (reject) kết quả nháp.
    Thực thi tiếp LangGraph và stream SSE.
    """
    config = {"configurable": {"thread_id": req.thread_id}}
    state = agent_app.get_state(config)
    
    if not state.next or "process_hitl_node" not in state.next:
        return {"status": "error", "message": "Thread không ở trạng thái chờ duyệt (pending)."}
        
    update_data = {"hitl_action": req.action}
    if req.action in ["edit", "approve"] and req.edited_content is not None:
        update_data["draft_answer"] = req.edited_content
        
    # Cập nhật trạng thái mới cho checkpointer
    agent_app.update_state(config, update_data)
    
    # Thực thi phần còn lại của graph đồng bộ (không dùng stream)
    # Tránh tình trạng Client (Trình duyệt) ngắt kết nối giữa chừng làm sập graph
    try:
        agent_app.invoke(None, config=config)
        return {"status": "success", "message": "Đã xử lý luồng thành công."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Cung cấp giao diện Web tĩnh từ FastAPI
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def chat_ui():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/admin")
def admin_ui():
    return FileResponse(os.path.join(static_dir, "admin.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
