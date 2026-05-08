import os
import sqlite3
import json
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Endpoint Chat.
    Thực thi LangGraph và stream SSE. Nếu bị ngắt bởi HITL, trả về status="pending".
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    def event_generator():
        try:
            for event in agent_app.stream({"question": req.question}, config=config):
                state = agent_app.get_state(config)
                # Kiểm tra nếu đồ thị đang bị đóng băng tại process_hitl_node
                if state.next and "process_hitl_node" in state.next:
                    yield f"data: {json.dumps({'status': 'pending', 'message': 'Đang chờ Admin duyệt kết quả tìm kiếm ngoài (HITL)...', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
                    break
                    
                for node, update in event.items():
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
    
    def event_generator():
        try:
            # Truyền None để resume từ điểm bị interrupt
            for event in agent_app.stream(None, config=config):
                for node, update in event.items():
                    if "final_answer" in update:
                        yield f"data: {json.dumps({'status': 'done', 'answer': update['final_answer'], 'thread_id': req.thread_id}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
