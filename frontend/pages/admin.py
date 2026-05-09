import os
import streamlit as st
import requests
import json
import sseclient

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Admin Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Admin Dashboard (HITL)")

st.subheader("Danh sách yêu cầu chờ duyệt")
if st.button("Làm mới danh sách"):
    pass # Force rerun
    
try:
    res = requests.get(f"{BACKEND_URL}/pending")
    if res.status_code == 200:
        pending_list = res.json().get("pending", [])
        if not pending_list:
            st.info("Hiện không có yêu cầu nào đang chờ xử lý.")
        else:
            for p in pending_list:
                with st.expander(f"Mã Thread: {p['thread_id']} | Câu hỏi: {p['question'][:40]}..."):
                    st.markdown(f"**Câu hỏi gốc:** {p['question']}")
                    
                    draft = p.get('draft_answer', '')
                    edited_content = st.text_area(
                        "Câu trả lời nháp từ Web Search (Có thể sửa):", 
                        value=draft, 
                        key=f"text_{p['thread_id']}", 
                        height=200
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    def process_resume(action: str, content: str = None):
                        payload = {"thread_id": p["thread_id"], "action": action}
                        if content:
                            payload["edited_content"] = content
                        
                        try:
                            r = requests.post(f"{BACKEND_URL}/resume", json=payload, stream=True)
                            sse = sseclient.SSEClient(r)
                            for ev in sse.events():
                                pass
                            st.success("Xử lý thành công! (User đã có thể bấm Kiểm tra phản hồi ở Tab Chat)")
                        except Exception as e:
                            st.error(f"Lỗi kết nối API: {e}")

                    with col1:
                        if st.button("✅ Duyệt", key=f"app_{p['thread_id']}", use_container_width=True):
                            process_resume("approve", draft)
                            
                    with col2:
                        if st.button("✍️ Sửa & Gửi", key=f"edit_{p['thread_id']}", use_container_width=True):
                            process_resume("edit", edited_content)
                            
                    with col3:
                        if st.button("❌ Từ chối", type="primary", key=f"rej_{p['thread_id']}", use_container_width=True):
                            process_resume("reject")
                            
    else:
        st.error("Không thể lấy danh sách pending từ Backend.")
except Exception as e:
    st.warning("Không thể kết nối đến Backend.")
