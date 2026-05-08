import os
import streamlit as st
import requests
import json
import sseclient
import uuid
import re

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Trợ lý PL Giao thông VN", page_icon="🚦", layout="wide")
st.title("🚦 Trợ lý Pháp luật Giao thông Việt Nam")

# Chia 2 Tab
tab1, tab2 = st.tabs(["💬 Chat UI", "🛡️ Admin (HITL)"])

# ====== TAB 1: CHAT UI ======
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # Hiển thị tin nhắn cũ
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Hỏi tôi về luật giao thông..."):
        # Thêm tin nhắn user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Gửi request có stream
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": prompt, "thread_id": st.session_state.thread_id},
                    stream=True
                )
                
                client = sseclient.SSEClient(response)
                for event in client.events():
                    if event.data:
                        data = json.loads(event.data)
                        if data.get("status") == "pending":
                            full_response = f"⏳ *{data['message']}*"
                            message_placeholder.markdown(full_response)
                        elif data.get("status") == "done":
                            ans = data["answer"]
                            # Xử lý citation in nghiêng. Bắt các block [...] không phải là Markdown link.
                            formatted_answer = re.sub(r'(?<!\!)\[(.*?)\](?!\()', r'*[\1]*', ans)
                            full_response = formatted_answer
                            message_placeholder.markdown(full_response)
                        elif data.get("status") == "error":
                            full_response = f"❌ Lỗi: {data['message']}"
                            message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Lỗi kết nối Backend: {e}"
                message_placeholder.markdown(full_response)
                
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# ====== TAB 2: ADMIN UI ======
with tab2:
    st.subheader("Danh sách yêu cầu chờ duyệt (HITL)")
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
                    # Giao diện Expander cho từng pending thread
                    with st.expander(f"Mã Thread: {p['thread_id']} | Câu hỏi: {p['question'][:40]}..."):
                        st.markdown(f"**Câu hỏi gốc:** {p['question']}")
                        
                        draft = p.get('draft_answer', '')
                        edited_content = st.text_area(
                            "Câu trả lời nháp từ Web Search (Có thể sửa trực tiếp):", 
                            value=draft, 
                            key=f"text_{p['thread_id']}", 
                            height=200
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        
                        def process_resume(action: str, content: str = None):
                            payload = {"thread_id": p["thread_id"], "action": action}
                            if content:
                                payload["edited_content"] = content
                            
                            # Cần tiêu thụ stream từ resume để nhận final_answer
                            r = requests.post(f"{BACKEND_URL}/resume", json=payload, stream=True)
                            final_ans = ""
                            try:
                                sse = sseclient.SSEClient(r)
                                for ev in sse.events():
                                    if ev.data:
                                        d = json.loads(ev.data)
                                        if d.get("status") == "done":
                                            final_ans = d["answer"]
                                        elif d.get("status") == "error":
                                            st.error(d["message"])
                            except Exception as e:
                                st.error(str(e))
                                
                            # Nếu cùng chung session_state, đẩy ngược vào chat UI
                            if final_ans and st.session_state.thread_id == p["thread_id"]:
                                fmt_ans = re.sub(r'(?<!\!)\[(.*?)\](?!\()', r'*[\1]*', final_ans)
                                st.session_state.messages.append({"role": "assistant", "content": fmt_ans})
                                
                            st.rerun()

                        with col1:
                            if st.button("✅ Duyệt (Nguyên bản)", key=f"app_{p['thread_id']}", use_container_width=True):
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
        st.warning("Không thể kết nối đến Backend. Hãy chắc chắn Backend đang chạy.")
