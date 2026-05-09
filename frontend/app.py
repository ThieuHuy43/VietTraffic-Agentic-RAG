import os
import streamlit as st
import requests
import json
import sseclient
import uuid
import re

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Trợ lý PL Giao thông", page_icon="💬", layout="centered")
st.title("💬 Chatbot Pháp luật Giao thông")
st.caption("Hãy nhập câu hỏi. Admin có thể sử dụng giao diện Admin ở thanh bên trái (Sidebar).")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "is_pending" not in st.session_state:
    st.session_state.is_pending = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Cơ chế chủ động kiểm tra phản hồi từ Backend nếu đang Pending
if st.session_state.is_pending:
    st.warning("⏳ Hệ thống đang chờ Admin duyệt phản hồi. Vui lòng đợi...")
    if st.button("🔄 Kiểm tra phản hồi từ Admin"):
        try:
            res = requests.get(f"{BACKEND_URL}/status/{st.session_state.thread_id}")
            if res.status_code == 200:
                data = res.json()
                if data["status"] == "done":
                    st.session_state.is_pending = False
                    ans = data["answer"]
                    fmt_ans = re.sub(r'(?<!\!)\[(.*?)\](?!\()', r'*[\1]*', ans)
                    st.session_state.messages.append({"role": "assistant", "content": fmt_ans})
                    st.rerun()
                elif data["status"] == "pending":
                    st.info("Admin vẫn chưa duyệt. Bạn có thể thử lại sau vài giây.")
        except Exception as e:
            st.error(f"Lỗi hệ thống: {e}")

if not st.session_state.is_pending:
    if prompt := st.chat_input("Hỏi tôi về luật giao thông..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
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
                            st.session_state.is_pending = True
                            full_response = f"⏳ *{data['message']}*"
                            message_placeholder.markdown(full_response)
                        elif data.get("status") == "done":
                            ans = data["answer"]
                            formatted_answer = re.sub(r'(?<!\!)\[(.*?)\](?!\()', r'*[\1]*', ans)
                            full_response = formatted_answer
                            message_placeholder.markdown(full_response)
                        elif data.get("status") == "error":
                            full_response = f"❌ Lỗi: {data['message']}"
                            message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Lỗi kết nối Backend: {e}"
                message_placeholder.markdown(full_response)
                
            if not st.session_state.is_pending:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
