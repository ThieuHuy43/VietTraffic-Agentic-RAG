import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
import sys

# Thêm đường dẫn để import custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever.qdrant_retriever import QdrantHybridRetriever

# Khởi tạo mô hình
if os.getenv("DEEPSEEK_API_KEY"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="deepseek-v4-flash", 
        api_key=os.getenv("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com/v1",
        temperature=0
    )
    print("Using LLM: DeepSeek (deepseek-chat)", flush=True)
elif os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    print("Using LLM: Groq (llama3-70b-8192)", flush=True)
elif os.getenv("GEMINI_API_KEY"):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    print("Using LLM: Google Gemini", flush=True)
else:
    raise ValueError("Vui lòng cung cấp ít nhất một API Key: DEEPSEEK_API_KEY, GROQ_API_KEY hoặc GEMINI_API_KEY")

def router_node(state):
    """Phân loại intent của câu hỏi."""
    prompt = f"""Phân loại câu hỏi của người dùng vào 1 trong 2 loại: 'hoi_luat', 'chao_hoi'.
Câu hỏi: {state.get('question')}
Trả lời chỉ 1 từ duy nhất:"""
    res = llm.invoke(prompt)
    intent = res.content.strip().lower()
    if 'chao_hoi' in intent:
        return {"intent": "chao_hoi"}
    return {"intent": "hoi_luat"}

def greeting_node(state):
    return {"final_answer": "Chào bạn, tôi là Trợ lý Pháp luật Giao thông Việt Nam. Tôi có thể giúp gì cho bạn?"}

def retrieve_node(state):
    """Truy xuất tài liệu từ Qdrant (Hybrid + RRF)."""
    retriever = QdrantHybridRetriever(top_k=2)
    chunks = retriever.retrieve(state.get("question", ""))
    return {"chunks": chunks}

def grade_node(state):
    """Đánh giá xem tài liệu có liên quan và đủ để trả lời hay không."""
    chunks = state.get("chunks", [])
    if not chunks:
         return {"is_relevant": False}
         
    context = "\\n".join([c["text"] for c in chunks])
    prompt = f"""Đánh giá xem tài liệu sau có liên quan và đủ để trả lời câu hỏi không.
Câu hỏi: {state.get('question')}
Tài liệu: {context}
Trả lời (yes/no):"""
    res = llm.invoke(prompt)
    if "yes" in res.content.lower():
         return {"is_relevant": True}
    return {"is_relevant": False}

def web_search_node(state):
    """Sử dụng Tavily để tìm kiếm ngoài khi tài liệu Qdrant không đủ."""
    try:
        search = TavilySearchResults(max_results=3)
        results = search.invoke(state.get("question", ""))
        context = "\\n".join([r["content"] for r in results])
        
        # Nháp câu trả lời dựa trên web search để chờ duyệt
        prompt = f"""Dựa vào thông tin web: {context}
Trả lời câu hỏi: {state.get('question')}"""
        res = llm.invoke(prompt)
        return {"draft_answer": res.content}
    except Exception as e:
        return {"draft_answer": f"Lỗi gọi Tavily API: {str(e)}"}

def process_hitl_node(state):
    """Node xử lý trung gian sau khi Admin can thiệp."""
    return {}

def fallback_node(state):
    """Trả về câu trả lời mặc định nếu bị Admin từ chối."""
    return {"final_answer": "Rất tiếc, tôi không tìm thấy thông tin trong CSDL luật và yêu cầu lấy dữ liệu ngoài đã bị từ chối. Vui lòng hỏi câu khác cụ thể hơn."}

def generate_node(state):
    """Sinh câu trả lời dựa trên tài liệu pháp luật (có xử lý xung đột effective_date) hoặc kết quả HITL."""
    
    # Nếu là luồng đi từ HITL (đã duyệt hoặc sửa)
    if state.get("hitl_action") in ["approve", "edit"]:
         return {"final_answer": state.get("draft_answer", "")}
         
    chunks = state.get("chunks", [])
    
    # Xử lý Conflict Resolution (Ưu tiên văn bản mới nếu có văn bản thay thế trong chunks)
    active_docs = {}
    superseded_by = {}
    
    for c in chunks:
        doc_id = c["doc_id"]
        if doc_id not in active_docs:
            active_docs[doc_id] = []
        active_docs[doc_id].append(c)
        if c.get("supersedes"):
            superseded_by[c["supersedes"]] = c
            
    valid_chunks = []
    conflict_notes = []
    
    for doc_id, docs in active_docs.items():
        if doc_id in superseded_by:
            new_doc = superseded_by[doc_id]
            note = f"(Lưu ý: Theo truy xuất, '{doc_id}' đã bị thay thế bởi '{new_doc['doc_id']}' - có hiệu lực từ {new_doc.get('effective_date', 'unknown')})"
            if note not in conflict_notes:
                conflict_notes.append(note)
            # Bỏ qua không nạp docs cũ vào context
        else:
            valid_chunks.extend(docs)
            
    # Tạo context chuẩn
    context_list = []
    for c in valid_chunks:
        chuong = f"Chương {c['chuong']}" if c.get('chuong') else ""
        dieu = f"Điều {c['dieu']}" if c.get('dieu') else ""
        khoan = f"Khoản {c['khoan']}" if c.get('khoan') else ""
        prefix = f"[{c['doc_id']} {chuong} {dieu} {khoan}]".strip()
        context_list.append(f"{prefix}: {c['text']}")
        
    context_str = "\\n".join(context_list)
    notes_str = "\\n".join(conflict_notes)
    
    prompt = f"""Bạn là Trợ lý Pháp luật Giao thông VN.
Trả lời câu hỏi của người dùng dựa trên tài liệu pháp luật sau đây.

{notes_str}
Tài liệu:
{context_str}

Yêu cầu bắt buộc: Trích dẫn rõ ràng [Tên văn bản, Điều X, Khoản Y] vào cuối mỗi lập luận.

Câu hỏi: {state.get('question')}
"""
    # -- ĐOẠN CODE CHÈN THÊM ĐỂ DEBUG --
    # Đếm ước lượng số ký tự và số từ
    char_count = len(prompt)
    word_count = len(prompt.split())
    # Khoảng 1 token ~ 0.75 từ (Rule of thumb)
    est_tokens = int(word_count / 0.75) 
    
    print("\n" + "="*50, flush=True)
    print("🚀 [DEBUG RAG] - NỘI DUNG PROMPT SẮP GỬI VÀO LLM:", flush=True)
    print("="*50, flush=True)
    print(f"Ước lượng Tokens : ~{est_tokens} tokens", flush=True)
    print(f"Số từ (Words)    : {word_count} words", flush=True)
    print(f"Số ký tự (Chars) : {char_count} chars", flush=True)
    print("-" * 50, flush=True)
    # Bỏ comment dòng dưới nếu muốn in toàn bộ cục văn bản luật ra xem:
    print(prompt, flush=True) 
    print("="*50 + "\n", flush=True)
    # -----------------------------------

    res = llm.invoke(prompt)
    return {"final_answer": res.content}
