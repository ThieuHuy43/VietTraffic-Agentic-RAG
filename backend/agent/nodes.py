import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
import sys

# Thêm đường dẫn để import custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever.qdrant_retriever import QdrantHybridRetriever

LLM_PROVIDER = "unknown"
MAX_GRADE_CONTEXT_CHARS = 8000
MAX_GENERATE_CONTEXT_CHARS = 14000
MAX_WEB_CONTEXT_CHARS = 6000
MAX_CHUNK_CHARS = 3500

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))

def debug_llm_prompt(node_name: str, prompt: str):
    print("\n" + "=" * 50, flush=True)
    print(f"[DEBUG LLM] Node: {node_name}", flush=True)
    print(f"[DEBUG LLM] Provider: {LLM_PROVIDER}", flush=True)
    print(f"[DEBUG LLM] Estimated prompt tokens: ~{estimate_tokens(prompt)}", flush=True)
    print(f"[DEBUG LLM] Words: {len(prompt.split())}", flush=True)
    print(f"[DEBUG LLM] Chars: {len(prompt)}", flush=True)
    print("=" * 50 + "\n", flush=True)

def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated}\n...[đã rút gọn]"

def build_limited_context(items, formatter, max_chars: int) -> str:
    parts = []
    total = 0
    for item in items:
        part = formatter(item)
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(part) > remaining:
            part = truncate_text(part, remaining)
        parts.append(part)
        total += len(part) + 1
    return "\n".join(parts)

def preferred_doc_types_for_question(question: str):
    q = (question or "").lower()
    penalty_keywords = ["phạt", "mức phạt", "bao nhiêu tiền", "lỗi", "giam xe", "tạm giữ", "tước bằng"]
    technical_keywords = ["biển báo", "vạch kẻ", "tốc độ", "đèn tín hiệu", "bằng b2", "giấy phép lái xe", "qcvn"]
    law_keywords = ["được phép", "quy tắc", "độ tuổi", "trách nhiệm", "quyền", "nghĩa vụ"]

    if any(k in q for k in penalty_keywords):
        return ["nghi_dinh"]
    if any(k in q for k in technical_keywords):
        return ["quy_chuan", "thong_tu"]
    if any(k in q for k in law_keywords):
        return ["luat"]
    return None

# Khởi tạo mô hình
if os.getenv("DEEPSEEK_API_KEY"):
    LLM_PROVIDER = "deepseek"
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0
    )
    print("Using LLM: DeepSeek (deepseek-v4-flash)", flush=True)
elif os.getenv("GROQ_API_KEY"):
    LLM_PROVIDER = "groq"
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=2048,
        temperature=0
    )
    print("Using LLM: Groq (llama-3.3-70b-versatile)", flush=True)
elif os.getenv("GEMINI_API_KEY"):
    LLM_PROVIDER = "gemini"
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    print("Using LLM: Google Gemini", flush=True)
else:
    raise ValueError("Vui lòng cung cấp ít nhất một API Key: DEEPSEEK_API_KEY, GROQ_API_KEY hoặc GEMINI_API_KEY")

def router_node(state):
    """Phân loại intent của câu hỏi."""
    prompt = f"""Phân loại câu hỏi của người dùng vào 1 trong 2 loại: 'hoi_luat', 'chao_hoi'.
Câu hỏi: {state.get('question')}
Trả lời chỉ 1 từ duy nhất:"""
    debug_llm_prompt("router_node", prompt)
    res = llm.invoke(prompt)
    intent = res.content.strip().lower()
    if 'chao_hoi' in intent:
        return {"intent": "chao_hoi"}
    return {"intent": "hoi_luat"}

def greeting_node(state):
    return {"final_answer": "Chào bạn, tôi là Trợ lý Pháp luật Giao thông Việt Nam. Tôi có thể giúp gì cho bạn?"}

def retrieve_node(state):
    """Truy xuất tài liệu từ Qdrant (Hybrid + RRF)."""
    question = state.get("question", "")
    retriever = QdrantHybridRetriever(top_k=8)
    chunks = retriever.retrieve(question, preferred_doc_types_for_question(question))
    return {"chunks": chunks}

def grade_node(state):
    """Đánh giá xem tài liệu có liên quan và đủ để trả lời hay không."""
    chunks = state.get("chunks", [])
    if not chunks:
         return {"is_relevant": False}
         
    context = build_limited_context(
        chunks,
        lambda c: f"[{c.get('citation', c.get('doc_id', 'unknown'))}] {truncate_text(c['text'], MAX_CHUNK_CHARS)}",
        MAX_GRADE_CONTEXT_CHARS
    )
    prompt = f"""Đánh giá xem tài liệu sau có liên quan và đủ để trả lời câu hỏi không.
Câu hỏi: {state.get('question')}
Tài liệu: {context}
Trả lời (yes/no):"""
    debug_llm_prompt("grade_node", prompt)
    res = llm.invoke(prompt)
    if "yes" in res.content.lower():
         return {"is_relevant": True}
    return {"is_relevant": False}

def web_search_node(state):
    """Sử dụng Tavily để tìm kiếm ngoài khi tài liệu Qdrant không đủ."""
    try:
        from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
        
        # Giới hạn các nguồn uy tín
        legal_domains = [
            "thuvienphapluat.vn", 
            "luatvietnam.vn", 
            "chinhphu.vn", 
            "bocongan.gov.vn", 
            "csgt.vn", 
            "mt.gov.vn",
            "xaydungchinhsach.chinhphu.vn"
        ]
        
        api_wrapper = TavilySearchAPIWrapper()
        raw_response = api_wrapper.raw_results(
            state.get("question", ""),
            max_results=3,
            include_domains=legal_domains
        )
        
        results = raw_response if isinstance(raw_response, list) else raw_response.get("results", [])
        
        def format_web_result(r):
            title = r.get("title", "Không rõ tiêu đề")
            url = r.get("url", "Không rõ URL")
            content = truncate_text(r.get("content", ""), 2000)
            return f"[Nguồn Web: {title} | URL: {url}]\nNội dung: {content}"
            
        context = build_limited_context(
            results,
            format_web_result,
            MAX_WEB_CONTEXT_CHARS
        )
        
        # Nháp câu trả lời dựa trên web search để chờ duyệt
        prompt = f"""Dựa vào thông tin từ Web Search (Các nguồn pháp lý chính thống):
{context}

Yêu cầu bắt buộc:
- Ngay cả khi lấy dữ liệu từ Web, bạn vẫn BẮT BUỘC phải trích dẫn rõ tên Luật, Nghị định, Điều, Khoản mà bài báo/website nhắc đến (ví dụ: [Theo Luật Giao thông đường bộ 2008, Điều 5, Khoản 1]).
- BẮT BUỘC chèn Link URL nguồn tham khảo ở cuối câu trả lời theo cấu trúc: [Nguồn tham khảo](URL) để người dùng có thể tự bấm vào đọc.
- Tuyệt đối không trả lời suông mà không có nguồn gốc pháp lý rõ ràng.
- Nếu bài viết từ Web không nhắc đến tên Điều/Luật cụ thể, hãy ghi chú thêm: "Tuy nhiên, bài viết/nguồn mạng không trích dẫn cụ thể căn cứ pháp lý".

Câu hỏi: {state.get('question')}"""
        debug_llm_prompt("web_search_node", prompt)
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
        
        # Rule 1: Bắt chặn luật hết hiệu lực ngay lập tức dù không lấy được bản thay thế
        if c.get("status") == "superseded":
            note = f"[CẢNH BÁO ĐỎ: Nguồn '{doc_id}' ĐÃ HẾT HIỆU LỰC (superseded). Tuyệt đối thận trọng khi tư vấn. Yêu cầu báo cho người dùng biết điều này nếu phải sử dụng nó.]"
            if note not in conflict_notes:
                conflict_notes.append(note)
                
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
            
    # Tạo context chuẩn: mỗi đoạn có citation machine-readable để LLM trích dẫn đúng.
    def format_chunk(c):
        citation = c.get("citation")
        if not citation:
            citation_parts = [c.get("doc_id", "unknown")]
            if c.get("dieu"):
                citation_parts.append(f"Điều {c['dieu']}")
            if c.get("khoan"):
                citation_parts.append(f"Khoản {c['khoan']}")
            citation = ", ".join(citation_parts)

        title = c.get("dieu_title") or c.get("chuong_title") or ""
        text = truncate_text(c["text"], MAX_CHUNK_CHARS)
        return f"Nguồn: [{citation}]\nTiêu đề: {title}\nNội dung: {text}"

    context_str = build_limited_context(
        valid_chunks,
        format_chunk,
        MAX_GENERATE_CONTEXT_CHARS
    )
    notes_str = "\n".join(conflict_notes)
    
    prompt = f"""Bạn là Trợ lý Pháp luật Giao thông VN.
Chỉ được trả lời dựa trên tài liệu trong Context bên dưới. Không dùng kiến thức ngoài Context.
Nếu Context không có đủ căn cứ, hãy nói rõ: "Không tìm thấy thông tin đủ chắc chắn trong CSDL luật hiện có."

{notes_str}
Context:
{context_str}

Yêu cầu bắt buộc:
- Mỗi ý pháp lý hoặc mức phạt phải kết thúc bằng citation đúng nguyên văn từ dòng "Nguồn", ví dụ [NghiDinh_100_2019, Điều 5, Khoản 1].
- Không trích dẫn nguồn không xuất hiện trong Context.
- Trả lời súc tích, ưu tiên thông tin cốt lõi, không chép dài nguyên văn điều luật nếu không cần.

Câu hỏi: {state.get('question')}
"""
    debug_llm_prompt("generate_node", prompt)

    res = llm.invoke(prompt)
    return {"final_answer": res.content}
