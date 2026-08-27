import os
import re
import sys
import time
from typing import Literal
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Thêm đường dẫn để import custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever.qdrant_retriever import QdrantHybridRetriever

LLM_PROVIDER = "unknown"
MAX_GRADE_CONTEXT_CHARS = 10000
MAX_GENERATE_CONTEXT_CHARS = 10000
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

def invoke_with_retry(fn, retries: int = 2, base_delay: float = 1.0):
    """Gọi fn() (LLM invoke, tool call...) với retry + backoff cho lỗi thoáng qua (rate limit, network)."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                delay = base_delay * (2 ** attempt)
                print(f"[WARN] Lời gọi thất bại (lần {attempt + 1}/{retries + 1}): {e}. Retry sau {delay}s...", flush=True)
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc

_GREETING_WORDS = {"chào", "chao", "hi", "hii", "hello", "helo", "hey", "alo"}

def is_greeting(question: str) -> bool:
    """Fast-path rule-based: chỉ khớp các lời chào ngắn, rõ ràng để bỏ qua 1 lượt gọi LLM ở router_node."""
    q = (question or "").strip().lower().rstrip("!?.,")
    if not q:
        return False
    words = q.split()
    if len(words) > 4:
        return False
    return any(w in _GREETING_WORDS for w in words)

class RouterOutput(BaseModel):
    intent: Literal["hoi_luat", "chao_hoi"]

class GradeOutput(BaseModel):
    is_relevant: bool

_structured_llm_cache = {}
# Một số provider (VD: DeepSeek qua API tương thích OpenAI) hiện chưa hỗ trợ response_format/structured
# output -> lỗi này KHÔNG phải thoáng qua (transient), retry lại chỉ tốn thời gian vô ích mỗi request.
# Cờ này bật False ngay lần đầu phát hiện provider không hỗ trợ, để các lượt gọi sau bỏ qua hẳn nhánh
# structured output thay vì thử lại và luôn thất bại.
_structured_output_supported = True

def get_structured_llm(schema_cls):
    """Lazy cache cho các biến thể llm.with_structured_output(schema) theo từng schema."""
    key = schema_cls.__name__
    if key not in _structured_llm_cache:
        _structured_llm_cache[key] = llm.with_structured_output(schema_cls)
    return _structured_llm_cache[key]

def _mark_structured_output_unsupported():
    global _structured_output_supported
    _structured_output_supported = False

_retriever = None

def get_retriever():
    """Singleton retriever: tránh load lại SentenceTransformer + mở QdrantClient mới ở mỗi request."""
    global _retriever
    if _retriever is None:
        _retriever = QdrantHybridRetriever(top_k=8, candidate_k=25)
    return _retriever

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
elif os.getenv("GEMINI_API_KEY"):
    LLM_PROVIDER = "gemini"
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    print("Using LLM: Google Gemini", flush=True)
else:
    raise ValueError("Vui lòng cung cấp ít nhất một API Key: DEEPSEEK_API_KEY hoặc GEMINI_API_KEY")

def router_node(state):
    """Phân loại intent của câu hỏi."""
    question = state.get("question", "")

    # Fast-path: lời chào ngắn, rõ ràng thì bỏ qua lượt gọi LLM.
    if is_greeting(question):
        return {"intent": "chao_hoi"}

    prompt = f"""Phân loại câu hỏi của người dùng vào 1 trong 2 loại: 'hoi_luat', 'chao_hoi'.
Câu hỏi: {question}
Trả lời chỉ 1 từ duy nhất:"""
    debug_llm_prompt("router_node", prompt)

    if _structured_output_supported:
        try:
            structured_llm = get_structured_llm(RouterOutput)
            result = structured_llm.invoke(prompt)  # thử 1 lần, không retry (lỗi không hỗ trợ là vĩnh viễn)
            return {"intent": result.intent}
        except Exception as e:
            _mark_structured_output_unsupported()
            print(f"[WARN] Structured output thất bại ở router_node, fallback sang parse string: {e}", flush=True)

    res = invoke_with_retry(lambda: llm.invoke(prompt))
    intent = res.content.strip().lower()
    if 'chao_hoi' in intent:
        return {"intent": "chao_hoi"}
    return {"intent": "hoi_luat"}

def greeting_node(state):
    return {"final_answer": "Chào bạn, tôi là Trợ lý Pháp luật Giao thông Việt Nam. Tôi có thể giúp gì cho bạn?"}

def format_chat_history(chat_history: list, max_turns: int = 3) -> str:
    if not chat_history:
        return ""
    turns = []
    for turn in chat_history[-max_turns:]:
        turns.append(f"Người dùng: {turn.get('question', '')}\nTrợ lý: {turn.get('answer', '')}")
    return "Lịch sử hội thoại gần đây:\n" + "\n\n".join(turns) + "\n\n"

def expand_query(question: str, chat_history: list) -> str:
    """Diễn giải câu hỏi thông tục sang thuật ngữ pháp lý giao thông chính thức, vì văn bản luật
    thường dùng cách hành văn trang trọng khác hẳn cách người dùng hỏi (VD: "vượt đèn đỏ" -> "không
    chấp hành hiệu lệnh của đèn tín hiệu giao thông"), khiến cả dense lẫn sparse retrieval dễ bỏ sót
    đúng điều khoản nếu chỉ search bằng nguyên văn câu hỏi thông tục.

    Đồng thời dùng chat_history (nếu có) để viết lại câu hỏi follow-up ("còn xe máy thì sao?") thành
    câu hỏi độc lập, đầy đủ ngữ cảnh — tái dùng đúng 1 lượt gọi LLM này, không thêm round-trip riêng
    cho việc "nhớ" hội thoại, để tránh phình thêm latency."""
    history_str = format_chat_history(chat_history)
    prompt = f"""{history_str}Dựa vào lịch sử hội thoại ở trên (nếu có) để hiểu ngữ cảnh, hãy viết lại
Câu hỏi mới nhất của người dùng thành MỘT câu hỏi độc lập, đầy đủ ý nghĩa (không cần đọc lịch sử vẫn
hiểu được), đồng thời diễn giải sang thuật ngữ pháp lý giao thông đường bộ Việt Nam chính thức, đúng
cách hành văn thường dùng trong luật/nghị định (ví dụ: "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của
đèn tín hiệu giao thông"; "kẹp 3" -> "chở quá số người quy định").
Nếu Câu hỏi mới nhất đã độc lập, không phụ thuộc lịch sử, chỉ cần diễn giải thuật ngữ như bình thường.
Chỉ trả về DUY NHẤT câu hỏi đã viết lại, không thêm giải thích hay tiền tố nào khác.
Câu hỏi mới nhất: {question}
Câu hỏi viết lại:"""
    debug_llm_prompt("expand_query", prompt)
    try:
        res = invoke_with_retry(lambda: llm.invoke(prompt))
        expanded = res.content.strip()
        return expanded if expanded else question
    except Exception as e:
        print(f"[WARN] expand_query thất bại, dùng nguyên câu hỏi gốc: {e}", flush=True)
        return question

def retrieve_node(state):
    """Truy xuất tài liệu từ Qdrant (Hybrid + RRF). Kết hợp câu hỏi gốc + bản diễn giải (đã chuẩn hóa
    follow-up + thuật ngữ pháp lý) để tăng khả năng khớp cả theo nghĩa thông tục lẫn văn phong luật
    chính thức."""
    question = state.get("question", "")
    chat_history = state.get("chat_history", [])
    expanded = expand_query(question, chat_history)
    search_query = f"{question} {expanded}" if expanded.strip().lower() != question.strip().lower() else question
    chunks = get_retriever().retrieve(search_query, preferred_doc_types_for_question(question))
    return {"chunks": chunks, "expanded_query": expanded}

def effective_question(state) -> str:
    """Câu hỏi độc lập, đã chuẩn hóa (resolve follow-up + thuật ngữ pháp lý) để dùng cho các bước
    grade/generate/web_search — tránh dùng thẳng câu hỏi thô (VD: "Còn xe máy thì sao?") vốn không
    có nghĩa nếu tách khỏi lịch sử hội thoại."""
    return state.get("expanded_query") or state.get("question", "")

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
    prompt = f"""Đánh giá xem Tài liệu dưới đây có ĐỦ CĂN CỨ để trả lời Câu hỏi không.
Tài liệu gồm nhiều đoạn trích rời rạc từ nhiều Điều/Khoản khác nhau, có thể lẫn nhiều đoạn không liên quan.
CHỈ CẦN ít nhất MỘT đoạn trong đó nêu đúng thông tin để trả lời Câu hỏi là được coi là "yes",
kể cả khi các đoạn còn lại không liên quan hoặc tài liệu không đầy đủ ở khía cạnh khác.
Câu hỏi: {effective_question(state)}
Tài liệu: {context}
Trả lời (yes/no):"""
    debug_llm_prompt("grade_node", prompt)

    if _structured_output_supported:
        try:
            structured_llm = get_structured_llm(GradeOutput)
            result = structured_llm.invoke(prompt)  # thử 1 lần, không retry (lỗi không hỗ trợ là vĩnh viễn)
            return {"is_relevant": result.is_relevant}
        except Exception as e:
            _mark_structured_output_unsupported()
            print(f"[WARN] Structured output thất bại ở grade_node, fallback sang parse string: {e}", flush=True)

    res = invoke_with_retry(lambda: llm.invoke(prompt))
    print(f"[DEBUG] grade_node raw response: {res.content!r}", flush=True)
    return {"is_relevant": "yes" in res.content.lower()}

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
        
        search_query = effective_question(state)
        api_wrapper = TavilySearchAPIWrapper()
        raw_response = invoke_with_retry(lambda: api_wrapper.raw_results(
            search_query,
            max_results=3,
            include_domains=legal_domains
        ))
        
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

Câu hỏi: {search_query}"""
        debug_llm_prompt("web_search_node", prompt)
        res = invoke_with_retry(lambda: llm.invoke(prompt))
        return {"draft_answer": res.content}
    except Exception as e:
        return {"draft_answer": f"Lỗi gọi Tavily API: {str(e)}"}

def process_hitl_node(state):
    """Node xử lý trung gian sau khi Admin can thiệp."""
    return {}

def append_chat_history(state, answer: str, max_turns: int = 3) -> list:
    """Thêm turn hiện tại (câu hỏi gốc + câu trả lời) vào chat_history, giữ tối đa max_turns gần nhất
    để tránh phình vô hạn qua các lượt hỏi tiếp theo."""
    history = state.get("chat_history", [])
    new_history = history + [{"question": state.get("question", ""), "answer": answer}]
    return new_history[-max_turns:]

def fallback_node(state):
    """Trả về câu trả lời mặc định nếu bị Admin từ chối."""
    answer = "Rất tiếc, tôi không tìm thấy thông tin trong CSDL luật và yêu cầu lấy dữ liệu ngoài đã bị từ chối. Vui lòng hỏi câu khác cụ thể hơn."
    return {"final_answer": answer, "chat_history": append_chat_history(state, answer)}

def generate_node(state):
    """Sinh câu trả lời dựa trên tài liệu pháp luật (có xử lý xung đột effective_date) hoặc kết quả HITL."""

    # Nếu là luồng đi từ HITL (đã duyệt hoặc sửa)
    if state.get("hitl_action") in ["approve", "edit"]:
         answer = state.get("draft_answer", "")
         return {"final_answer": answer, "chat_history": append_chat_history(state, answer)}


    chunks = state.get("chunks", [])

    # Xử lý Conflict Resolution (Ưu tiên văn bản mới nếu có văn bản thay thế trong chunks)
    active_docs = {}
    superseded_by = {}
    amending_doc_by_target = {}  # base_doc_id -> doc_id sửa đổi/bổ sung nó (nếu cả hai cùng có trong chunks)
    conflict_notes = []

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
        if c.get("amends"):
            amending_doc_by_target[c["amends"]] = doc_id

    valid_chunks = []

    for doc_id, docs in active_docs.items():
        if doc_id in superseded_by:
            new_doc = superseded_by[doc_id]
            note = f"(Lưu ý: Theo truy xuất, '{doc_id}' đã bị thay thế bởi '{new_doc['doc_id']}' - có hiệu lực từ {new_doc.get('effective_date', 'unknown')})"
            if note not in conflict_notes:
                conflict_notes.append(note)
            # Bỏ qua không nạp docs cũ vào context
            continue

        valid_chunks.extend(docs)

        # Rule 1 (sửa đổi/bổ sung): văn bản amending KHÔNG thay thế toàn bộ base_doc,
        # nên phải giữ lại cả hai và yêu cầu LLM tổng hợp, ưu tiên bản sửa đổi khi có mâu thuẫn.
        if doc_id in amending_doc_by_target:
            amending_doc_id = amending_doc_by_target[doc_id]
            note = (
                f"[LƯU Ý QUAN TRỌNG: Văn bản '{amending_doc_id}' SỬA ĐỔI, BỔ SUNG một số điều của "
                f"'{doc_id}' (không thay thế toàn bộ). Phải đối chiếu và TỔNG HỢP nội dung của CẢ HAI "
                f"văn bản khi trả lời; nếu cùng một Điều/Khoản mà nội dung khác nhau, ưu tiên áp dụng "
                f"quy định của '{amending_doc_id}'.]"
            )
            if note not in conflict_notes:
                conflict_notes.append(note)
            
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

Câu hỏi: {effective_question(state)}
"""
    debug_llm_prompt("generate_node", prompt)

    res = invoke_with_retry(lambda: llm.invoke(prompt))
    answer = res.content
    return {"final_answer": answer, "chat_history": append_chat_history(state, answer)}
