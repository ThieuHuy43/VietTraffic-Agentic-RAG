import concurrent.futures
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Literal
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Thêm đường dẫn để import custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever.qdrant_retriever import QdrantHybridRetriever

LLM_PROVIDER = "unknown"
# Với top_k=8 sau rerank, 1 chunk (đặc biệt chunk bảng) có thể dài tới MAX_CHUNK_CHARS=3500 ->
# 8 chunk có thể cần tới ~28000 ký tự để không bị cắt. Ngân sách 10000 trước đây (khi chưa có
# retrieve_multi) khiến thứ tự rerank quyết định chunk nào "lọt" vào context: đã xác minh thực
# tế 1 chunk đúng (QCVN_41_2019 Điều 16 Bảng) qua được grade_node (do có chunk khác cũng liên
# quan) nhưng bị cắt khỏi context của generate_node vì xếp hạng thấp hơn budget cho phép, khiến
# generate_node báo "không tìm thấy thông tin" dù dữ liệu đúng đã có trong top_k. Nâng ngân sách
# để giảm phụ thuộc thứ tự, chấp nhận đánh đổi độ trễ (đã có reranker/retrieve_multi lọc chunk
# tốt hơn nên ít rủi ro tăng token lãng phí so với giai đoạn top_k lớn trước khi có reranker).
MAX_GRADE_CONTEXT_CHARS = 16000
MAX_GENERATE_CONTEXT_CHARS = 16000
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

_llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=12)

# Speculative retrieval: router_node submit truy vấn Qdrant cho câu hỏi GỐC ngay khi bắt đầu
# (song song với lượt gọi LLM của chính nó), lưu future theo thread_id để retrieve_node lấy lại
# kết quả (đã fetch xong hoặc gần xong, vì router_node luôn mất 2-10s+) thay vì phải tự fetch từ
# đầu. Có lock vì nhiều request (thread_id khác nhau) có thể chạy đồng thời qua FastAPI threadpool.
_speculative_futures: Dict[str, "concurrent.futures.Future"] = {}
_speculative_lock = threading.Lock()

def invoke_with_retry(fn, retries: int = 2, base_delay: float = 1.0, timeout: float = 25.0):
    """Gọi fn() (LLM invoke, tool call...) với retry + backoff cho lỗi thoáng qua (rate limit,
    network), VÀ với timeout cứng cho mỗi lần thử.

    Đã đo thực tế: DeepSeek đôi khi mất >100s cho MỘT lượt gọi đơn lẻ mà KHÔNG ném exception (chỉ
    đơn thuần rất chậm) — trước đây trường hợp này không được retry vì fn() vẫn "thành công", chỉ
    là chậm. Chạy fn() trong thread riêng (qua _llm_executor) và giới hạn thời gian chờ bằng
    future.result(timeout=...); hết giờ thì coi như thất bại và thử lại — thread cũ vẫn chạy ngầm
    tới khi xong rồi tự bỏ kết quả, không ảnh hưởng tới request hiện tại. Biến 1 lần "ăn may" rất
    chậm, không giới hạn trên, thành tối đa (retries+1) x timeout giây."""
    last_exc = None
    for attempt in range(retries + 1):
        future = _llm_executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            last_exc = e
            if attempt < retries:
                print(f"[WARN] Lời gọi vượt quá {timeout}s (lần {attempt + 1}/{retries + 1}), thử lại...", flush=True)
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
    expanded_query: str
    # Nếu câu hỏi hỏi về NHIỀU hành vi vi phạm/khía cạnh pháp lý riêng biệt cùng lúc (VD: "vượt
    # đèn đỏ VÀ không đội mũ bảo hiểm VÀ không mang bằng lái, tổng phạt bao nhiêu?"), liệt kê mỗi
    # hành vi thành 1 câu hỏi độc lập ở đây để retrieve_node truy riêng từng cái - tránh gộp
    # chung 1 câu hỏi dài làm loãng embedding (cùng vấn đề đã xác minh với expand_query trước
    # đây). Rỗng nếu câu hỏi chỉ hỏi về 1 vấn đề duy nhất.
    sub_queries: List[str] = []

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

def format_chat_history(chat_history: list, max_turns: int = 3) -> str:
    if not chat_history:
        return ""
    turns = []
    for turn in chat_history[-max_turns:]:
        turns.append(f"Người dùng: {turn.get('question', '')}\nTrợ lý: {turn.get('answer', '')}")
    return "Lịch sử hội thoại gần đây:\n" + "\n\n".join(turns) + "\n\n"

def router_node(state, config=None):
    """Phân loại intent CÙNG LÚC với diễn giải câu hỏi (chuẩn hoá follow-up + thuật ngữ pháp lý)
    trong 1 lượt gọi LLM duy nhất — trước đây đây là 2 lượt riêng (router_node cũ chỉ phân loại,
    rồi expand_query() gọi lại bên trong retrieve_node). Gộp lại giảm 1/4 số lượt gọi LLM tuần
    tự mỗi request, đồng thời giảm 1 điểm rủi ro "trúng" đúng lúc DeepSeek bị chậm bất thường
    (đã đo thực tế 1 lượt gọi đơn lẻ có thể mất 125s+ dù không có lỗi).

    Speculative retrieval: submit truy vấn Qdrant cho câu hỏi GỐC ngay tại đây (chạy nền song
    song với lượt gọi LLM bên dưới, không đợi LLM xong) — đã đo thực tế retrieve_node (thuần
    CPU/Qdrant, không LLM) luôn ổn định ~5s trong khi router_node mất 2-10s+, nên phần lớn thời
    gian retrieve câu hỏi gốc có thể chồng lấp vào lúc LLM đang chạy thay vì đợi tuần tự."""
    question = state.get("question", "")
    chat_history = state.get("chat_history", [])

    # Fast-path: lời chào ngắn, rõ ràng thì bỏ qua lượt gọi LLM (và bỏ qua cả speculative fetch -
    # nhánh chao_hoi không bao giờ tới retrieve_node nên submit sẽ lãng phí).
    if is_greeting(question):
        return {"intent": "chao_hoi", "expanded_query": question, "sub_queries": []}

    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    if thread_id:
        future = _llm_executor.submit(
            get_retriever().fetch_candidates, question, preferred_doc_types_for_question(question)
        )
        with _speculative_lock:
            _speculative_futures[thread_id] = future

    history_str = format_chat_history(chat_history)
    prompt = f"""{history_str}Phân loại câu hỏi mới nhất của người dùng vào 1 trong 2 loại:
'hoi_luat' (hỏi về luật giao thông) hoặc 'chao_hoi' (chào hỏi/giao tiếp thông thường).

Đồng thời, nếu là 'hoi_luat': dựa vào lịch sử hội thoại ở trên (nếu có) để hiểu ngữ cảnh, viết lại
câu hỏi thành MỘT câu hỏi độc lập, đầy đủ ý nghĩa (không cần đọc lịch sử vẫn hiểu được), đồng thời
diễn giải sang thuật ngữ pháp lý giao thông đường bộ Việt Nam chính thức, đúng cách hành văn thường
dùng trong luật/nghị định (ví dụ: "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu giao
thông"; "kẹp 3" -> "chở quá số người quy định"). Nếu câu hỏi đã độc lập, chỉ cần diễn giải thuật
ngữ như bình thường. Nếu là 'chao_hoi', giữ nguyên câu hỏi gốc cho trường QUERY.

Nếu câu hỏi hỏi về NHIỀU hành vi vi phạm/khía cạnh pháp lý RIÊNG BIỆT cùng lúc (ví dụ: "tôi vượt
đèn đỏ và không đội mũ bảo hiểm và không mang bằng lái, tổng phạt bao nhiêu?"), liệt kê MỖI hành vi
thành 1 câu hỏi độc lập (đã diễn giải thuật ngữ pháp lý), phân cách bằng " | ", vào trường
SUBQUERIES. Nếu câu hỏi chỉ hỏi về 1 vấn đề duy nhất, để SUBQUERIES là "không có".

Câu hỏi mới nhất: {question}

Trả lời ĐÚNG theo định dạng sau, không thêm giải thích:
INTENT: <hoi_luat hoặc chao_hoi>
QUERY: <câu hỏi đã viết lại>
SUBQUERIES: <vi phạm 1 | vi phạm 2 | ... hoặc "không có">"""
    debug_llm_prompt("router_node", prompt)

    if _structured_output_supported:
        try:
            structured_llm = get_structured_llm(RouterOutput)
            result = structured_llm.invoke(prompt)  # thử 1 lần, không retry (lỗi không hỗ trợ là vĩnh viễn)
            return {
                "intent": result.intent,
                "expanded_query": result.expanded_query or question,
                "sub_queries": result.sub_queries,
            }
        except Exception as e:
            _mark_structured_output_unsupported()
            print(f"[WARN] Structured output thất bại ở router_node, fallback sang parse string: {e}", flush=True)

    res = invoke_with_retry(lambda: llm.invoke(prompt))
    content = res.content.strip()
    intent_match = re.search(r"INTENT:\s*(\w+)", content, re.IGNORECASE)
    query_match = re.search(r"QUERY:\s*(.+?)(?:\n+SUBQUERIES:|\Z)", content, re.IGNORECASE | re.DOTALL)
    subqueries_match = re.search(r"SUBQUERIES:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
    intent = "chao_hoi" if intent_match and "chao_hoi" in intent_match.group(1).lower() else "hoi_luat"
    expanded_query = query_match.group(1).strip() if query_match else question
    sub_queries = []
    if subqueries_match:
        raw = subqueries_match.group(1).strip()
        if "không có" not in raw.lower():
            sub_queries = [s.strip() for s in raw.split("|") if s.strip()]
    return {"intent": intent, "expanded_query": expanded_query or question, "sub_queries": sub_queries}

def greeting_node(state):
    return {"final_answer": "Chào bạn, tôi là Trợ lý Pháp luật Giao thông Việt Nam. Tôi có thể giúp gì cho bạn?"}

MAX_SUB_QUERIES = 4  # giới hạn số vi phạm/khía cạnh retrieve riêng, tránh cost tăng vô hạn

def retrieve_node(state, config=None):
    """Truy xuất tài liệu từ Qdrant (Hybrid + RRF). Truy riêng câu hỏi gốc + bản diễn giải (đã
    chuẩn hóa follow-up + thuật ngữ pháp lý, do router_node sinh ra) rồi gộp candidate pool,
    THAY VÌ nối chuỗi 2 câu thành 1 query dài — đã xác minh thực tế nối chuỗi có thể pha loãng
    dense embedding khiến chunk đúng bị rớt khỏi candidate pool dù truy riêng câu hỏi gốc vẫn
    tìm thấy đúng.

    Câu hỏi GỐC: lấy từ future do router_node submit trước đó (speculative retrieval - khả năng
    cao đã fetch xong vì router_node vừa mất 2-10s+ để gọi LLM); fallback tự fetch đồng bộ nếu
    không có future (resume sau HITL, hoặc gọi graph trực tiếp không qua config/thread_id).

    Câu hỏi hỏi NHIỀU vi phạm/khía cạnh cùng lúc (VD tính tổng mức phạt nhiều lỗi): router_node
    đã tách sẵn thành sub_queries, ở đây truy RIÊNG từng cái (cùng lý do tránh pha loãng embedding
    như trên) và nới top_k để đủ chỗ cho chunk của mỗi vi phạm thay vì chỉ top_k mặc định."""
    question = state.get("question", "")
    expanded = state.get("expanded_query") or question
    sub_queries = (state.get("sub_queries") or [])[:MAX_SUB_QUERIES]
    preferred_doc_types = preferred_doc_types_for_question(question)
    retriever = get_retriever()

    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    future = None
    if thread_id:
        with _speculative_lock:
            future = _speculative_futures.pop(thread_id, None)

    if future is not None:
        try:
            original_candidates = future.result(timeout=15)
        except Exception as e:
            print(f"[WARN] Speculative retrieval thất bại/timeout, fetch lại đồng bộ: {e}", flush=True)
            original_candidates = retriever.fetch_candidates(question, preferred_doc_types)
    else:
        original_candidates = retriever.fetch_candidates(question, preferred_doc_types)

    candidate_dicts = [original_candidates]
    if expanded.strip().lower() != question.strip().lower():
        candidate_dicts.append(retriever.fetch_candidates(expanded, preferred_doc_types))
    for sub_q in sub_queries:
        candidate_dicts.append(retriever.fetch_candidates(sub_q, preferred_doc_types))

    # Nhiều vi phạm -> cần nhiều chunk hơn để mỗi vi phạm đều có đủ căn cứ trong context, không
    # chỉ top_k mặc định (vốn tính cho 1 vấn đề duy nhất).
    top_k = min(20, 6 * len(sub_queries)) if len(sub_queries) > 1 else None
    chunks = retriever.rerank_merged(expanded, candidate_dicts, top_k=top_k)
    return {"chunks": chunks}

def effective_question(state) -> str:
    """Câu hỏi độc lập, đã chuẩn hóa (resolve follow-up + thuật ngữ pháp lý) để dùng cho các bước
    grade/generate/web_search — tránh dùng thẳng câu hỏi thô (VD: "Còn xe máy thì sao?") vốn không
    có nghĩa nếu tách khỏi lịch sử hội thoại."""
    return state.get("expanded_query") or state.get("question", "")

def _grade_once(prompt: str) -> bool:
    if _structured_output_supported:
        try:
            structured_llm = get_structured_llm(GradeOutput)
            result = structured_llm.invoke(prompt)  # thử 1 lần, không retry (lỗi không hỗ trợ là vĩnh viễn)
            return result.is_relevant
        except Exception as e:
            _mark_structured_output_unsupported()
            print(f"[WARN] Structured output thất bại ở grade_node, fallback sang parse string: {e}", flush=True)

    res = invoke_with_retry(lambda: llm.invoke(prompt))
    print(f"[DEBUG] grade_node raw response: {res.content!r}", flush=True)
    return "yes" in res.content.lower()

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

    # DeepSeek (và nhiều model MoE khác) không hoàn toàn deterministic dù temperature=0, nhất là
    # ở câu hỏi biên (borderline): đã xác minh thực tế cùng 1 context ~16000 ký tự (không đổi
    # giữa các lần gọi) nhưng verdict đổi qua lại yes/no giữa 5 lần gọi liên tiếp. Chạy 2 lượt
    # self-consistency SONG SONG (độc lập, cùng prompt) qua _llm_executor thay vì tuần tự — lấy
    # kết quả "yes" đầu tiên (short-circuit), không cần đợi lượt còn lại. Giữ nguyên lợi ích giảm
    # rủi ro rơi oan vào HITL/web_search vì 1 lần LLM đoán sai, nhưng KHÔNG cộng dồn latency của
    # 2 lượt gọi tuần tự (trước đây lượt 2 chỉ bắt đầu sau khi lượt 1 xong).
    futures = [_llm_executor.submit(_grade_once, prompt) for _ in range(2)]
    is_relevant = False
    for future in concurrent.futures.as_completed(futures):
        try:
            if future.result():
                is_relevant = True
                break
        except Exception as e:
            print(f"[WARN] grade_node self-consistency 1 nhánh lỗi: {e}", flush=True)
    return {"is_relevant": is_relevant}

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

    sub_queries = state.get("sub_queries") or []
    multi_violation_instruction = ""
    if len(sub_queries) > 1:
        violations_list = "\n".join(f"- {v}" for v in sub_queries)
        multi_violation_instruction = f"""
Câu hỏi này hỏi về NHIỀU hành vi vi phạm cùng lúc:
{violations_list}
Trả lời theo dạng DANH SÁCH, mỗi hành vi 1 mục riêng kèm mức phạt + citation riêng (nếu Context có
đủ căn cứ cho hành vi đó; nếu thiếu thì ghi rõ "không tìm thấy thông tin cho hành vi này" thay vì
bỏ qua). Sau đó thêm mục "**Tổng cộng**": cộng các khoảng tiền phạt tìm được lại (VD 200.000-
300.000đ + 3.000.000-5.000.000đ = 3.200.000-5.300.000đ), và nêu rõ đây là tổng ước tính từ các mức
phạt riêng lẻ, có thể còn hình phạt bổ sung (tước GPLX, tạm giữ phương tiện...) nếu Context có đề
cập.
"""

    prompt = f"""Bạn là Trợ lý Pháp luật Giao thông VN.
Chỉ được trả lời dựa trên tài liệu trong Context bên dưới. Không dùng kiến thức ngoài Context.
Nếu Context không có đủ căn cứ, hãy nói rõ: "Không tìm thấy thông tin đủ chắc chắn trong CSDL luật hiện có."
{multi_violation_instruction}
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

    # Dùng llm.stream() (thay vì .invoke()) để LangGraph có thể phát từng token qua
    # stream_mode="messages" ở tầng graph.stream() trong main.py, cho phép FastAPI forward token
    # thật ra SSE ngay khi sinh ra thay vì đợi toàn bộ câu trả lời xong mới trả về. Không bọc
    # invoke_with_retry (retry giữa chừng 1 stream đã gửi 1 phần token ra client là vô nghĩa);
    # nếu stream lỗi ngay từ đầu (trước khi có token nào), fallback về gọi invoke thường.
    try:
        answer = "".join(chunk.content for chunk in llm.stream(prompt))
    except Exception as e:
        print(f"[WARN] generate_node streaming thất bại, fallback sang invoke thường: {e}", flush=True)
        answer = invoke_with_retry(lambda: llm.invoke(prompt)).content
    return {"final_answer": answer, "chat_history": append_chat_history(state, answer)}
