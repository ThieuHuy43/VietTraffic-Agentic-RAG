from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    question: str
    intent: str
    expanded_query: Optional[str]
    # Nếu câu hỏi hỏi nhiều vi phạm/khía cạnh pháp lý riêng biệt cùng lúc (VD: tính tổng mức phạt
    # nhiều lỗi), router_node tách mỗi cái thành 1 câu hỏi độc lập ở đây để retrieve_node truy
    # riêng từng cái (tránh gộp chung làm loãng embedding, cùng vấn đề đã gặp với expanded_query).
    sub_queries: List[str]
    chunks: List[Dict[str, Any]]
    is_relevant: bool
    draft_answer: Optional[str]
    final_answer: str
    hitl_action: Optional[str]
    # Lịch sử hội thoại (tối đa vài turn gần nhất) để expand_query có thể chuẩn hóa câu hỏi
    # follow-up ("còn xe máy thì sao?") thành câu hỏi độc lập trước khi retrieve/generate.
    chat_history: List[Dict[str, str]]
