from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    question: str
    intent: str
    expanded_query: Optional[str]
    chunks: List[Dict[str, Any]]
    is_relevant: bool
    draft_answer: Optional[str]
    final_answer: str
    hitl_action: Optional[str]
    # Lịch sử hội thoại (tối đa vài turn gần nhất) để expand_query có thể chuẩn hóa câu hỏi
    # follow-up ("còn xe máy thì sao?") thành câu hỏi độc lập trước khi retrieve/generate.
    chat_history: List[Dict[str, str]]
