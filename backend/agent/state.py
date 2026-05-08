from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    question: str
    intent: str
    chunks: List[Dict[str, Any]]
    is_relevant: bool
    draft_answer: Optional[str]
    final_answer: str
    hitl_action: Optional[str]
