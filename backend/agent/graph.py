import os
import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from .state import AgentState
from .nodes import (
    router_node, greeting_node, retrieve_node, grade_node,
    web_search_node, process_hitl_node, fallback_node, generate_node
)

def check_intent(state: AgentState):
    if state.get("intent") == "chao_hoi":
        return "greeting_node"
    return "retrieve_node"

def check_relevance(state: AgentState):
    if state.get("is_relevant"):
        return "generate_node"
    return "web_search_node"

def hitl_router(state: AgentState):
    action = state.get("hitl_action")
    if action in ["reject", "timeout"]:
        return "fallback_node"
    # Các trường hợp approve, edit
    return "generate_node"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router_node", router_node)
    workflow.add_node("greeting_node", greeting_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grade_node", grade_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("process_hitl_node", process_hitl_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("fallback_node", fallback_node)
    
    workflow.set_entry_point("router_node")
    
    workflow.add_conditional_edges("router_node", check_intent)
    workflow.add_edge("greeting_node", END)
    
    workflow.add_edge("retrieve_node", grade_node)
    workflow.add_conditional_edges("grade_node", check_relevance)
    
    workflow.add_edge("web_search_node", "process_hitl_node")
    # HITL interrupt
    workflow.add_conditional_edges("process_hitl_node", hitl_router)
    
    workflow.add_edge("generate_node", END)
    workflow.add_edge("fallback_node", END)
    
    db_dir = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "langgraph.sqlite")
    
    # Kết nối SQLite cho checkpointer. Sử dụng check_same_thread=False cho FastAPI.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Biên dịch đồ thị, cấu hình dừng trước node process_hitl_node
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["process_hitl_node"]
    )
    return app
