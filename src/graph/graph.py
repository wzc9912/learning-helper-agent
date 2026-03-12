"""
Langgraph 状态图构建
"""
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes import (
    check_exit_node,
    optimize_query_node,
    ask_user_source_choice_node,
    check_kb_coverage_node,
    search_web_node,
    interrupt_user_confirmation_node,
    download_pages_node,
    chunk_documents_node,
    embed_and_store_node,
    retrieve_node,
    generate_answer_node,
    update_kb_summary_node,
)


def should_exit_condition(state: AgentState) -> str:
    if state.get("should_exit", False):
        return END
    return "optimize_query"


def user_source_choice_condition(state: AgentState) -> str:
    # 如果两者都是 False，说明用户选择自动判断
    if not state.get("user_choice_kb", False) and not state.get("user_choice_web", False):
        return "check_kb_coverage"
    if state.get("user_choice_kb", False):
        return "retrieve"
    return "search_web"


def user_confirmation_condition(state: AgentState) -> str:
    if state.get("user_confirmed", False):
        return "download_pages"
    return "optimize_query"


def create_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("check_exit", check_exit_node)
    workflow.add_node("optimize_query", optimize_query_node)
    workflow.add_node("ask_user_source_choice", ask_user_source_choice_node)
    workflow.add_node("check_kb_coverage", check_kb_coverage_node)
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("interrupt_user_confirmation", interrupt_user_confirmation_node)
    workflow.add_node("download_pages", download_pages_node)
    workflow.add_node("chunk_documents", chunk_documents_node)
    workflow.add_node("embed_and_store", embed_and_store_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("update_kb_summary", update_kb_summary_node)

    workflow.set_entry_point("check_exit")

    workflow.add_conditional_edges("check_exit", should_exit_condition, {END: END, "optimize_query": "optimize_query"})
    workflow.add_edge("optimize_query", "ask_user_source_choice")
    workflow.add_conditional_edges("ask_user_source_choice", user_source_choice_condition, {
        "check_kb_coverage": "check_kb_coverage",
        "retrieve": "retrieve",
        "search_web": "search_web"
    })
    workflow.add_conditional_edges("check_kb_coverage", user_source_choice_condition, {
        "retrieve": "retrieve",
        "search_web": "search_web"
    })
    workflow.add_edge("search_web", "interrupt_user_confirmation")
    workflow.add_conditional_edges("interrupt_user_confirmation", user_confirmation_condition, {
        "download_pages": "download_pages",
        "optimize_query": "optimize_query"
    })
    workflow.add_edge("download_pages", "chunk_documents")
    workflow.add_edge("chunk_documents", "embed_and_store")
    workflow.add_edge("embed_and_store", "update_kb_summary")
    workflow.add_edge("update_kb_summary", "retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow
