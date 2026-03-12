"""
Langgraph 状态定义
"""
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[List, add_messages]
    user_query: str
    optimized_query: str
    user_choice_kb: bool
    user_choice_web: bool
    search_results: List[dict]
    selected_results: List[dict]
    downloaded_docs: List[str]
    document_chunks: List[dict]
    retrieved_chunks: List[dict]
    user_confirmed: bool
    user_refusal_reason: Optional[str]
    answer: str
    should_exit: bool
    _results_displayed: bool
    _awaiting_input: bool
