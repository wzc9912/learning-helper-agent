"""
Langgraph 节点函数实现
"""
import os
import sys
from typing import Dict
from langgraph.types import interrupt
from openai import OpenAI

from src.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, DOCUMENTS_DIR,
    TOP_K_RETRIEVAL, TOP_K_RESULTS
)
from src.utils.prompts import GENERATE_ANSWER_PROMPT, OPTIMIZE_QUERY_PROMPT, INITIAL_OPTIMIZE_QUERY_PROMPT
from src.tools.search import search_web
from src.tools.scraper import download_page
from src.processing.chunking import MarkdownChunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.processing.kb_summary import KnowledgeBaseSummary


# ============ 全局变量 ============
_vector_store = None
_bm25_retriever = None
_kb_summary = None
_chunker = None
_llm_client = None


def _get_llm_client():
    """获取 OpenAI 客户端"""
    global _llm_client
    if _llm_client is None:
        if not LLM_API_KEY:
            raise ValueError("LLM_API_KEY is not set. Please set OPENAI_API_KEY in your .env file.")
        _llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _llm_client


def _call_llm(prompt: str, max_tokens: int = 2000) -> str:
    """调用 LLM"""
    client = _get_llm_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的学习助手。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content or ""
        return content
    except Exception as e:
        raise Exception(f"LLM API 调用失败: {e}")


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
    return _bm25_retriever


def _get_kb_summary():
    global _kb_summary
    if _kb_summary is None:
        _kb_summary = KnowledgeBaseSummary()
    return _kb_summary


def _get_chunker():
    global _chunker
    if _chunker is None:
        _chunker = MarkdownChunker()
    return _chunker


# ============ 节点函数 ============

def check_exit_node(state: Dict) -> Dict:
    """检查用户是否要退出"""
    user_query = state.get("user_query", "").strip().lower()
    exit_commands = ["exit", "quit", "退出", "结束"]
    should_exit = any(cmd in user_query for cmd in exit_commands)

    if should_exit:
        print("\n👋 感谢使用学习助手，再见！")
        return {"should_exit": True}

    return {"should_exit": False, "_results_displayed": False, "_awaiting_input": False}


def optimize_query_node(state: Dict) -> Dict:
    """优化用户查询"""
    original_query = state.get("user_query", "")
    refusal_reason = state.get("user_refusal_reason", "")

    # 总是使用 LLM 优化查询
    try:
        if refusal_reason:
            # 用户拒绝后的重新优化
            print(f"\n🔄 根据你的反馈优化查询...", flush=True)
            prompt = OPTIMIZE_QUERY_PROMPT.format(
                original_query=original_query,
                refusal_reason=refusal_reason
            )
            optimized = _call_llm(prompt, max_tokens=200).strip()
            print(f"✓ 优化后的查询: {optimized}", flush=True)
            return {"optimized_query": optimized}
        else:
            # 首次查询优化
            print(f"\n🔄 优化搜索查询...", flush=True)
            prompt = INITIAL_OPTIMIZE_QUERY_PROMPT.format(query=original_query)
            optimized = _call_llm(prompt, max_tokens=200).strip()
            print(f"✓ 优化后的查询: {optimized}", flush=True)
            return {"optimized_query": optimized}
    except Exception as e:
        print(f"⚠️ 查询优化失败，使用原始查询: {e}", flush=True)
        return {"optimized_query": original_query}


def check_kb_coverage_node(state: Dict) -> Dict:
    """检查知识库是否包含相关内容"""
    kb_summary = _get_kb_summary()
    summary = kb_summary.get_summary()

    if summary == "知识库目前为空。":
        print("✓ 知识库为空，将进行网络搜索", flush=True)
        return {"user_choice_kb": False, "user_choice_web": True}

    query = state.get("optimized_query", "")
    print(f"\n🔍 分析知识库是否包含相关内容...", flush=True)

    try:
        prompt = f"""请判断以下知识库内容是否足够回答用户的问题。

知识库内容概要：
{summary}

用户问题：{query}

请直接回答：
- 如果知识库包含相关内容，回答：是
- 如果知识库不包含或内容不足，回答：否"""

        result = _call_llm(prompt, max_tokens=100).strip().lower()

        if "是" in result or "yes" in result:
            print("✓ 知识库包含相关内容", flush=True)
            return {"user_choice_kb": True, "user_choice_web": False}
        else:
            print("✓ 知识库内容不足，将进行网络搜索", flush=True)
            return {"user_choice_kb": False, "user_choice_web": True}

    except Exception as e:
        print(f"⚠️ 判断失败，将进行网络搜索: {e}", flush=True)
        return {"user_choice_kb": False, "user_choice_web": True}


def ask_user_source_choice_node(state: Dict) -> Dict:
    """询问用户查询来源"""
    user_query = state.get("optimized_query", "")

    print("\n" + "="*60, flush=True)
    print("请选择查询来源：", flush=True)
    print("="*60, flush=True)
    print(f"你的问题：{user_query}", flush=True)
    print("", flush=True)
    print("  [a] 自动判断    - 让模型决定是从知识库检索还是网络搜索", flush=True)
    print("  [k] 知识库 (KB)  - 从已下载的文档中检索", flush=True)
    print("  [s] 网络 (Web)   - 搜索网络并下载新文档", flush=True)
    print("="*60, flush=True)

    user_choice = interrupt({
        "action": "choose_source",
        "prompt": "你的选择 (a/k/s，默认a): ",
    })

    choice = str(user_choice).strip().lower()

    # 默认自动判断
    if not choice or choice in ['a', 'auto', '自动']:
        print("✓ 选择自动判断", flush=True)
        return {"user_choice_kb": False, "user_choice_web": False}
    elif choice in ['k', 'kb', '知识库']:
        print("✓ 选择从知识库检索", flush=True)
        return {"user_choice_kb": True, "user_choice_web": False, "user_refusal_reason": None}
    else:
        print("✓ 选择网络搜索", flush=True)
        return {"user_choice_kb": False, "user_choice_web": True, "user_refusal_reason": None}


def search_web_node(state: Dict) -> Dict:
    """网页搜索"""
    optimized_query = state.get("optimized_query", "")
    print(f"\n🔎 正在搜索: {optimized_query}", flush=True)

    results = search_web(optimized_query, max_results=TOP_K_RESULTS)

    if results:
        print(f"\n✓ 找到 {len(results)} 个相关结果\n", flush=True)
        # 打印搜索结果
        print("="*60, flush=True)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}", flush=True)
            print(f"   URL: {result['url']}", flush=True)
        print("="*60, flush=True)
    else:
        print("\n✗ 未找到相关结果\n", flush=True)

    return {"search_results": results, "_results_displayed": True}


def interrupt_user_confirmation_node(state: Dict) -> Dict:
    """中断等待用户确认"""
    results = state.get("search_results", [])
    awaiting_input = state.get("_awaiting_input", False)

    if not results:
        print("\n没有搜索结果可以确认", flush=True)
        return {"user_confirmed": False, "user_refusal_reason": "没有搜索结果"}

    # 只在第一次执行时打印提示（不是 resume）
    if not awaiting_input:
        print("\n输入数字选择要下载的页面（如: 1）", flush=True)
        print("输入 'n' 或 'n: 拒绝理由' 重新搜索（例如：n: 结果不相关）", flush=True)
        # 调用输入处理函数，然后设置标志
        result = _handle_user_input(results)
        result["_awaiting_input"] = True
        return result

    # resume 后直接处理，不打印
    return _handle_user_input(results)


def _handle_user_input(results: list) -> dict:
    """处理用户输入（内部函数）"""
    user_input = interrupt({
        "action": "confirm_download",
        "prompt": "你的选择: ",
        "search_results": results
    })

    input_str = str(user_input).strip()

    # 检查是否输入了数字
    try:
        choice = int(input_str)
        if 1 <= choice <= len(results):
            selected = [results[choice - 1]]
            print(f"\n✓ 选择了第 {choice} 个结果", flush=True)
            return {"user_confirmed": True, "selected_results": selected, "_results_displayed": False, "_awaiting_input": False}
        else:
            print(f"\n⚠️ 无效的选择，请输入 1-{len(results)} 之间的数字", flush=True)
            return {"user_confirmed": False, "user_refusal_reason": "选择超出范围", "_results_displayed": False, "_awaiting_input": False}
    except ValueError:
        pass

    # 检查是否拒绝
    if input_str.lower().startswith('n'):
        refusal_reason = "用户未提供具体原因"
        if ':' in input_str or '：' in input_str:
            parts = input_str.split(':', 1) if ':' in input_str else input_str.split('：', 1)
            if len(parts) > 1:
                refusal_reason = parts[1].strip()
        print(f"\n✗ 用户取消下载", flush=True)
        return {"user_confirmed": False, "user_refusal_reason": refusal_reason, "_results_displayed": False, "_awaiting_input": False}

    # 无效输入
    print(f"\n⚠️ 无效的输入", flush=True)
    return {"user_confirmed": False, "user_refusal_reason": "无效输入", "_results_displayed": False, "_awaiting_input": False}


def download_pages_node(state: Dict) -> Dict:
    """下载网页"""
    selected_results = state.get("selected_results", state.get("search_results", []))
    downloaded_docs = []

    print(f"\n📥 开始下载网页...", flush=True)

    for result in selected_results:
        url = result.get("url", "")
        if url:
            filepath = download_page(url, DOCUMENTS_DIR)
            if filepath:
                downloaded_docs.append(filepath)

    if downloaded_docs:
        print(f"\n✓ 成功下载 {len(downloaded_docs)} 个网页", flush=True)
    else:
        print("\n✗ 网页下载失败", flush=True)

    return {"downloaded_docs": downloaded_docs}


def chunk_documents_node(state: Dict) -> Dict:
    """文档分块"""
    downloaded_docs = state.get("downloaded_docs", [])

    if not downloaded_docs:
        return {"document_chunks": []}

    print(f"\n📄 开始文档分块...", flush=True)

    chunker = _get_chunker()
    all_chunks = chunker.chunk_documents(downloaded_docs)

    print(f"✓ 文档分块完成，共 {len(all_chunks)} 个块", flush=True)

    return {"document_chunks": all_chunks}


def embed_and_store_node(state: Dict) -> Dict:
    """向量化并存储文档块"""
    document_chunks = state.get("document_chunks", [])

    if not document_chunks:
        return {}

    print(f"\n💾 开始向量化并存储文档...", flush=True)

    vector_store = _get_vector_store()
    bm25_retriever = _get_bm25_retriever()

    vector_store.add_documents(document_chunks)
    bm25_retriever.index_documents(document_chunks)

    print("✓ 文档存储完成", flush=True)

    return {}


def retrieve_node(state: Dict) -> Dict:
    """检索相关文档块"""
    optimized_query = state.get("optimized_query", "")

    print(f"\n🔍 检索相关内容...", flush=True)

    vector_store = _get_vector_store()
    bm25_retriever = _get_bm25_retriever()

    dense_results = vector_store.search(optimized_query, top_k=TOP_K_RETRIEVAL)
    sparse_results = bm25_retriever.search(optimized_query, top_k=TOP_K_RETRIEVAL)
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results)

    if fused_results:
        print(f"✓ 检索到 {len(fused_results)} 个相关文档块", flush=True)
    else:
        print("✓ 未检索到相关内容", flush=True)

    return {"retrieved_chunks": fused_results[:TOP_K_RETRIEVAL]}


def generate_answer_node(state: Dict) -> Dict:
    """生成回答"""
    user_query = state.get("user_query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    messages = state.get("messages", [])

    retrieved_content = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("metadata", {}).get("url", chunk.get("metadata", {}).get("filename", "未知来源"))
        content = chunk.get("content", "")
        retrieved_content += f"\n[来源 {i}: {source}]\n{content}\n"

    chat_history = ""
    if messages:
        for msg in messages[-3:]:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            chat_history += f"{content}\n"

    try:
        if not retrieved_chunks:
            answer = "抱歉，知识库中没有找到相关内容。你可以选择 [s] 网络搜索来获取更多信息。"
        else:
            prompt = GENERATE_ANSWER_PROMPT.format(
                query=user_query,
                retrieved_content=retrieved_content,
                chat_history=chat_history or "（无历史对话）"
            )
            answer = _call_llm(prompt, max_tokens=2000)
    except Exception as e:
        answer = f"生成回答时出错: {e}"

    return {"answer": answer}


def update_kb_summary_node(state: Dict) -> Dict:
    """更新知识库总结"""
    downloaded_docs = state.get("downloaded_docs", [])

    if not downloaded_docs:
        return {}

    print(f"\n📝 更新知识库总结...", flush=True)

    kb_summary = _get_kb_summary()

    new_summaries = []
    for filepath in downloaded_docs:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            title = content.split("\n")[0].replace("# ", "").strip()
            if not title:
                title = os.path.basename(filepath)

            summary = kb_summary.generate_document_summary(title, content)
            if summary:
                new_summaries.append(summary)

        except Exception as e:
            print(f"处理文档时出错: {e}")

    if new_summaries:
        kb_summary.update_summary(new_summaries)

    return {}
