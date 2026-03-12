"""
网页搜索工具 - Tavily API 集成
"""
from typing import List, Dict
from tavily import TavilyClient
from src.config import TAVILY_API_KEY


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    使用 Tavily API 搜索网页

    Args:
        query: 搜索查询
        max_results: 返回的最大结果数

    Returns:
        搜索结果列表，每个结果包含 title, url, snippet
    """
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is not set. Please set it in your .env file.")

    client = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )

        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", ""),
            })

        return results

    except Exception as e:
        print(f"搜索时出错: {e}")
        return []
