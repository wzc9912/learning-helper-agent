"""
嵌入模型封装 - 阿里云 DashScope Embedding API
支持原生 API 和 OpenAI 兼容接口
"""
from typing import List
from langchain.embeddings.base import Embeddings
import time
from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_USE_OPENAI_API
)


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope 嵌入模型封装"""

    def __init__(self):
        self.model = EMBEDDING_MODEL
        if not EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY is not set. Please set DASHSCOPE_API_KEY in your .env file.")

        self.use_openai_api = EMBEDDING_USE_OPENAI_API

        if self.use_openai_api:
            # 使用 OpenAI 兼容接口
            from openai import OpenAI
            self.client = OpenAI(
                api_key=EMBEDDING_API_KEY,
                base_url=EMBEDDING_BASE_URL
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        if self.use_openai_api:
            return self._embed_documents_openai(texts)
        else:
            all_embeddings = []
            for text in texts:
                embedding = self._embed_text_with_retry(text)
                all_embeddings.append(embedding)
            return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        if self.use_openai_api:
            return self._embed_query_openai(text)
        else:
            return self._embed_text_with_retry(text)

    def _embed_documents_openai(self, texts: List[str]) -> List[List[float]]:
        """使用 OpenAI 兼容接口嵌入多个文档"""
        # 阿里云 API 批量限制为 10，需要分批处理
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                raise Exception(f"OpenAI API 嵌入失败 (batch {i//batch_size + 1}): {e}")

        return all_embeddings

    def _embed_query_openai(self, text: str) -> List[float]:
        """使用 OpenAI 兼容接口嵌入单个查询"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI API 嵌入失败: {e}")

    def _embed_text_with_retry(self, text: str, max_retries: int = 3) -> List[float]:
        """使用 DashScope 原生 API 获取嵌入向量，带重试机制"""
        from dashscope import TextEmbedding

        for attempt in range(max_retries):
            try:
                resp = TextEmbedding.call(
                    model=self.model,
                    input=text,
                    api_key=EMBEDDING_API_KEY,
                    timeout=10
                )
                if resp.status_code == 200:
                    return resp.output['embeddings'][0]['embedding']
                else:
                    raise Exception(f"Embedding failed: {resp.message}")

            except Exception as e:
                error_msg = str(e)
                # SSL 错误或连接错误，进行重试
                if any(keyword in error_msg for keyword in ['SSL', 'EOF', 'connection', 'timeout']):
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"⚠️ API 连接错误，{wait_time}秒后重试... ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        # 最后一次重试失败，尝试使用 OpenAI 接口
                        print("⚠️ 原生 API 失败，尝试使用 OpenAI 兼容接口...")
                        return self._embed_query_openai(text)
                else:
                    raise Exception(f"Embedding API 调用失败: {e}")

        raise Exception("Embedding failed after retries")
