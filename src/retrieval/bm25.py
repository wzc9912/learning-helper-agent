"""
BM25 稀疏检索实现
"""
from typing import List, Dict
from rank_bm25 import BM25Okapi
import re
import jieba


class BM25Retriever:
    """BM25 稀疏检索器"""

    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []

    def _tokenize(self, text: str) -> List[str]:
        """智能分词：支持中英文"""
        # 检测是否包含中文
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))

        if has_chinese:
            # 中文内容使用 jieba 分词
            return list(jieba.cut(text))
        else:
            # 英文内容：转小写 + 按空格和标点分词
            text = text.lower()
            tokens = re.findall(r'\b\w+\b', text)
            return tokens

    def index_documents(self, chunks: List[Dict[str, str]]) -> None:
        """索引文档块"""
        self.documents = chunks

        self.tokenized_docs = []
        for chunk in chunks:
            tokens = self._tokenize(chunk["content"])
            self.tokenized_docs.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_docs)

        print(f"✓ 已索引 {len(chunks)} 个文档块到 BM25")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25 搜索"""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取最高分的结果，即使分数为0
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            # 返回所有结果，不管分数是否为0
            results.append({
                "id": self.documents[idx].get("id", f"doc_{idx}"),
                "content": self.documents[idx]["content"],
                "metadata": self.documents[idx].get("metadata", {}),
                "score": float(scores[idx])
            })

        return results
