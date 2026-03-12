"""
ChromaDB 向量存储封装
"""
import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from src.tools.embeddings import DashScopeEmbeddings


class VectorStore:
    """ChromaDB 向量存储封装"""

    def __init__(self):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embeddings = DashScopeEmbeddings()

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """添加文档块到向量存储"""
        if not chunks:
            return

        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]
        ids = [chunk.get("id", f"doc_{i}") for i, chunk in enumerate(chunks)]

        embeddings = self.embeddings.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✓ 已添加 {len(chunks)} 个文档块到向量存储")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量相似度搜索"""
        query_embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if results["distances"] else 0
                })

        return formatted_results

    def get_collection_count(self) -> int:
        """获取集合中的文档数量"""
        return self.collection.count()

    def clear_collection(self) -> None:
        """清空集合"""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ 向量存储已清空")
