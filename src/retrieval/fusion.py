"""
RRF (Reciprocal Rank Fusion) 融合算法
"""
from typing import List, Dict
from src.config import RRF_K


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = RRF_K
) -> List[Dict]:
    """RRF 融合密集向量和 BM25 检索结果"""
    all_doc_ids = set()
    doc_ranks = {}

    for rank, doc in enumerate(dense_results):
        doc_id = doc.get("id", f"doc_{rank}")
        all_doc_ids.add(doc_id)
        if doc_id not in doc_ranks:
            doc_ranks[doc_id] = {}
        doc_ranks[doc_id]["dense"] = rank
        doc_ranks[doc_id]["dense_doc"] = doc

    for rank, doc in enumerate(sparse_results):
        doc_id = doc.get("id", f"doc_{rank}")
        all_doc_ids.add(doc_id)
        if doc_id not in doc_ranks:
            doc_ranks[doc_id] = {}
        doc_ranks[doc_id]["sparse"] = rank
        doc_ranks[doc_id]["sparse_doc"] = doc

    doc_scores = {}
    for doc_id in all_doc_ids:
        score = 0.0
        doc_info = doc_ranks.get(doc_id, {})

        if "dense" in doc_info:
            score += 1 / (k + doc_info["dense"] + 1)
        if "sparse" in doc_info:
            score += 1 / (k + doc_info["sparse"] + 1)

        doc_scores[doc_id] = score

    sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for doc_id, score in sorted_results:
        doc_info = doc_ranks.get(doc_id, {})
        if "dense_doc" in doc_info:
            doc = doc_info["dense_doc"].copy()
        elif "sparse_doc" in doc_info:
            doc = doc_info["sparse_doc"].copy()
        else:
            continue

        doc["score"] = score
        final_results.append(doc)

    return final_results
