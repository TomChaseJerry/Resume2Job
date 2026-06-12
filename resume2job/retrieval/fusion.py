# -*- coding: utf-8 -*-
"""RRF（Reciprocal Rank Fusion）多路召回融合。

把多个有序命中列表（不同 query × 不同检索通道）融合为一个排名：

    rrf_score(job) = Σ_lists 1 / (k + rank_in_list)

只依赖名次、不依赖各通道分数的量纲，因此 BM25（无界分数）与向量相似度
（0~1）可以直接融合，无需分数对齐 —— 这是 RRF 相对线性加权的核心优势。
k 取业界常用的 60（见 core.config.RRF_K），k 越大头部名次的权重差越平缓。
"""

from typing import List

from resume2job.core import config

_MAX_TERMS = 8


def rrf_fuse(ranked_lists: List[list], k: int = None) -> list:
    """融合多个有序命中列表，返回按 rrf_score 降序的去重列表。

    每个元素是与向量/BM25 通道同构的 hit dict（含 job_id）。
    同一 job 在多个列表命中时：
        - rrf_score 累加（多通道共识的岗位自然上浮）；
        - matched_terms 合并去重（解释「为什么召回」）；
        - document/metadata 等取首次出现的版本（同一岗位各通道内容一致）。
    融合后 retrieval_score 即 rrf_score。
    """
    k = config.RRF_K if k is None else k
    fused: dict = {}

    for hits in ranked_lists:
        for rank, hit in enumerate(hits or [], start=1):
            job_id = hit.get("job_id")
            if not job_id:
                continue
            contribution = 1.0 / (k + rank)
            if job_id not in fused:
                merged = dict(hit)
                merged["rrf_score"] = contribution
                merged["matched_terms"] = list(hit.get("matched_terms") or [])[:_MAX_TERMS]
                fused[job_id] = merged
                continue

            existing = fused[job_id]
            existing["rrf_score"] += contribution
            # 保留各通道的原始分数便于解释（向量 / BM25 命中各自写入）
            for score_key in ("bm25_score", "vector_score"):
                if score_key in hit and score_key not in existing:
                    existing[score_key] = hit[score_key]
            seen = {t.lower() for t in existing["matched_terms"]}
            for term in hit.get("matched_terms") or []:
                if len(existing["matched_terms"]) >= _MAX_TERMS:
                    break
                if term.lower() not in seen:
                    seen.add(term.lower())
                    existing["matched_terms"].append(term)

    result = list(fused.values())
    for hit in result:
        hit["retrieval_score"] = hit["rrf_score"]
    result.sort(key=lambda h: h["rrf_score"], reverse=True)
    return result
