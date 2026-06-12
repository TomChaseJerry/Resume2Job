# -*- coding: utf-8 -*-
"""BM25 稀疏检索模块（混合检索的关键词通道）。

与向量检索（Chroma）互补：
    - 向量通道擅长语义泛化（「大模型应用」召回「LLM 工程」岗）；
    - BM25 通道擅长精确关键词命中（「LangGraph」「RLHF」这类低频专有名词），
      不受 embedding 语义平滑稀释。

语料直接来自 Chroma collection 的 documents（与向量通道共用同一份 index_text 与
metadata），保证两个通道的可比性与过滤一致性。岗位库规模小（百级以内），
每次检索时全量构建内存索引即可；构建结果按 collection 条数做轻量缓存。

分词：jieba 搜索引擎模式（lcut_for_search），中英混排友好；统一小写。
"""

import re
from typing import List, Optional

from rank_bm25 import BM25Okapi

# 仅保留中英文与数字 token，剔除标点 / 单字噪音（中文单字保留意义不大）
_TOKEN_RE = re.compile(r"[一-龥a-z0-9+#\.]+")


def tokenize(text: str) -> List[str]:
    """jieba 搜索引擎模式分词 + 小写 + 去标点。空文本返回 []。"""
    import jieba  # 延迟导入：首次加载词典较慢，且无 BM25 需求的链路不必付出该成本

    if not isinstance(text, str) or not text.strip():
        return []
    tokens = []
    for tok in jieba.lcut_for_search(text.lower()):
        tok = tok.strip()
        if tok and _TOKEN_RE.fullmatch(tok):
            tokens.append(tok)
    return tokens


def _match_where(metadata: dict, where: Optional[dict]) -> bool:
    """对单条 metadata 应用 Chroma 风格的 where 过滤（仅支持本项目用到的
    等值条件与 {"$and": [...]} 组合，与 build_where_filter 的输出对齐）。"""
    if not where:
        return True
    if "$and" in where:
        return all(_match_where(metadata, cond) for cond in where["$and"])
    for field, expected in where.items():
        if (metadata or {}).get(field) != expected:
            return False
    return True


class BM25Corpus:
    """基于 Chroma collection 全量 documents 构建的内存 BM25 索引。"""

    def __init__(self, ids: list, documents: list, metadatas: list):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self._bm25 = BM25Okapi([tokenize(doc) for doc in documents]) if documents else None

    @classmethod
    def from_collection(cls, collection) -> "BM25Corpus":
        """从 Chroma collection 全量加载语料（与向量通道共用 index_text）。"""
        res = collection.get(include=["documents", "metadatas"])
        ids = res.get("ids") or []
        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        return cls(ids, documents, metadatas)

    def search(self, query_text: str, n_results: int, where: Optional[dict] = None) -> list:
        """BM25 检索，返回与向量通道同构的命中列表（按归一化分数降序）。

        bm25 原始分数无界，做 max 归一化到 [0,1] 写入 retrieval_score / bm25_score，
        便于纯 BM25 模式下的展示；混合模式只用名次（RRF），归一化不影响融合结果。
        """
        if self._bm25 is None:
            return []
        tokens = tokenize(query_text)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            (i for i in range(len(self.ids)) if scores[i] > 0),
            key=lambda i: scores[i], reverse=True,
        )

        max_score = scores[ranked[0]] if ranked else 1.0
        hits = []
        for i in ranked:
            metadata = self.metadatas[i] if isinstance(self.metadatas[i], dict) else {}
            if not _match_where(metadata, where):
                continue
            norm = float(scores[i] / max_score) if max_score > 0 else 0.0
            hits.append({
                "job_id": self.ids[i],
                "company": metadata.get("company"),
                "title": metadata.get("title"),
                "direction": metadata.get("direction"),
                "city": metadata.get("city"),
                "retrieval_score": norm,
                "bm25_score": norm,
                "matched_terms": [],   # 由调用方统一用 extract_matched_terms 填充
                "document": self.documents[i] or "",
                "metadata": metadata,
                "jd_profile": _load_jd_profile(metadata),
            })
            if len(hits) >= max(1, n_results):
                break
        return hits


def _load_jd_profile(metadata: dict) -> dict:
    """从 metadata.jd_profile_json 还原结构化 JD（失败返回空字典）。"""
    import json

    raw = (metadata or {}).get("jd_profile_json")
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
    return {}


# ===== 轻量缓存：同一 collection 条数不变时复用已建索引 =====
_CACHE: dict = {}


def get_bm25_corpus(collection) -> BM25Corpus:
    """获取（必要时构建）BM25 索引；以 collection 名 + 条数为缓存键。

    岗位库追加新 JD 后 count 变化，自动触发重建；百级语料重建开销可忽略。
    """
    try:
        key = (collection.name, collection.count())
    except Exception:
        return BM25Corpus.from_collection(collection)
    if key not in _CACHE:
        _CACHE.clear()  # 只保留当前版本，避免陈旧索引堆积
        _CACHE[key] = BM25Corpus.from_collection(collection)
    return _CACHE[key]
