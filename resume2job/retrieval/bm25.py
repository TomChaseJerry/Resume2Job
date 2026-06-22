# -*- coding: utf-8 -*-
"""BM25 稀疏检索模块（混合检索的关键词通道）。

与向量检索（Chroma）互补：
    - 向量通道擅长语义泛化（「大模型应用」召回「LLM 工程」岗）；
    - BM25 通道擅长精确关键词命中（「LangGraph」「RLHF」这类低频专有名词），
      不受 embedding 语义平滑稀释。

语料来自 SQLite 事实源（jobs 表的 index_text，与向量通道入库的是同一份文本），
过滤字段（city / direction / education_level）也取自 SQLite，保证两个通道的
可比性与过滤一致性；SQLite 无可用语料时回退到 Chroma documents（兼容未迁移的旧库）。
岗位库规模小（百级以内），全量构建内存索引即可；按 jobs 表行数做轻量缓存。

BM25 命中不携带 jd_profile —— 与向量通道一致，由 retriever 在融合后统一从
SQLite 批量回填（hydration）。

分词：jieba 搜索引擎模式（lcut_for_search），中英混排友好；统一小写。
"""

import re
from typing import List

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


class BM25Corpus:
    """基于岗位库全量 index_text 构建的内存 BM25 索引。"""

    def __init__(self, ids: list, documents: list, metadatas: list):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self._bm25 = BM25Okapi([tokenize(doc) for doc in documents]) if documents else None

    @classmethod
    def from_jobs_store(cls) -> "BM25Corpus":
        """从 SQLite 事实源加载语料（首选路径）。"""
        from resume2job.storage import jobs_store

        rows = jobs_store.all_jobs_for_index()
        ids = [r["job_id"] for r in rows]
        documents = [r["index_text"] for r in rows]
        metadatas = [
            {k: v for k, v in r.items() if k not in ("job_id", "index_text") and v}
            for r in rows
        ]
        return cls(ids, documents, metadatas)

    @classmethod
    def from_collection(cls, collection) -> "BM25Corpus":
        """从 Chroma collection 加载语料（SQLite 未迁移时的兼容回退）。"""
        res = collection.get(include=["documents", "metadatas"])
        ids = res.get("ids") or []
        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        return cls(ids, documents, metadatas)

    def search(self, query_text: str, n_results: int, allowed_ids=None) -> list:
        """BM25 检索，返回与向量通道同构的命中列表（按归一化分数降序）。

        allowed_ids 非空时只在该 job_id 集合内打分召回（硬约束召回前预筛，与向量通道一致）。
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
            if allowed_ids is not None and self.ids[i] not in allowed_ids:
                continue   # 硬约束预筛：不在 eligible 集合的岗位不进 BM25 命中
            metadata = self.metadatas[i] if isinstance(self.metadatas[i], dict) else {}
            norm = float(scores[i] / max_score) if max_score > 0 else 0.0
            hits.append({
                "job_id": self.ids[i],
                "company": metadata.get("company"),
                "title": metadata.get("title"),
                "direction": metadata.get("direction"),
                "city": metadata.get("city"),
                "retrieval_score": norm,
                "bm25_score": norm,
                "matched_terms": [],   # 由 retriever 在回填后统一填充
                "document": self.documents[i] or "",
                "metadata": metadata,
                "jd_profile": {},      # 由 retriever 从 SQLite 统一回填
            })
            if len(hits) >= max(1, n_results):
                break
        return hits


# ===== 轻量缓存：岗位库条数不变时复用已建索引 =====
_CACHE: dict = {}


def get_bm25_corpus(collection=None) -> BM25Corpus:
    """获取（必要时构建）BM25 索引；以来源 + 条数为缓存键。

    首选 SQLite 事实源；其 index_text 为空（旧库未迁移）且提供了 collection 时
    回退到 Chroma documents。岗位库追加新 JD 后条数变化，自动触发重建；
    百级语料重建开销可忽略。
    """
    from resume2job.storage import jobs_store

    n = jobs_store.count()
    key = ("sqlite", jobs_store.DB_PATH, n)
    if key not in _CACHE:
        corpus = BM25Corpus.from_jobs_store()
        if not corpus.ids and collection is not None:
            print("[WARN] SQLite 无可用 BM25 语料（未迁移？），回退 Chroma documents。"
                  "建议运行 scripts/rebuild_index.py --migrate")
            try:
                key = ("chroma", collection.name, collection.count())
            except Exception:
                return BM25Corpus.from_collection(collection)
            if key in _CACHE:
                return _CACHE[key]
            corpus = BM25Corpus.from_collection(collection)
        _CACHE.clear()  # 只保留当前版本，避免陈旧索引堆积
        _CACHE[key] = corpus
    return _CACHE[key]
