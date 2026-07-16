# -*- coding: utf-8 -*-
"""ingest/lifecycle.py — 岗位生命周期与索引更新（接入的统一核心）。

这是「采集 → 规范化 → 校验 → 去重 → 更新 → 索引 → 下线」的唯一编排处。批量建库
（retrieval.indexer.index_jobs）与运行时粘贴入库（storage.jd_ingest_node）都改为经此入口，
从根上消除「两条入库路径各写各的、SQLite/BM25/Chroma 漂移」的隐患。

ingest_record 的判定：
    身份模式（payload.job_id 给定，如本地文件 stem）：
        既有且哈希未变 → unchanged（确认向量在、刷新 last_verified_at，零 token）；
        既有但内容变了 → updated（重解析 + 删旧向量重嵌入）；不存在 → created。
    全量去重模式（无 job_id，如用户粘贴）：
        精确去重(公司+标题+哈希) → 命中即 duplicate；
        URL / source_job_id 命中 → 作为更新目标；
        语义去重(向量近邻 > 阈值) → duplicate；否则 created。
质量闸门：strict=True 时 is_valid=False 的岗位被拦截（invalid，不入库）；默认非严格（入库并记 quality_score）。

依赖方向：本模块**顶层** import indexer 的纯函数（build_index_text 等）；indexer / jd_ingest 反过来
**惰性** import 本模块，避免导入期环。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import chromadb

from resume2job.storage import jobs_store
from resume2job.storage.jobs_store import compute_jd_hash
from resume2job.storage.paths import CHROMA_DIR, COLLECTION_NAME
from resume2job.parsing.jd_parser import parse_jd
from resume2job.core.llm import get_embedding
from resume2job.ingest import versions
from resume2job.ingest import validator
from resume2job.ingest.normalizer import normalize_raw_payload
from resume2job.ingest.models import RawJobPayload, IngestResult, JobRecord, STATUS_ACTIVE, STATUS_EXPIRED, STATUS_REMOVED
# indexer 的纯函数（顶层 import 安全：indexer 不在顶层 import 本模块）
from resume2job.retrieval.indexer import build_index_text, build_chroma_metadata, job_exists

# 语义去重阈值（归一化向量下 cosine = 1 - L2²/2，与 storage/jd_ingest 历史口径一致）
SIMILARITY_THRESHOLD = 0.92
# 长期未验证多少天判过期（sweep_stale 默认）
DEFAULT_STALE_DAYS = 90


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Chroma collection / 向量写入 / 语义去重
# ---------------------------------------------------------------------------
def get_jobs_collection(collection=None, db_path: Optional[str] = None):
    """返回 jobs collection：显式传入则用之，否则按 paths 打开统一向量库。"""
    if collection is not None:
        return collection
    path = db_path or CHROMA_DIR
    os.makedirs(path, exist_ok=True)
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _write_vector(collection, job_id: str, vector, index_text: str, jd_profile: dict, source: str) -> None:
    """写入一条向量 + 最小可过滤 metadata（与 indexer / jd_ingest 完全一致）。"""
    collection.add(
        ids=[job_id],
        embeddings=[vector],
        documents=[index_text],
        metadatas=[build_chroma_metadata(job_id, jd_profile, source=source)],
    )


def _semantic_duplicate(vector, collection):
    """语义去重：Chroma 查最近邻，L2 距离转余弦相似度与阈值比较。

    返回 (命中的 job_id 或 None, 相似度)；collection 空 / 查询异常时返回 (None, 0.0)。
    """
    if collection is None:
        return None, 0.0
    try:
        result = collection.query(query_embeddings=[vector], n_results=1, include=["distances"])
    except Exception:
        return None, 0.0
    distances = result.get("distances") or [[]]
    ids = result.get("ids") or [[]]
    if not distances or not distances[0]:
        return None, 0.0
    similarity = 1 - distances[0][0] / 2
    if similarity > SIMILARITY_THRESHOLD:
        return (ids[0][0] if ids and ids[0] else None), similarity
    return None, similarity


def _parse_failed_result(profile, job_id=None) -> IngestResult:
    reason = "parse_failed"
    if isinstance(profile, dict) and profile.get("error"):
        reason = f"{profile.get('error_stage') or 'parse_failed'}: {profile.get('error')}"
    return IngestResult(action="failed", job_id=job_id, reason=reason)


def _resolve_profile(jd_profile, jd_text):
    """优先复用调用方传入的已解析画像（跨链路一致 + 省 token），否则现场 parse_jd。"""
    if isinstance(jd_profile, dict) and jd_profile and not jd_profile.get("error"):
        return jd_profile
    return parse_jd(jd_text)


# ---------------------------------------------------------------------------
# 核心：单条记录接入
# ---------------------------------------------------------------------------
def ingest_record(payload: RawJobPayload, *, jd_profile: Optional[dict] = None,
                  collection=None, semantic_dedup: Optional[bool] = None,
                  strict: bool = False, embed: bool = True) -> IngestResult:
    """把一条 RawJobPayload 接入库（事实源 + 索引），返回 IngestResult。"""
    payload = normalize_raw_payload(payload)
    jd_text = payload.raw_jd
    if not jd_text.strip():
        return IngestResult(action="failed", reason="empty_jd_text")

    cur_hash = compute_jd_hash(jd_text)
    identity_mode = bool(payload.job_id)
    if semantic_dedup is None:
        semantic_dedup = not identity_mode
    col = get_jobs_collection(collection) if embed else None
    now = _utc_now_iso()

    existing = None
    content_changed = False

    if identity_mode:
        job_id = payload.job_id
        existing = jobs_store.get_job(job_id)
        content_changed = bool(existing) and existing.get("jd_hash") != cur_hash
        # 幂等：已存在且内容未变 → 确认向量在、刷新验证时间，不重解析/重嵌入
        if existing and existing.get("jd_profile") and not content_changed:
            profile = existing["jd_profile"]
            index_text = existing.get("index_text") or build_index_text(profile)
            if embed and col is not None and index_text.strip() and not job_exists(col, job_id):
                try:
                    _write_vector(col, job_id, get_embedding(index_text), index_text, profile, payload.source)
                    jobs_store.update_embedding_version(job_id, versions.embedding_version())
                except Exception as e:
                    print(f"[lifecycle] 警告：补建向量失败 {job_id}：{e}")
            jobs_store.touch_verified(job_id, now)
            q = validator.validate_job(profile, jd_text)
            return IngestResult(action="unchanged", job_id=job_id,
                                quality_score=q.quality_score, warnings=q.warnings)
        profile = _resolve_profile(jd_profile, jd_text)
    else:
        profile = _resolve_profile(jd_profile, jd_text)
        if not isinstance(profile, dict) or profile.get("error"):
            return _parse_failed_result(profile)
        company = profile.get("company") or payload.company or "unknown_company"
        title = profile.get("title") or payload.title or "unknown_title"
        # ① 精确去重（公司 + 标题 + 原文哈希）
        dup = jobs_store.get_job_id_by_exact(company, title, cur_hash)
        if dup:
            jobs_store.touch_verified(dup, now)
            return IngestResult(action="duplicate", job_id=dup, is_duplicate=True, reason="exact")
        # ② URL / source_job_id → 既有更新目标（增量同步）
        existing_id = None
        if payload.canonical_url:
            existing_id = jobs_store.get_job_id_by_canonical_url(payload.canonical_url)
        if not existing_id and payload.source_job_id:
            existing_id = jobs_store.get_job_id_by_source(payload.source, payload.source_job_id)
        if existing_id:
            existing = jobs_store.get_job(existing_id)
            content_changed = bool(existing) and existing.get("jd_hash") != cur_hash
            job_id = existing_id
        else:
            job_id = f"job_{uuid4().hex[:8]}"

    # 解析失败统一处理（identity 模式 content_changed 走到此处）
    if not isinstance(profile, dict) or profile.get("error"):
        return _parse_failed_result(profile, job_id=payload.job_id)

    # ---- 质量闸门 ----
    q = validator.validate_job(profile, jd_text)
    if strict and not q.is_valid:
        return IngestResult(action="invalid", job_id=payload.job_id,
                            quality_score=q.quality_score, warnings=q.warnings, reason="quality_gate")

    # ---- index_text ----
    index_text = build_index_text(profile)
    if not index_text.strip():
        return IngestResult(action="failed", job_id=job_id, reason="empty_index_text")

    # ---- embedding（全量去重模式在此做语义去重）----
    vector = None
    if embed and col is not None:
        try:
            vector = get_embedding(index_text)
        except Exception as e:
            return IngestResult(action="failed", job_id=job_id, reason=f"embedding_failed: {e}")
        if semantic_dedup and existing is None:
            dup_id, sim = _semantic_duplicate(vector, col)
            if dup_id:
                jobs_store.touch_verified(dup_id, now)
                return IngestResult(action="duplicate", job_id=dup_id, is_duplicate=True,
                                    similarity=sim, reason="semantic")

    # ---- 写 SQLite（事实源；版本 / 生命周期 / 质量一并落库）----
    record = JobRecord.from_jd_profile(
        job_id, jd_text, profile, index_text=index_text, source=payload.source,
        source_job_id=payload.source_job_id, canonical_url=payload.canonical_url,
        collected_at=payload.collected_at, last_verified_at=now,
        quality_score=q.quality_score, status=STATUS_ACTIVE,
    )
    try:
        jobs_store.upsert_job(record.to_store_dict())
    except Exception as e:
        return IngestResult(action="failed", job_id=job_id, reason=f"sqlite_write_failed: {e}")

    # ---- 写 Chroma（内容变更先删旧向量再重嵌入，保持双通道一致）----
    if embed and col is not None and vector is not None:
        try:
            if content_changed:
                try:
                    col.delete(ids=[job_id])
                except Exception as e:
                    print(f"[lifecycle] 警告：删除旧向量失败（将覆盖写入）{job_id}：{e}")
            _write_vector(col, job_id, vector, index_text, profile, payload.source)
            jobs_store.update_embedding_version(job_id, versions.embedding_version())
        except Exception as e:
            # SQLite 已写入，向量可由 scripts/rebuild_index 补建
            return IngestResult(action=("updated" if existing else "created"), job_id=job_id,
                                quality_score=q.quality_score, warnings=q.warnings,
                                reason=f"chroma_write_failed: {e}")

    return IngestResult(action=("updated" if existing else "created"), job_id=job_id,
                        quality_score=q.quality_score, warnings=q.warnings)


# ---------------------------------------------------------------------------
# 批量接入（驱动一个 connector 跑完，汇总动作计数）
# ---------------------------------------------------------------------------
def ingest_all(connector, *, collection=None, strict: bool = False,
               embed: bool = True, verbose: bool = True) -> dict:
    """把一个 connector 的全部产出逐条接入；返回 {counts, results}。"""
    col = get_jobs_collection(collection) if embed else None
    counts = {"created": 0, "updated": 0, "unchanged": 0, "duplicate": 0, "invalid": 0, "failed": 0}
    results = []
    for payload in connector.fetch():
        res = ingest_record(payload, collection=col, strict=strict, embed=embed)
        counts[res.action] = counts.get(res.action, 0) + 1
        results.append(res)
        if verbose:
            label = payload.job_id or res.job_id or "?"
            extra = f"  ({res.reason})" if res.reason else ""
            qs = f"  q={res.quality_score}" if res.quality_score is not None else ""
            print(f"[ingest] {res.action:9s} {label}{extra}{qs}")
    if verbose:
        summary = "，".join(f"{k}={v}" for k, v in counts.items() if v)
        print(f"[ingest] 完成：{summary or '无记录'}")
    return {"counts": counts, "results": results}


# ---------------------------------------------------------------------------
# 生命周期状态流转
# ---------------------------------------------------------------------------
def mark_expired(job_id: str) -> None:
    """标记岗位过期（不再召回，保留历史）。"""
    jobs_store.set_status(job_id, STATUS_EXPIRED)


def mark_removed(job_id: str) -> None:
    """标记岗位已下线（来源删除 / 显式移除，不再召回）。"""
    jobs_store.set_status(job_id, STATUS_REMOVED)


def reactivate(job_id: str) -> None:
    """重新激活岗位并刷新验证时间。"""
    jobs_store.set_status(job_id, STATUS_ACTIVE, _utc_now_iso())


def sweep_stale(max_age_days: int = DEFAULT_STALE_DAYS) -> int:
    """把「active 且 last_verified_at 超过 max_age_days 未刷新」的岗位批量标记过期，返回条数。"""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
    n = jobs_store.expire_stale(cutoff)
    print(f"[lifecycle] 过期清扫：{n} 条（未验证超过 {max_age_days} 天）已标记为 expired")
    return n


# ---------------------------------------------------------------------------
# 旧库回填版本 / 生命周期列（纯 SQLite + 规则校验，无 LLM）
# ---------------------------------------------------------------------------
def backfill_lifecycle_fields() -> int:
    """为既有岗位回填新增的版本 / 生命周期 / 质量列。

    从已存的 jd_profile / index_text / 时间戳就地推算（不重解析、不重嵌入、不调 LLM）：
      - content_hash      ← index_text 的内容哈希；
      - parser/index_text/embedding_version ← 当前版本戳（假定既有数据由当前流程产出）；
      - quality_score     ← validator.validate_job（纯规则）；
      - collected_at      ← created_at；last_verified_at ← updated_at or created_at；
      - status            ← 保留既有（NULL → active）。
    返回回填条数。
    """
    from resume2job.storage.jobs_store import compute_content_hash
    rows = jobs_store.all_rows()
    n = 0
    ev = versions.embedding_version()
    for r in rows:
        index_text = r.get("index_text") or ""
        profile = r.get("jd_profile") if isinstance(r.get("jd_profile"), dict) else {}
        q = validator.validate_job(profile, r.get("jd_text") or "")
        jobs_store.update_lifecycle_fields(
            r["job_id"],
            status=(r.get("status") or STATUS_ACTIVE),
            content_hash=compute_content_hash(index_text),
            parser_version=versions.PARSER_VERSION,
            embedding_version=ev,
            index_text_version=versions.INDEX_TEXT_VERSION,
            quality_score=q.quality_score,
            collected_at=(r.get("collected_at") or r.get("created_at")),
            last_verified_at=(r.get("last_verified_at") or r.get("updated_at") or r.get("created_at")),
        )
        n += 1
    print(f"[lifecycle] 版本 / 生命周期列回填完成：{n} 条")
    return n
