# -*- coding: utf-8 -*-
"""ingest/models.py — 统一的岗位数据结构（接入边界的类型契约）。

三个数据结构：
    - RawJobPayload：connector 的产出契约——「任意来源的一条原始岗位」。各来源（本地文件 / CSV /
      用户粘贴 / 官网）都先转成它，下游接入逻辑只认这一种输入。
    - JobRecord：一条标准岗位的**完整**类型化记录，镜像 SQLite jobs 表全部列 + 版本 / 生命周期字段，
      并提供 from_jd_profile / from_store_row / to_store_dict 三个桥接方法，在「解析产物 dict」「SQLite
      行 dict」「类型化记录」之间无损转换。它是入库写路径的契约（lifecycle 用它装配后再交 jobs_store），
      但**不**强行替换检索 / 评分链路里的裸 dict（那些经 jobs_store 回填，保持现状）。
    - IngestResult：lifecycle.ingest_record 的返回——一次接入的结果（动作 / 是否重复 / 质量 / 告警）。

存储约定：版本 / 生命周期字段对应 jobs 表新增列（见 storage/jobs_store.py 的 _MIGRATION_COLUMNS）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ===== 生命周期状态 =====
STATUS_ACTIVE = "active"      # 在招、可被召回
STATUS_EXPIRED = "expired"    # 长期未验证 / 已过期，不再召回（保留历史）
STATUS_REMOVED = "removed"    # 来源已下线 / 显式删除，不再召回

# ===== 来源标识（约定常量；source 字段本身是自由字符串，便于扩展新来源）=====
SOURCE_BATCH = "batch_indexed"
SOURCE_USER = "user_uploaded"
SOURCE_CSV = "csv_import"
SOURCE_OFFICIAL = "official_career"

# IngestResult.action 中代表「岗位最终在库且可召回」的动作集合
OK_ACTIONS = ("created", "updated", "unchanged", "duplicate")


# ---------------------------------------------------------------------------
# RawJobPayload —— connector 产出契约
# ---------------------------------------------------------------------------
@dataclass
class RawJobPayload:
    """一条来自任意来源的原始岗位。connector 负责把各自的格式转成它。

    job_id 是否提供决定接入的去重模式：
        - 提供（如本地文件以文件名 stem 作 job_id）→ **身份模式**：按 job_id 幂等 upsert，不做语义去重；
        - 留空（如用户粘贴）→ **全量去重模式**：精确（公司+标题+哈希）+ 语义（向量近邻）去重后生成新 id。
    company/title 仅为提示（解析器 parse_jd 仍是字段权威）。
    """
    source: str
    raw_jd: str
    job_id: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    source_job_id: Optional[str] = None   # 来源系统的外部 ID（增量同步 / 去重）
    canonical_url: Optional[str] = None    # 岗位规范 URL（按 URL 去重 / 增量更新）
    collected_at: Optional[str] = None     # 来源采集时间（ISO 字符串）
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IngestResult —— 一次接入的结果
# ---------------------------------------------------------------------------
@dataclass
class IngestResult:
    """lifecycle.ingest_record 的返回。

    action ∈ {created（新岗位）/ updated（内容变更，已重解析+重嵌入）/ unchanged（哈希未变，幂等跳过）/
              duplicate（命中精确/语义/URL 去重，复用既有）/ invalid（质量闸门拦截）/ failed（解析/写入/嵌入失败）}。
    """
    action: str
    job_id: Optional[str] = None
    is_duplicate: bool = False
    quality_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    similarity: Optional[float] = None      # 语义去重命中时的相似度
    reason: str = ""                          # failed / invalid 时的说明

    @property
    def ok(self) -> bool:
        """岗位是否最终在库且可召回。"""
        return self.action in OK_ACTIONS

    @property
    def indexed(self) -> bool:
        """本次是否真的写 / 刷新了索引（created / updated）。"""
        return self.action in ("created", "updated")


# ---------------------------------------------------------------------------
# JobRecord —— 标准岗位的完整类型化记录
# ---------------------------------------------------------------------------
class JobRecord(BaseModel):
    """一条标准岗位记录，镜像 jobs 表全部列 + 版本 / 生命周期字段。"""

    # --- 标识与业务字段（与现有 jobs 表对齐）---
    job_id: str
    company: str
    title: str
    city: Optional[str] = None
    direction: Optional[str] = None
    education_level: Optional[str] = None
    jd_text: str
    jd_hash: str
    jd_profile: Dict[str, Any] = Field(default_factory=dict)
    index_text: str = ""
    requirements: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    source: str = SOURCE_BATCH
    embedding_id: Optional[str] = None

    # --- 硬约束派生列（jd_parser.derive_constraint_fields 产出）---
    cities: List[str] = Field(default_factory=list)
    job_types: List[str] = Field(default_factory=list)
    min_degree_rank: Optional[int] = None
    city_status: str = "unknown"
    education_status: str = "unknown"

    # --- 生命周期 ---
    status: str = STATUS_ACTIVE
    source_job_id: Optional[str] = None
    canonical_url: Optional[str] = None
    collected_at: Optional[str] = None        # 来源采集时间（ISO）
    last_verified_at: Optional[str] = None    # 最近一次确认仍有效的时间（ISO）

    # --- 版本戳（见 ingest/versions.py）---
    content_hash: Optional[str] = None         # index_text 的内容哈希（驱动重嵌入判断）
    parser_version: Optional[str] = None
    embedding_version: Optional[str] = None
    index_text_version: Optional[str] = None

    # --- 质量 ---
    quality_score: Optional[float] = None

    # ---------- 构造：解析产物 → 记录 ----------
    @classmethod
    def from_jd_profile(cls, job_id: str, jd_text: str, jd_profile: dict, *,
                        index_text: str, source: str = SOURCE_BATCH,
                        source_job_id: Optional[str] = None, canonical_url: Optional[str] = None,
                        collected_at: Optional[str] = None, last_verified_at: Optional[str] = None,
                        quality_score: Optional[float] = None,
                        status: str = STATUS_ACTIVE) -> "JobRecord":
        """由 parse_jd 的产物 jd_profile + index_text 装配一条完整记录，并盖上当前版本戳。"""
        # 延迟导入，避免任何导入期环（models 被广泛 import）
        from resume2job.storage.jobs_store import compute_jd_hash, compute_content_hash
        from resume2job.parsing.jd_parser import derive_constraint_fields
        from resume2job.ingest import versions

        jd_profile = jd_profile if isinstance(jd_profile, dict) else {}
        loc = jd_profile.get("location") if isinstance(jd_profile.get("location"), dict) else {}
        skills = list(dict.fromkeys([*(jd_profile.get("hard_skills") or []),
                                     *(jd_profile.get("tools_or_frameworks") or [])]))
        cf = derive_constraint_fields(jd_profile)
        return cls(
            job_id=job_id,
            company=jd_profile.get("company") or "unknown_company",
            title=jd_profile.get("title") or "unknown_title",
            city=(loc.get("city") or None),
            direction=(jd_profile.get("direction") or None),
            education_level=(jd_profile.get("education_level") or None),
            jd_text=jd_text or "",
            jd_hash=compute_jd_hash(jd_text or ""),
            jd_profile=jd_profile,
            index_text=index_text or "",
            requirements=list(jd_profile.get("responsibilities") or []),
            skills=skills,
            source=source or SOURCE_BATCH,
            embedding_id=job_id,
            cities=cf["cities_json"],
            job_types=cf["job_types_json"],
            min_degree_rank=cf["min_degree_rank"],
            city_status=cf["city_status"],
            education_status=cf["education_status"],
            status=status or STATUS_ACTIVE,
            source_job_id=source_job_id,
            canonical_url=canonical_url,
            collected_at=collected_at,
            last_verified_at=last_verified_at,
            content_hash=compute_content_hash(index_text or ""),
            parser_version=versions.PARSER_VERSION,
            embedding_version=versions.embedding_version(),
            index_text_version=versions.INDEX_TEXT_VERSION,
            quality_score=quality_score,
        )

    # ---------- 构造：SQLite 行 → 记录 ----------
    @classmethod
    def from_store_row(cls, row: dict) -> "JobRecord":
        """由 jobs_store 返回的行 dict 还原记录（*_json 字段按需反序列化）。"""
        import json

        def _loads(v, default):
            if isinstance(v, (list, dict)):
                return v
            if isinstance(v, str) and v.strip():
                try:
                    return json.loads(v)
                except Exception:
                    return default
            return default

        jd_profile = row.get("jd_profile")
        if not isinstance(jd_profile, dict):
            jd_profile = _loads(row.get("jd_profile_json"), {})
        return cls(
            job_id=row.get("job_id"),
            company=row.get("company") or "unknown_company",
            title=row.get("title") or "unknown_title",
            city=(row.get("city") or None),
            direction=(row.get("direction") or None),
            education_level=(row.get("education_level") or None),
            jd_text=row.get("jd_text") or "",
            jd_hash=row.get("jd_hash") or "",
            jd_profile=jd_profile if isinstance(jd_profile, dict) else {},
            index_text=row.get("index_text") or "",
            requirements=_loads(row.get("requirements"), []),
            skills=_loads(row.get("skills"), []),
            source=row.get("source") or SOURCE_BATCH,
            embedding_id=row.get("embedding_id"),
            cities=_loads(row.get("cities_json"), []),
            job_types=_loads(row.get("job_types_json"), []),
            min_degree_rank=row.get("min_degree_rank"),
            city_status=row.get("city_status") or "unknown",
            education_status=row.get("education_status") or "unknown",
            status=row.get("status") or STATUS_ACTIVE,
            source_job_id=row.get("source_job_id"),
            canonical_url=row.get("canonical_url"),
            collected_at=row.get("collected_at"),
            last_verified_at=row.get("last_verified_at"),
            content_hash=row.get("content_hash"),
            parser_version=row.get("parser_version"),
            embedding_version=row.get("embedding_version"),
            index_text_version=row.get("index_text_version"),
            quality_score=row.get("quality_score"),
        )

    # ---------- 写出：记录 → jobs_store.upsert_job 的入参 dict ----------
    def to_store_dict(self) -> dict:
        """转成 jobs_store.upsert_job 期望的 dict（cities→cities_json、job_types→job_types_json）。"""
        return {
            "job_id": self.job_id,
            "company": self.company,
            "title": self.title,
            "city": self.city or "",
            "direction": self.direction or "",
            "education_level": self.education_level or "",
            "jd_text": self.jd_text,
            "jd_hash": self.jd_hash,
            "jd_profile": self.jd_profile,
            "index_text": self.index_text,
            "requirements": self.requirements,
            "skills": self.skills,
            "source": self.source,
            "embedding_id": self.embedding_id or self.job_id,
            "cities_json": self.cities,
            "job_types_json": self.job_types,
            "min_degree_rank": self.min_degree_rank,
            "city_status": self.city_status,
            "education_status": self.education_status,
            "status": self.status,
            "source_job_id": self.source_job_id,
            "canonical_url": self.canonical_url,
            "content_hash": self.content_hash,
            "collected_at": self.collected_at,
            "last_verified_at": self.last_verified_at,
            "parser_version": self.parser_version,
            "embedding_version": self.embedding_version,
            "index_text_version": self.index_text_version,
            "quality_score": self.quality_score,
        }
