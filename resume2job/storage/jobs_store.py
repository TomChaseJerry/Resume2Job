# -*- coding: utf-8 -*-
"""jobs 表统一读写（SQLite = 岗位数据的单一事实源）。

架构约定（标准「关系库做事实源、向量库做索引」分层）：
    - SQLite jobs 表存岗位的**全部**业务数据：原文、完整结构化画像
      （jd_profile_json）、检索文本（index_text）、可过滤字段、哈希、来源；
    - Chroma 只存 embedding + job_id + 少量可过滤标量（city / direction /
      education_level，pre-filter 必须发生在向量库内部，属正常反规范化投影）；
    - 检索通道只返回 job_id + 分数，结果由本模块批量回填（hydration）；
    - 换 embedding 模型 = 从 SQLite 重建 Chroma（scripts/rebuild_index.py），
      零解析成本。

所有写入方（批量建库 retrieval/indexer、运行时 storage/jd_ingest）必须经由
upsert_job 写入，禁止旁路写 jobs 表。
"""

import os
import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from resume2job.storage.paths import SQLITE_PATH

# 模块级路径，便于验收脚本（pipeline.py）monkeypatch 到隔离目录
DB_PATH = SQLITE_PATH

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    company         TEXT NOT NULL,
    title           TEXT NOT NULL,
    city            TEXT,
    direction       TEXT,
    education_level TEXT,
    jd_text         TEXT NOT NULL,
    jd_hash         TEXT NOT NULL,
    jd_profile_json TEXT NOT NULL DEFAULT '{}',  -- 完整结构化画像（事实源）
    index_text      TEXT NOT NULL DEFAULT '',    -- 检索文本（向量/BM25 共用语料）
    requirements    TEXT NOT NULL DEFAULT '[]',  -- JSON 数组字符串
    skills          TEXT NOT NULL DEFAULT '[]',  -- JSON 数组字符串
    source          TEXT NOT NULL DEFAULT 'user_uploaded',
    embedding_id    TEXT,                        -- 向量库中对应的 document id
    created_at      TEXT NOT NULL,
    updated_at      TEXT
);
"""
_CREATE_UNIQUE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_company_title_hash ON jobs(company, title, jd_hash);
"""
_CREATE_CITY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_jobs_city ON jobs(city);
"""
_CREATE_HASH_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(jd_hash);
"""

# 旧库（2026-06 重构前建的表）缺这些列，init_db 时幂等补齐
_MIGRATION_COLUMNS = (
    ("direction", "TEXT"),
    ("education_level", "TEXT"),
    ("jd_profile_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("index_text", "TEXT NOT NULL DEFAULT ''"),
    ("updated_at", "TEXT"),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    """获取连接；确保 data/ 目录存在，行结果以 sqlite3.Row 返回。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建 jobs 表与索引，并为旧库补齐新列。幂等；失败仅打印警告。"""
    try:
        with _conn() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            # 旧表迁移：逐列 ALTER，已存在则忽略
            for col, decl in _MIGRATION_COLUMNS:
                try:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")
                except sqlite3.OperationalError:
                    pass  # duplicate column name -> 已是新表
            conn.execute(_CREATE_UNIQUE_INDEX_SQL)
            conn.execute(_CREATE_CITY_INDEX_SQL)
            conn.execute(_CREATE_HASH_INDEX_SQL)
            conn.commit()
    except Exception as e:
        print(f"[jobs_store] 警告：初始化 jobs 表失败：{e}")


def compute_jd_hash(jd_text: str) -> str:
    """对 jd_text.strip() 做 MD5，返回十六进制字符串（精确去重键）。"""
    return hashlib.md5((jd_text or "").strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------
def upsert_job(job: dict) -> None:
    """写入/更新一条岗位（事实源唯一写入口）。

    job 必填：job_id, company, title, jd_text, jd_hash；
    建议提供：jd_profile（dict）、index_text、city/direction/education_level、
              requirements/skills（list）、source。
    已存在的 job_id 整行覆盖（画像更新即生效），created_at 保留首次写入值。
    """
    jd_profile = job.get("jd_profile")
    if not isinstance(jd_profile, dict):
        jd_profile = {}
    now = _utc_now_iso()

    with _conn() as conn:
        row = conn.execute(
            "SELECT created_at FROM jobs WHERE job_id = ?", (job["job_id"],)
        ).fetchone()
        created_at = row["created_at"] if row else (job.get("created_at") or now)

        conn.execute(
            "INSERT OR REPLACE INTO jobs "
            "(job_id, company, title, city, direction, education_level, "
            " jd_text, jd_hash, jd_profile_json, index_text, "
            " requirements, skills, source, embedding_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job["job_id"],
                job.get("company") or "unknown_company",
                job.get("title") or "unknown_title",
                job.get("city") or "",
                job.get("direction") or "",
                job.get("education_level") or "",
                job.get("jd_text") or "",
                job["jd_hash"],
                json.dumps(jd_profile, ensure_ascii=False),
                job.get("index_text") or "",
                json.dumps(job.get("requirements") or [], ensure_ascii=False),
                json.dumps(job.get("skills") or [], ensure_ascii=False),
                job.get("source") or "user_uploaded",
                job.get("embedding_id") or job["job_id"],
                created_at,
                now,
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------
def _row_to_dict(row: sqlite3.Row) -> dict:
    """行 → dict，并反序列化 jd_profile_json。"""
    d = dict(row)
    try:
        d["jd_profile"] = json.loads(d.get("jd_profile_json") or "{}")
    except json.JSONDecodeError:
        d["jd_profile"] = {}
    if not isinstance(d["jd_profile"], dict):
        d["jd_profile"] = {}
    return d


def get_job(job_id: str) -> Optional[dict]:
    """按 job_id 取整行（含反序列化的 jd_profile）；不存在返回 None。"""
    if not job_id:
        return None
    try:
        with _conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _row_to_dict(row) if row else None
    except Exception as e:
        print(f"[jobs_store] 警告：读取 job {job_id} 失败：{e}")
        return None


def get_jobs_by_ids(job_ids: List[str]) -> Dict[str, dict]:
    """批量取岗位（检索结果回填用），返回 {job_id: row_dict}。"""
    ids = [j for j in (job_ids or []) if j]
    if not ids:
        return {}
    try:
        with _conn() as conn:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE job_id IN ({placeholders})", ids
            ).fetchall()
        return {row["job_id"]: _row_to_dict(row) for row in rows}
    except Exception as e:
        print(f"[jobs_store] 警告：批量读取 jobs 失败：{e}")
        return {}


def get_job_id_by_exact(company: str, title: str, jd_hash: str) -> Optional[str]:
    """精确去重：按 (company, title, jd_hash) 查询，命中返回 job_id。"""
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT job_id FROM jobs WHERE company = ? AND title = ? AND jd_hash = ? LIMIT 1",
                (company, title, jd_hash),
            ).fetchone()
        return row["job_id"] if row else None
    except Exception:
        return None


def get_profile_by_hash(jd_hash: str) -> Optional[dict]:
    """按 JD 原文哈希取已入库画像（跨链路一致性复用，零 token）。

    返回 {"job_id", "jd_profile"} 或 None；画像为空 dict 时视为未命中。
    """
    if not jd_hash:
        return None
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT job_id, jd_profile_json FROM jobs WHERE jd_hash = ? LIMIT 1",
                (jd_hash,),
            ).fetchone()
        if not row:
            return None
        profile = json.loads(row["jd_profile_json"] or "{}")
        if isinstance(profile, dict) and profile:
            return {"job_id": row["job_id"], "jd_profile": profile}
    except Exception as e:
        print(f"[jobs_store] 警告：按哈希查画像失败：{e}")
    return None


def all_jobs_for_index() -> List[dict]:
    """取全部岗位的索引视图（BM25 语料 / 重建向量库用）。

    返回 [{job_id, index_text, city, direction, education_level, company, title}]，
    只含 index_text 非空的行。
    """
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT job_id, index_text, city, direction, education_level, company, title "
                "FROM jobs WHERE index_text != '' ORDER BY job_id"
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[jobs_store] 警告：读取索引视图失败：{e}")
        return []


def count() -> int:
    """jobs 表总行数（BM25 缓存键等轻量用途）。"""
    try:
        with _conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    except Exception:
        return 0


# 模块首次导入时建表（幂等，与 profile_cache / conversation_store 约定一致）
init_db()
