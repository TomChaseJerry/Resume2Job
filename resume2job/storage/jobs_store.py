# -*- coding: utf-8 -*-
"""jobs 表统一读写（SQLite = 岗位数据的单一事实源）。

架构约定（标准「关系库做事实源、向量库做索引」分层）：
    - SQLite jobs 表存岗位的**全部**业务数据：原文、完整结构化画像（jd_profile_json）、
      检索文本（index_text）、哈希、来源，以及**硬约束召回前预筛派生列**
      （cities_json / job_types_json / min_degree_rank / city_status / education_status，
      由 jd_parser.derive_constraint_fields 入库时算好，供 get_eligible_jobs 资格预筛）；
    - Chroma 只存 embedding + job_id + 少量展示标量（city / direction / education_level）；
      **硬约束不在 Chroma 里 pre-filter**——召回前先用 SQLite eligibility 预筛得 allowed_job_ids，
      再用 Chroma where job_id $in 限定检索范围（BM25 通道用同一 id 集合过滤）；
    - 检索通道只返回 job_id + 分数，结果由本模块批量回填（hydration）；
    - 换 embedding 模型 = 从 SQLite 重建 Chroma（scripts/rebuild_index.py），零解析成本。

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
    -- 硬约束预过滤列（召回前 eligibility 用；由 jd_parser.derive_constraint_fields 入库时算好）
    cities_json     TEXT NOT NULL DEFAULT '[]',         -- 规范化城市数组（多城市岗多个）
    job_types_json  TEXT NOT NULL DEFAULT '[]',         -- 岗位类型数组（实习/校招/社招）
    min_degree_rank INTEGER,                            -- 最低学历 rank（本1硕2博3）；NULL=不限/未明确
    city_status     TEXT NOT NULL DEFAULT 'unknown',    -- explicit / unknown
    education_status TEXT NOT NULL DEFAULT 'unknown',   -- explicit / unrestricted / unknown
    -- 生命周期 / 版本 / 质量列（由 resume2job/ingest 写入；见 ingest/models.py、ingest/versions.py）
    status          TEXT NOT NULL DEFAULT 'active',     -- active / expired / removed（仅 active 可召回）
    source_job_id   TEXT,                               -- 来源系统外部 ID（增量同步 / 去重）
    canonical_url   TEXT,                               -- 岗位规范 URL（按 URL 去重 / 增量更新）
    content_hash    TEXT,                               -- index_text 内容哈希（驱动重嵌入判断）
    collected_at    TEXT,                               -- 来源采集时间（ISO）
    last_verified_at TEXT,                              -- 最近确认仍有效的时间（ISO）
    parser_version  TEXT,                               -- 解析器版本戳
    embedding_version TEXT,                             -- embedding 版本戳（= 模型名）
    index_text_version TEXT,                            -- index_text 拼接版本戳
    quality_score   REAL,                               -- 质量分 [0,1]（ingest/validator）
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
_CREATE_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""
_CREATE_URL_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_jobs_canonical_url ON jobs(canonical_url);
"""

# 旧库（2026-06 重构前建的表）缺这些列，init_db 时幂等补齐
_MIGRATION_COLUMNS = (
    ("direction", "TEXT"),
    ("education_level", "TEXT"),
    ("jd_profile_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("index_text", "TEXT NOT NULL DEFAULT ''"),
    ("updated_at", "TEXT"),
    # 硬约束预过滤列（2026-06-21 加；旧库 ALTER 补齐，再跑 indexer.backfill_constraint_fields 回填值）
    ("cities_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("job_types_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("min_degree_rank", "INTEGER"),
    ("city_status", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("education_status", "TEXT NOT NULL DEFAULT 'unknown'"),
    # 生命周期 / 版本 / 质量列（2026-06 加；旧库 ALTER 补齐，再跑 ingest.lifecycle.backfill_lifecycle_fields 回填值）
    ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ("source_job_id", "TEXT"),
    ("canonical_url", "TEXT"),
    ("content_hash", "TEXT"),
    ("collected_at", "TEXT"),
    ("last_verified_at", "TEXT"),
    ("parser_version", "TEXT"),
    ("embedding_version", "TEXT"),
    ("index_text_version", "TEXT"),
    ("quality_score", "REAL"),
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
            conn.execute(_CREATE_STATUS_INDEX_SQL)
            conn.execute(_CREATE_URL_INDEX_SQL)
            conn.commit()
    except Exception as e:
        print(f"[jobs_store] 警告：初始化 jobs 表失败：{e}")


def compute_jd_hash(jd_text: str) -> str:
    """对 jd_text.strip() 做 MD5，返回十六进制字符串（精确去重键 / 原文变更检测）。"""
    return hashlib.md5((jd_text or "").strip().encode("utf-8")).hexdigest()


def compute_content_hash(index_text: str) -> str:
    """对 index_text.strip() 做 MD5：归一后**检索文本**（送 embedding / BM25 的文本）的内容指纹。

    与 compute_jd_hash 分工：jd_hash 反映原文是否变；content_hash 反映检索文本是否变——
    解析结果或 index_text 拼接逻辑（INDEX_TEXT_VERSION）改了，即便原文一致它也会变，
    用于判断 Chroma 向量是否需要重嵌入。失败返回空串。
    """
    return hashlib.md5((index_text or "").strip().encode("utf-8")).hexdigest()


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
            "SELECT created_at, collected_at FROM jobs WHERE job_id = ?", (job["job_id"],)
        ).fetchone()
        created_at = (row["created_at"] if row else None) or job.get("created_at") or now
        # collected_at（来源采集时间）：入参优先 → 既有保留 → 回退 created_at（首次入库）
        collected_at = job.get("collected_at") or (row["collected_at"] if row else None) or created_at

        conn.execute(
            "INSERT OR REPLACE INTO jobs "
            "(job_id, company, title, city, direction, education_level, "
            " jd_text, jd_hash, jd_profile_json, index_text, "
            " requirements, skills, source, embedding_id, "
            " cities_json, job_types_json, min_degree_rank, city_status, education_status, "
            " status, source_job_id, canonical_url, content_hash, collected_at, last_verified_at, "
            " parser_version, embedding_version, index_text_version, quality_score, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                # 硬约束预过滤列（写入方经 jd_parser.derive_constraint_fields 算好传入；缺省给空/unknown）
                json.dumps(job.get("cities_json") or [], ensure_ascii=False),
                json.dumps(job.get("job_types_json") or [], ensure_ascii=False),
                job.get("min_degree_rank"),
                job.get("city_status") or "unknown",
                job.get("education_status") or "unknown",
                # 生命周期 / 版本 / 质量列（ingest 写入；缺省 status=active、last_verified_at=now）
                job.get("status") or "active",
                job.get("source_job_id"),
                job.get("canonical_url"),
                job.get("content_hash"),
                collected_at,
                job.get("last_verified_at") or now,
                job.get("parser_version"),
                job.get("embedding_version"),
                job.get("index_text_version"),
                job.get("quality_score"),
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
    """jobs 表总行数（轻量用途）。"""
    try:
        with _conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    except Exception:
        return 0


def corpus_signature() -> str:
    """可检索语料的轻量内容签名：对所有 index_text 非空岗位的 (job_id, jd_hash) 有序哈希。

    供 BM25 缓存键用，**替代 count()**——岗位内容**原地更新**（条数不变、jd_hash 变了）也能触发
    重建，避免「BM25 看旧 JD、dense 看新 JD」的双通道陈旧。百级语料每次召回算一次 MD5，开销可忽略。
    失败返回空串（调用方退化为「每次都重建」，不会用错语料）。
    """
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT job_id, jd_hash FROM jobs WHERE index_text != '' ORDER BY job_id"
            ).fetchall()
        h = hashlib.md5()
        for r in rows:
            h.update((r["job_id"] or "").encode("utf-8"))
            h.update(b"\x00")
            h.update((r["jd_hash"] or "").encode("utf-8"))
            h.update(b"\x01")
        return h.hexdigest()
    except Exception as e:
        print(f"[jobs_store] 警告：计算语料签名失败：{e}")
        return ""


def all_jobs_min() -> List[dict]:
    """取全部岗位的 {job_id, jd_profile, jd_text}（回填硬约束列 / 重判 job_type 用，纯 SQLite）。"""
    try:
        with _conn() as conn:
            rows = conn.execute("SELECT job_id, jd_profile_json, jd_text FROM jobs").fetchall()
        out = []
        for r in rows:
            try:
                prof = json.loads(r["jd_profile_json"] or "{}")
            except json.JSONDecodeError:
                prof = {}
            out.append({"job_id": r["job_id"], "jd_profile": prof if isinstance(prof, dict) else {},
                        "jd_text": r["jd_text"] or ""})
        return out
    except Exception as e:
        print(f"[jobs_store] 警告：读取全量画像失败：{e}")
        return []


def get_eligible_jobs(city, user_degree_rank, job_type) -> Dict[str, dict]:
    """硬约束资格预筛（召回**前**）：返回 {job_id: {city_match_status, education_match_status}}。

    取原语而非约束对象（storage 层不依赖 retrieval）。规则：
      - 仅 index_text 非空（已可检索）的岗位；
      - 城市：用户指定时保留「cities 含该城市」或「city_status=unknown（地点待确认）」；未指定不卡；
      - 学历：保留「无明确要求（min_degree_rank IS NULL，即 unrestricted/unknown）」或「要求 ≤ 候选人学历」；
              候选人学历未知（rank=None）时不卡；
      - 岗位类型：job_types 含该类型（数组为空的旧数据宽松保留）。
    city_match_status：exact（命中或未限城市）/ unknown（靠 city_status=unknown 进池，地点待确认）；
    education_match_status：satisfied（明确要求且 ≤ 候选人）/ unverified（不限/未知/候选人学历未知）。
    """
    want_city = (str(city).strip() if city else None)
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT job_id, cities_json, min_degree_rank FROM jobs "
                "WHERE index_text != '' AND status = 'active' "
                "  AND (:rank IS NULL OR min_degree_rank IS NULL OR min_degree_rank <= :rank) "
                "  AND (job_types_json = '[]' "
                "       OR EXISTS (SELECT 1 FROM json_each(job_types_json) WHERE value = :jt)) "
                "  AND (:city IS NULL OR city_status = 'unknown' "
                "       OR EXISTS (SELECT 1 FROM json_each(cities_json) WHERE value = :city))",
                {"rank": user_degree_rank, "jt": job_type, "city": want_city},
            ).fetchall()
    except Exception as e:
        print(f"[jobs_store] 警告：eligibility 预筛失败：{e}")
        return {}

    out: Dict[str, dict] = {}
    for r in rows:
        try:
            cities = json.loads(r["cities_json"] or "[]")
        except json.JSONDecodeError:
            cities = []
        city_match = "exact" if (not want_city or want_city in cities) else "unknown"
        mr = r["min_degree_rank"]
        edu_match = ("satisfied" if (mr is not None and user_degree_rank is not None
                                     and mr <= user_degree_rank) else "unverified")
        out[r["job_id"]] = {"city_match_status": city_match, "education_match_status": edu_match}
    return out


def update_constraint_fields(job_id: str, cities_json, job_types_json,
                             min_degree_rank, city_status, education_status) -> None:
    """回填/更新单条岗位的硬约束预过滤列（只动这 5 列，不碰其他列）。"""
    if not job_id:
        return
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE jobs SET cities_json=?, job_types_json=?, min_degree_rank=?, "
                "city_status=?, education_status=? WHERE job_id=?",
                (json.dumps(cities_json or [], ensure_ascii=False),
                 json.dumps(job_types_json or [], ensure_ascii=False),
                 min_degree_rank, city_status or "unknown", education_status or "unknown",
                 job_id),
            )
            conn.commit()
    except Exception as e:
        print(f"[jobs_store] 警告：回填约束列失败 {job_id}：{e}")


# ---------------------------------------------------------------------------
# 生命周期 / 去重 / 回填（ingest 模块用）
# ---------------------------------------------------------------------------
def get_job_id_by_canonical_url(url: str) -> Optional[str]:
    """按岗位规范 URL 查 job_id（增量同步 / URL 去重）；未命中返回 None。"""
    if not url:
        return None
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT job_id FROM jobs WHERE canonical_url = ? LIMIT 1", (url,)
            ).fetchone()
        return row["job_id"] if row else None
    except Exception:
        return None


def get_job_id_by_source(source: str, source_job_id: str) -> Optional[str]:
    """按 (source, source_job_id) 查 job_id（来源系统外部 ID 去重）；未命中返回 None。"""
    if not source or not source_job_id:
        return None
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT job_id FROM jobs WHERE source = ? AND source_job_id = ? LIMIT 1",
                (source, source_job_id),
            ).fetchone()
        return row["job_id"] if row else None
    except Exception:
        return None


def touch_verified(job_id: str, now_iso: Optional[str] = None) -> None:
    """确认岗位仍有效：status→active、刷新 last_verified_at（幂等接入的轻量更新，不动 updated_at）。"""
    if not job_id:
        return
    now = now_iso or _utc_now_iso()
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE jobs SET status='active', last_verified_at=? WHERE job_id=?",
                (now, job_id),
            )
            conn.commit()
    except Exception as e:
        print(f"[jobs_store] 警告：touch_verified 失败 {job_id}：{e}")


def set_status(job_id: str, status: str, now_iso: Optional[str] = None) -> None:
    """设置岗位生命周期状态（active/expired/removed）；置 active 时可同时刷新 last_verified_at。"""
    if not job_id or not status:
        return
    try:
        with _conn() as conn:
            if now_iso:
                conn.execute("UPDATE jobs SET status=?, last_verified_at=? WHERE job_id=?",
                             (status, now_iso, job_id))
            else:
                conn.execute("UPDATE jobs SET status=? WHERE job_id=?", (status, job_id))
            conn.commit()
    except Exception as e:
        print(f"[jobs_store] 警告：set_status 失败 {job_id}：{e}")


def update_embedding_version(job_id: str, embedding_version: Optional[str]) -> None:
    """写入岗位对应向量的 embedding 版本戳（Chroma 写入成功后调用）。"""
    if not job_id:
        return
    try:
        with _conn() as conn:
            conn.execute("UPDATE jobs SET embedding_version=? WHERE job_id=?",
                         (embedding_version, job_id))
            conn.commit()
    except Exception as e:
        print(f"[jobs_store] 警告：update_embedding_version 失败 {job_id}：{e}")


def expire_stale(cutoff_iso: str) -> int:
    """把 active 且 last_verified_at 早于 cutoff 的岗位标记为 expired，返回条数。

    last_verified_at 为 NULL（从未验证 / 未回填）的不动——避免在回填前误过期旧库。
    """
    try:
        with _conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status='expired' "
                "WHERE status='active' AND last_verified_at IS NOT NULL AND last_verified_at < ?",
                (cutoff_iso,),
            )
            conn.commit()
            return cur.rowcount or 0
    except Exception as e:
        print(f"[jobs_store] 警告：expire_stale 失败：{e}")
        return 0


def update_lifecycle_fields(job_id: str, *, status=None, content_hash=None,
                            parser_version=None, embedding_version=None,
                            index_text_version=None, quality_score=None,
                            collected_at=None, last_verified_at=None) -> None:
    """回填 / 更新生命周期 + 版本 + 质量列（只动给定的非 None 列）。"""
    if not job_id:
        return
    cols = [("status", status), ("content_hash", content_hash),
            ("parser_version", parser_version), ("embedding_version", embedding_version),
            ("index_text_version", index_text_version), ("quality_score", quality_score),
            ("collected_at", collected_at), ("last_verified_at", last_verified_at)]
    sets, vals = [], []
    for col, val in cols:
        if val is not None:
            sets.append(f"{col}=?")
            vals.append(val)
    if not sets:
        return
    vals.append(job_id)
    try:
        with _conn() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id=?", vals)
            conn.commit()
    except Exception as e:
        print(f"[jobs_store] 警告：update_lifecycle_fields 失败 {job_id}：{e}")


def all_rows() -> List[dict]:
    """取全部岗位的完整行（含反序列化 jd_profile）；生命周期回填 / 审计用。"""
    try:
        with _conn() as conn:
            rows = conn.execute("SELECT * FROM jobs").fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        print(f"[jobs_store] 警告：读取全量行失败：{e}")
        return []


# 模块首次导入时建表（幂等，与 profile_cache / conversation_store 约定一致）
init_db()
