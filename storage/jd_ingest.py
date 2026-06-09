"""
storage/jd_ingest.py

JD 自动入库模块。

定位：
    用户粘贴一条 JD 后，在匹配评分之前先判断该 JD 是否已存在于岗位库。
    不存在则写入 SQLite jobs 表 + 生成 embedding 写入 Chroma；已存在则复用，不重复写入。

挂载位置：
    LangGraph 工作流的 jd_input_node 之后、job_analyzer_node 之前。

去重策略（两层，任一命中即判定重复）：
    1. 精确去重：SQL 按 (company, title, jd_hash) 查询；
    2. 语义去重：仅在第一层未命中时，用 embedding 在 Chroma 查最近邻，
       余弦相似度 > 阈值则判定重复。
"""

# ===== 1. import 区 =====
import os
import json
import sqlite3
import hashlib
from uuid import uuid4
from datetime import datetime, timezone

import chromadb

from state import AgentState
# 统一存储路径（单一事实来源）；CHROMA_DIR / COLLECTION_NAME 与岗位检索共用同一向量库
from storage.paths import SQLITE_PATH, CHROMA_DIR, COLLECTION_NAME


def _embed(jd_text: str) -> list:
    """对文本求 embedding（复用项目统一的 text-embedding-v3，与 job_indexer 同源）。返回 list[float]。"""
    from job_indexer import get_embedding  # 延迟导入，避免模块加载期触发网络/依赖
    return get_embedding(jd_text)


# ===== 2. 常量定义 =====
# SQLite（user_profiles / jobs）与岗位检索共用同一个 Chroma 库，统一在 storage/paths.py 管理。
DB_PATH = SQLITE_PATH
SIMILARITY_THRESHOLD = 0.92

# jobs 建表与索引 SQL
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    company       TEXT NOT NULL,
    title         TEXT NOT NULL,
    city          TEXT,
    jd_text       TEXT NOT NULL,
    jd_hash       TEXT NOT NULL,
    requirements  TEXT NOT NULL DEFAULT '[]',  -- JSON 数组字符串
    skills        TEXT NOT NULL DEFAULT '[]',  -- JSON 数组字符串
    source        TEXT NOT NULL DEFAULT 'user_uploaded',
    embedding_id  TEXT,                        -- 向量库中对应的 document id
    created_at    TEXT NOT NULL
);
"""
_CREATE_UNIQUE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_company_title_hash ON jobs(company, title, jd_hash);
"""
_CREATE_CITY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_jobs_city ON jobs(city);
"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _ensure_data_dir() -> None:
    """确保 data/ 目录存在。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _utc_now_iso() -> str:
    """当前 UTC 时间，ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _append_error(state: AgentState, error_obj: dict) -> AgentState:
    """把一条错误信息安全追加到 errors（不就地修改原列表），返回浅拷贝 State。"""
    new_state = dict(state)
    errors = list(state.get("errors") or [])
    errors.append(error_obj)
    new_state["errors"] = errors
    return new_state


# ===== 3. init_db =====
def init_db() -> None:
    """建 jobs 表与索引。幂等，可重复调用；失败仅打印警告，不抛异常。"""
    try:
        _ensure_data_dir()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_UNIQUE_INDEX_SQL)
            conn.execute(_CREATE_CITY_INDEX_SQL)
            conn.commit()
    except Exception as e:
        print(f"[jd_ingest] 警告：初始化 jobs 表失败：{e}")


# ===== 4. init_chroma =====
def init_chroma() -> "chromadb.Collection":
    """返回 jobs collection（不存在则新建）。

    与 job_indexer / job_retriever 共用同一个 collection，统一采用「写入时手动传入 embedding」
    的方式，因此 collection 不绑定 embedding_function（避免两套写入方式冲突）。
    """
    _ensure_data_dir()
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


# ===== 5. compute_jd_hash =====
def compute_jd_hash(jd_text: str) -> str:
    """对 jd_text.strip() 做 MD5，返回十六进制字符串。"""
    return hashlib.md5((jd_text or "").strip().encode("utf-8")).hexdigest()


# ===== 6. check_exact_duplicate =====
def check_exact_duplicate(company: str, title: str, jd_hash: str, conn) -> "str | None":
    """精确去重：按 (company, title, jd_hash) 查询，命中返回 job_id，否则 None。"""
    cur = conn.execute(
        "SELECT job_id FROM jobs WHERE company = ? AND title = ? AND jd_hash = ? LIMIT 1",
        (company, title, jd_hash),
    )
    row = cur.fetchone()
    return row[0] if row else None


# ===== 7. check_semantic_duplicate =====
def check_semantic_duplicate(embedding, collection) -> "tuple[str | None, float]":
    """语义去重：在 Chroma 查最近邻，L2 距离转余弦相似度后与阈值比较。

    注：Chroma 默认使用 L2 距离，相似度换算 similarity = 1 - distance / 2 在向量已归一化时成立；
        若 embedding 未归一化，此处为近似值。
    返回 (命中的 job_id 或 None, 相似度)；collection 为空 / 查询异常时返回 (None, 0.0)。
    """
    if collection is None:
        return (None, 0.0)
    try:
        result = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances"],
        )
    except Exception:
        return (None, 0.0)

    distances = result.get("distances") or [[]]
    ids = result.get("ids") or [[]]
    if not distances or not distances[0]:
        return (None, 0.0)

    distance = distances[0][0]
    similarity = 1 - distance / 2

    if similarity > SIMILARITY_THRESHOLD:
        existing_id = ids[0][0] if ids and ids[0] else None
        return (existing_id, similarity)
    return (None, similarity)


# ===== 8. write_to_sqlite =====
def write_to_sqlite(job_dict: dict, conn) -> None:
    """把一条 job 写入 SQLite；requirements / skills 序列化为 JSON 字符串。

    使用 INSERT OR IGNORE，防止并发写入时的主键 / 唯一索引冲突。
    """
    conn.execute(
        "INSERT OR IGNORE INTO jobs "
        "(job_id, company, title, city, jd_text, jd_hash, requirements, skills, source, embedding_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            job_dict["job_id"],
            job_dict["company"],
            job_dict["title"],
            job_dict.get("city", ""),
            job_dict["jd_text"],
            job_dict["jd_hash"],
            json.dumps(job_dict.get("requirements", []), ensure_ascii=False),
            json.dumps(job_dict.get("skills", []), ensure_ascii=False),
            job_dict.get("source", "user_uploaded"),
            job_dict.get("embedding_id"),
            job_dict["created_at"],
        ),
    )


# ===== 9. write_to_chroma =====
def write_to_chroma(job_id: str, jd_text: str, metadata: dict, collection) -> None:
    """把 JD 写入 Chroma，embedding 由 _embed 显式生成（与 job_indexer 写入方式一致）。"""
    collection.add(
        documents=[jd_text],
        embeddings=[_embed(jd_text)],
        metadatas=[metadata],
        ids=[job_id],
    )


# ===== 模块级 Chroma collection（导入时初始化一次）=====
try:
    _chroma_collection = init_chroma()
except Exception as e:
    print(f"[jd_ingest] 警告：初始化 Chroma 失败，语义去重 / 向量写入将被跳过：{e}")
    _chroma_collection = None


# ===== 10. jd_ingest_node（LangGraph 节点）=====
def jd_ingest_node(state: AgentState) -> AgentState:
    """JD 自动入库节点：精确去重 → 语义去重 → 写入 SQLite + Chroma。

    任一数据库 / 向量库操作异常都被捕获并追加到 state["errors"]，流程不中断；
    失败时 ingested_job_id 保持 ""、jd_is_duplicate 保持 False。
    """
    new_state = dict(state)  # 浅拷贝，不修改原有字段
    new_state.setdefault("ingested_job_id", "")
    new_state.setdefault("jd_is_duplicate", False)

    # ---- step 1：前置检查 ----
    jd_raw_text = state.get("jd_raw_text") or ""
    if not jd_raw_text:
        print("[jd_ingest] 警告：jd_raw_text 为空，跳过入库")
        return _append_error(
            new_state, {"node": "jd_ingest_node", "error": "jd_raw_text 为空，跳过入库"}
        )

    # ---- step 2：提取字段 ----
    job_analysis = state.get("job_analysis") or {}
    company = job_analysis.get("company") or ""
    title = job_analysis.get("title") or ""
    city = job_analysis.get("city") or ""
    required_skills = job_analysis.get("required_skills") or []
    responsibilities = job_analysis.get("responsibilities") or []

    if not company:
        company = "unknown_company"
        print("[jd_ingest] 警告：company 为空，使用 unknown_company")
    if not title:
        title = "unknown_title"
        print("[jd_ingest] 警告：title 为空，使用 unknown_title")

    # ---- step 3：计算 hash ----
    jd_hash = compute_jd_hash(jd_raw_text)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            # ---- step 4：第一层精确去重 ----
            existing_job_id = check_exact_duplicate(company, title, jd_hash, conn)
            if existing_job_id:
                new_state["ingested_job_id"] = existing_job_id
                new_state["jd_is_duplicate"] = True
                print(f"[jd_ingest] 精确重复，复用 job_id={existing_job_id}")
                return new_state

            # ---- step 5：生成 embedding ----
            try:
                embedding = _embed(jd_raw_text)
            except Exception as e:
                print(f"[jd_ingest] 错误：生成 embedding 失败：{e}")
                return _append_error(
                    new_state,
                    {"node": "jd_ingest_node", "error": str(e), "step": "embedding"},
                )

            # ---- step 6：第二层语义去重 ----
            existing_id, similarity = check_semantic_duplicate(embedding, _chroma_collection)
            if existing_id:
                new_state["ingested_job_id"] = existing_id
                new_state["jd_is_duplicate"] = True
                print(f"[jd_ingest] 检测到语义重复 JD，相似度={similarity:.3f}，复用 job_id={existing_id}")
                return new_state

            # ---- step 7：写入 SQLite 和 Chroma ----
            job_id = f"job_{uuid4().hex[:8]}"
            job_dict = {
                "job_id": job_id,
                "company": company,
                "title": title,
                "city": city,
                "jd_text": jd_raw_text,
                "jd_hash": jd_hash,
                "requirements": responsibilities,  # 职责 -> requirements
                "skills": required_skills,          # 技能 -> skills
                "source": "user_uploaded",
                "embedding_id": job_id,             # 与向量库 id 保持一致
                "created_at": _utc_now_iso(),
            }

            # 7a. 写入 SQLite
            try:
                write_to_sqlite(job_dict, conn)
                conn.commit()
            except Exception as e:
                print(f"[jd_ingest] 错误：写入 SQLite 失败：{e}")
                return _append_error(
                    new_state,
                    {"node": "jd_ingest_node", "error": str(e), "step": "sqlite"},
                )

            # 7b. 写入 Chroma
            try:
                metadata = {
                    "company": company,
                    "title": title,
                    "city": city,
                    "source": "user_uploaded",
                }
                write_to_chroma(job_id, jd_raw_text, metadata, _chroma_collection)
            except Exception as e:
                print(f"[jd_ingest] 错误：写入 Chroma 失败：{e}")
                return _append_error(
                    new_state,
                    {"node": "jd_ingest_node", "error": str(e), "step": "chroma"},
                )

            new_state["ingested_job_id"] = job_id
            new_state["jd_is_duplicate"] = False
            print(f"[jd_ingest] 新 JD 入库完成，job_id={job_id}，company={company}，title={title}")
            return new_state

    except Exception as e:
        # 兜底：SQLite 连接等未预期异常，不中断流程
        print(f"[jd_ingest] 错误：入库流程异常：{e}")
        return _append_error(
            new_state, {"node": "jd_ingest_node", "error": str(e), "step": "sqlite"}
        )


# 模块首次导入时建表（幂等）
init_db()
