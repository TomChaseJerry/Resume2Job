# -*- coding: utf-8 -*-
"""
storage/jd_ingest.py

JD 自动入库模块（LangGraph 节点，挂在 jd_input 之后）。

定位：
    用户粘贴一条 JD 后，在匹配评分之前先判断该 JD 是否已存在于岗位库。
    不存在则写入 SQLite jobs 表（事实源，含完整画像）+ Chroma（向量 + 最小
    可过滤 metadata）；已存在则复用，不重复写入。

去重策略（两层，任一命中即判定重复）：
    1. 精确去重：SQLite 按 (company, title, jd_hash) 查询；
    2. 语义去重：仅在第一层未命中时，用 embedding 在 Chroma 查最近邻，
       余弦相似度 > 阈值则判定重复。

存储分层（与 retrieval/indexer 一致，见 storage/jobs_store.py 模块说明）：
    - SQLite：jd_text / jd_profile_json / index_text / 全部业务字段；
    - Chroma：embedding + document(index_text) + {city, direction, education_level} 等过滤标量。
"""

import os

import chromadb

from resume2job.agent.state import AgentState
from resume2job.storage.paths import CHROMA_DIR, COLLECTION_NAME
from resume2job.storage import jobs_store
from resume2job.storage.jobs_store import compute_jd_hash
from resume2job.parsing.jd_parser import derive_constraint_fields  # 硬约束预过滤列派生

# 语义去重阈值（归一化向量下 cosine = 1 - L2²/2）
SIMILARITY_THRESHOLD = 0.92


def _embed(text: str) -> list:
    """对文本求 embedding（统一走 core.llm）。返回 list[float]。"""
    from resume2job.core.llm import get_embedding  # 延迟导入，避免模块加载期触发网络/依赖
    return get_embedding(text)


def _append_error(state: AgentState, message: str) -> AgentState:
    """把一条错误信息安全追加到 errors（不就地修改原列表），返回浅拷贝 State。"""
    new_state = dict(state)
    errors = list(state.get("errors") or [])
    errors.append(message)
    new_state["errors"] = errors
    return new_state


def init_chroma() -> "chromadb.Collection":
    """返回 jobs collection（不存在则新建）。

    与批量建库 / 检索共用同一个 collection，统一采用「写入时手动传入 embedding」
    的方式，因此 collection 不绑定 embedding_function（避免两套写入方式冲突）。
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


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


# ===== 模块级 Chroma collection（导入时初始化一次）=====
try:
    _chroma_collection = init_chroma()
except Exception as e:
    print(f"[jd_ingest] 警告：初始化 Chroma 失败，语义去重 / 向量写入将被跳过：{e}")
    _chroma_collection = None


# ===== jd_ingest_node（LangGraph 节点）=====
def jd_ingest_node(state: AgentState) -> AgentState:
    """JD 自动入库节点：精确去重 → 语义去重 → 写入 SQLite（事实源）+ Chroma（索引）。

    输入：state["jd_text"]（用户粘贴的 JD 原文）+ state["jd_profiles"][0]（jd_input 已解析）。
    任一数据库 / 向量库操作异常都被捕获并追加到 state["errors"]，流程不中断；
    失败时 ingested_job_id 保持 ""、jd_is_duplicate 保持 False。
    """
    from uuid import uuid4

    new_state = dict(state)  # 浅拷贝，不修改原有字段
    new_state.setdefault("ingested_job_id", "")
    new_state.setdefault("jd_is_duplicate", False)

    # 仅在「用户粘贴 JD」链路上工作；推荐链路（无 jd_text）静默跳过
    jd_text = state.get("jd_text") or ""
    if not jd_text.strip():
        return state

    print("[jd_ingest_node] 开始执行...")

    # ---- step 1：提取结构化字段（jd_input_node 已写入 jd_profiles）----
    jd_profiles = state.get("jd_profiles") or []
    jd_profile = jd_profiles[0] if jd_profiles and isinstance(jd_profiles[0], dict) else {}
    if not jd_profile:
        print("[jd_ingest] 警告：缺少结构化 jd_profile，跳过入库")
        return _append_error(new_state, "jd_ingest_node: 缺少结构化 jd_profile，跳过入库")

    location = jd_profile.get("location") if isinstance(jd_profile.get("location"), dict) else {}
    company = jd_profile.get("company") or "unknown_company"
    title = jd_profile.get("title") or "unknown_title"

    # ---- step 2：第一层精确去重（SQLite 事实源）----
    jd_hash = compute_jd_hash(jd_text)
    try:
        existing_job_id = jobs_store.get_job_id_by_exact(company, title, jd_hash)
        if existing_job_id:
            new_state["ingested_job_id"] = existing_job_id
            new_state["jd_is_duplicate"] = True
            print(f"[jd_ingest] 精确重复，复用 job_id={existing_job_id}")
            return new_state
    except Exception as e:
        print(f"[jd_ingest] 错误：精确去重查询失败：{e}")
        return _append_error(new_state, f"jd_ingest_node: 精确去重查询失败：{e}")

    # ---- step 3：生成 embedding（与批量建库一致：对 index_text 求向量）----
    # 延迟导入 indexer 工具，避免模块级循环依赖（indexer 也 import 本模块相关函数）
    from resume2job.retrieval.indexer import build_index_text, build_chroma_metadata

    index_text = build_index_text(jd_profile) or jd_text
    try:
        embedding = _embed(index_text)
    except Exception as e:
        print(f"[jd_ingest] 错误：生成 embedding 失败：{e}")
        return _append_error(new_state, f"jd_ingest_node: 生成 embedding 失败：{e}")

    # ---- step 4：第二层语义去重（Chroma 最近邻）----
    existing_id, similarity = check_semantic_duplicate(embedding, _chroma_collection)
    if existing_id:
        new_state["ingested_job_id"] = existing_id
        new_state["jd_is_duplicate"] = True
        print(f"[jd_ingest] 检测到语义重复 JD，相似度={similarity:.3f}，复用 job_id={existing_id}")
        return new_state

    # ---- step 5：写入 SQLite（事实源）----
    job_id = f"job_{uuid4().hex[:8]}"
    skills = list(dict.fromkeys([*(jd_profile.get("hard_skills") or []),
                                 *(jd_profile.get("tools_or_frameworks") or [])]))
    try:
        jobs_store.upsert_job({
            "job_id": job_id,
            "company": company,
            "title": title,
            "city": location.get("city") or "",
            "direction": jd_profile.get("direction") or "",
            "education_level": jd_profile.get("education_level") or "",
            "jd_text": jd_text,
            "jd_hash": jd_hash,
            "jd_profile": jd_profile,
            "index_text": index_text,
            "requirements": jd_profile.get("responsibilities") or [],
            "skills": skills,
            "source": "user_uploaded",
            # 硬约束预过滤列（cities_json/job_types_json/min_degree_rank/city_status/education_status）
            **derive_constraint_fields(jd_profile),
        })
    except Exception as e:
        print(f"[jd_ingest] 错误：写入 SQLite 失败：{e}")
        return _append_error(new_state, f"jd_ingest_node: 写入 SQLite 失败：{e}")

    # ---- step 6：写入 Chroma（向量 + 最小可过滤 metadata）----
    try:
        _chroma_collection.add(
            ids=[job_id],
            embeddings=[embedding],
            documents=[index_text],
            metadatas=[build_chroma_metadata(job_id, jd_profile, source="user_uploaded")],
        )
    except Exception as e:
        # SQLite 已写入；向量缺失可由 scripts/rebuild_index.py 补建
        print(f"[jd_ingest] 错误：写入 Chroma 失败：{e}")
        return _append_error(new_state, f"jd_ingest_node: 写入 Chroma 失败（SQLite 已入库，可用 rebuild_index 补建向量）：{e}")

    new_state["ingested_job_id"] = job_id
    new_state["jd_is_duplicate"] = False
    print(f"[jd_ingest] 新 JD 入库完成，job_id={job_id}，company={company}，title={title}")
    return new_state
