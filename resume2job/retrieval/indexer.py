"""
岗位知识库构建模块（Job Indexer）

批量解析 jd_folder 下的 .txt JD 文件，向量化后写入本地 ChromaDB（语义检索），
同时把结构化元数据写入 SQLite jobs 表（精确查询 / 去重），两库并存。

流程：
    扫描 .txt -> 读取原文 -> 用 job_id 查重
        ├─ 已在向量库：跳过向量化，仅从向量库 metadata 复用画像回填 SQLite jobs 表
        └─ 新条目：parse_jd -> 构造 index_text + metadata -> 获取 embedding
                    -> 写入 chromadb -> 写入 SQLite jobs 表
    -> 汇总日志
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime, timezone
from typing import Optional

import chromadb

# 复用已有的 JD 解析器
from resume2job.parsing.jd_parser import parse_jd
# 统一 Embedding（core 层单一事实来源）
from resume2job.core.llm import get_embedding
# 统一存储路径 + jobs 表写入工具（与运行时 jd_ingest_node 共用同一张表）
from resume2job.storage.paths import SQLITE_PATH
from resume2job.storage.jd_ingest import write_to_sqlite, compute_jd_hash


# ===== 常量 =====
# 统一向量库：与 JD 入库（storage/jd_ingest.py）、画像缓存同处 data/，见 storage/paths.py
from resume2job.storage.paths import CHROMA_DIR as DEFAULT_DB_PATH, COLLECTION_NAME as DEFAULT_COLLECTION_NAME
DEFAULT_JOB_PATH = './JDs'


# ===== index_text 构建 =====
def _join_list_field(items, sep: str = "、") -> str:
    """把列表字段拼接为可读字符串；非列表返回空。"""
    if not isinstance(items, list):
        return ""
    cleaned = [str(x).strip() for x in items if isinstance(x, (str, int, float)) and str(x).strip()]
    return sep.join(cleaned)


def build_index_text(jd_profile: dict) -> str:
    """从 jd_profile 拼接一段中文可读的检索文本。
    缺失字段直接跳过，避免输出 'None'。
    """
    if not isinstance(jd_profile, dict):
        return ""

    parts = []

    def _add(label: str, value: str):
        if value and str(value).strip():
            parts.append(f"{label}：{value}")

    _add("公司", jd_profile.get("company") or "")
    _add("岗位", jd_profile.get("title") or "")
    _add("岗位类型", jd_profile.get("job_type") or "")
    _add("方向", jd_profile.get("direction") or "")
    _add("业务场景", jd_profile.get("business_area") or "")
    _add("硬技能", _join_list_field(jd_profile.get("hard_skills")))
    _add("工具框架", _join_list_field(jd_profile.get("tools_or_frameworks")))
    _add("领域关键词", _join_list_field(jd_profile.get("domain_keywords")))
    _add("加分项", _join_list_field(jd_profile.get("preferred_skills")))

    # 岗位职责取前 3 条
    resp = jd_profile.get("responsibilities") or []
    if isinstance(resp, list) and resp:
        top3 = [str(x).strip() for x in resp[:3] if isinstance(x, (str, int, float)) and str(x).strip()]
        if top3:
            _add("岗位职责", "；".join(top3))

    _add("学历要求", jd_profile.get("education_requirement") or "")
    _add("经验要求", jd_profile.get("experience_requirement") or "")

    return "\n".join(parts)


# ===== metadata 构建（扁平化 + 列表 -> JSON 字符串）=====
def _meta_str(value) -> Optional[str]:
    """metadata 字符串字段安全转换：非空字符串返回 strip 后值，其余返回 None。"""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _meta_list_json(value) -> str:
    """列表字段统一存为 JSON 字符串，便于回读。"""
    if not isinstance(value, list):
        return "[]"
    cleaned = [v for v in value if v is not None]
    return json.dumps(cleaned, ensure_ascii=False)


def build_metadata(job_id: str, source_file: str, jd_profile: dict) -> dict:
    """构建写入 ChromaDB 的扁平 metadata。
    Chroma 限制值类型只能是 str/int/float/bool/None，因此：
      - location.* 展开为 city / district / office_address；
      - hard_skills / domain_keywords / tools_or_frameworks 转 JSON 字符串；
      - 完整 jd_profile 也以 JSON 字符串形式存在 jd_profile_json。
    """
    if not isinstance(jd_profile, dict):
        jd_profile = {}

    location = jd_profile.get("location") if isinstance(jd_profile.get("location"), dict) else {}

    meta = {
        "job_id": job_id,
        "source_file": source_file,
        "company": _meta_str(jd_profile.get("company")),
        "title": _meta_str(jd_profile.get("title")),
        "job_type": _meta_str(jd_profile.get("job_type")),
        "direction": _meta_str(jd_profile.get("direction")),
        "business_area": _meta_str(jd_profile.get("business_area")),
        "education_requirement": _meta_str(jd_profile.get("education_requirement")),
        "education_level": _meta_str(jd_profile.get("education_level")),
        "experience_requirement": _meta_str(jd_profile.get("experience_requirement")),
        "salary": _meta_str(jd_profile.get("salary")),
        "city": _meta_str(location.get("city")),
        "district": _meta_str(location.get("district")),
        "office_address": _meta_str(location.get("office_address")),
        "hard_skills": _meta_list_json(jd_profile.get("hard_skills")),
        "domain_keywords": _meta_list_json(jd_profile.get("domain_keywords")),
        "tools_or_frameworks": _meta_list_json(jd_profile.get("tools_or_frameworks")),
        "jd_profile_json": json.dumps(jd_profile, ensure_ascii=False),
    }

    # ChromaDB 1.x（Rust 后端）的 metadata 值不接受 None，否则报
    # "Cannot convert Python object to MetadataValue"。
    # 因此剔除所有 None 字段，缺失即不写入该 key。
    return {k: v for k, v in meta.items() if v is not None}


# ===== SQLite jobs 表回填 =====
def _utc_now_iso() -> str:
    """当前 UTC 时间，ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _profile_from_chroma(collection, job_id: str) -> Optional[dict]:
    """从向量库 metadata 的 jd_profile_json 复用已解析画像（用于回填已入库条目，零 token）。"""
    try:
        res = collection.get(ids=[job_id], include=["metadatas"])
        metas = res.get("metadatas") or []
        if metas and metas[0]:
            raw = metas[0].get("jd_profile_json")
            if raw:
                return json.loads(raw)
    except Exception as e:
        print(f"[WARN] 从向量库读取 jd_profile 失败（{job_id}）：{e}")
    return None


def lookup_ingested_jd_profile(jd_text: str) -> Optional[dict]:
    """按 JD 原文哈希在已入库岗位中查找并复用其结构化画像（零 token、跨链路一致性）。

    用途：用户直接粘贴的 JD 若与知识库中某条已入库 JD 文本完全一致，则复用入库时
    解析好的同一份 jd_profile，使「单 JD 评估」与「岗位推荐」两条链路看到**完全相同**
    的结构化画像，避免因重复解析产生字段差异导致同一 JD 评分不一致。

    流程：compute_jd_hash → SQLite jobs 按 jd_hash 取 job_id → 向量库 metadata 取 jd_profile。
    任何环节失败或未命中都返回 None，由调用方回退到实时 parse_jd。
    """
    if not isinstance(jd_text, str) or not jd_text.strip():
        return None
    try:
        jd_hash = compute_jd_hash(jd_text)
        with sqlite3.connect(SQLITE_PATH) as conn:
            row = conn.execute(
                "SELECT job_id FROM jobs WHERE jd_hash = ? LIMIT 1", (jd_hash,)
            ).fetchone()
        if not row or not row[0]:
            return None
        job_id = row[0]
        client = chromadb.PersistentClient(path=DEFAULT_DB_PATH)
        collection = client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)
        profile = _profile_from_chroma(collection, job_id)
        if isinstance(profile, dict) and profile:
            return profile
    except Exception as e:
        print(f"[WARN] 复用已入库 JD 画像失败，回退实时解析：{e}")
    return None


def backfill_jobs_table(job_id: str, jd_text: str, jd_profile: dict) -> None:
    """把一条 JD 的结构化信息写入 SQLite jobs 表，与 Chroma 向量库并存。

    幂等：write_to_sqlite 用 INSERT OR IGNORE，已存在则跳过；
    失败仅打印警告，不影响向量入库主流程。source 标记为 batch_indexed，
    便于与运行时用户粘贴入库（user_uploaded）区分来源。
    """
    if not isinstance(jd_profile, dict) or not jd_profile:
        return
    try:
        location = jd_profile.get("location") if isinstance(jd_profile.get("location"), dict) else {}
        hard = jd_profile.get("hard_skills") or []
        tools = jd_profile.get("tools_or_frameworks") or []
        skills = list(dict.fromkeys([*hard, *tools]))  # 合并去重、保序
        job_dict = {
            "job_id": job_id,
            "company": jd_profile.get("company") or "unknown_company",
            "title": jd_profile.get("title") or "unknown_title",
            "city": location.get("city") or "",
            "jd_text": jd_text,
            "jd_hash": compute_jd_hash(jd_text),
            "requirements": jd_profile.get("responsibilities") or [],
            "skills": skills,
            "source": "batch_indexed",
            "embedding_id": job_id,
            "created_at": _utc_now_iso(),
        }
        with sqlite3.connect(SQLITE_PATH) as conn:
            write_to_sqlite(job_dict, conn)
            conn.commit()
    except Exception as e:
        print(f"[WARN] 写入 SQLite jobs 表失败（{job_id}）：{e}")


# ===== job_id 查重 =====
def job_exists(collection, job_id: str) -> bool:
    """查询 ChromaDB 中是否已存在指定 job_id。"""
    try:
        result = collection.get(ids=[job_id])
    except Exception as e:
        print(f"[WARN] 查询 job_id 失败（视为不存在）：{e}")
        return False
    ids = result.get("ids") if isinstance(result, dict) else None
    return bool(ids)


# ===== 主流程 =====
def index_jobs(jd_folder: str, db_path: str = DEFAULT_DB_PATH,
               collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    """遍历 jd_folder 下 .txt 文件，逐条解析、向量化、入库。"""
    # 1) 准备 Chroma 持久化客户端 + collection
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    # 2) 列出所有 .txt 文件（按文件名排序，保证可复现）
    files = sorted(
        f for f in os.listdir(jd_folder)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(jd_folder, f))
    )

    if not files:
        print(f"[WARN] 文件夹中没有 .txt 文件：{jd_folder}")
        print("[DONE] 入库完成：成功 0 条，跳过 0 条，失败 0 条。")
        return

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for fname in files:
        fpath = os.path.join(jd_folder, fname)
        job_id = os.path.splitext(fname)[0]
        print(f"[START] 正在处理：{fname}")

        # 3) 读取原文（SQLite 回填也需要原文，故无论是否已在向量库都先读取）
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                jd_text = f.read()
        except Exception as e:
            print(f"[ERROR] 文件读取失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        if not jd_text or not jd_text.strip():
            print(f"[SKIP] 文件为空：{fname}")
            skip_count += 1
            continue

        # 4) 已在向量库：跳过向量化（省 token），但仍回填 SQLite jobs 表，
        #    画像直接从向量库 metadata 复用，不重新调用 parse_jd
        if job_exists(collection, job_id):
            print(f"[SKIP] job_id 已存在（向量库），回填 SQLite：{job_id}")
            cached_profile = _profile_from_chroma(collection, job_id)
            if cached_profile:
                backfill_jobs_table(job_id, jd_text, cached_profile)
            skip_count += 1
            continue

        # 5) 解析 JD
        try:
            jd_profile = parse_jd(jd_text)
        except Exception as e:
            print(f"[ERROR] 解析失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        if not isinstance(jd_profile, dict) or not jd_profile:
            print(f"[ERROR] 解析失败：{fname}")
            fail_count += 1
            continue

        # 6) 构造 index_text
        index_text = build_index_text(jd_profile)
        if not index_text.strip():
            print(f"[ERROR] index_text 为空：{fname}")
            fail_count += 1
            continue

        # 7) 获取向量
        try:
            vector = get_embedding(index_text)
        except Exception as e:
            print(f"[ERROR] Embedding 失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        # 8) 构造 metadata 并入库
        metadata = build_metadata(job_id, fname, jd_profile)
        try:
            collection.add(
                ids=[job_id],
                embeddings=[vector],
                documents=[index_text],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"[ERROR] 写入 ChromaDB 失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        # 9) 同步写入 SQLite jobs 表（结构化元数据，与向量库并存）
        backfill_jobs_table(job_id, jd_text, jd_profile)

        company = metadata.get("company") or "未知公司"
        title = metadata.get("title") or "未知岗位"
        print(f"[OK] 入库成功（向量库 + SQLite）：{job_id} - {company} - {title}")
        ok_count += 1

    # 9) 汇总
    print(f"[DONE] 入库完成：成功 {ok_count} 条，跳过 {skip_count} 条，失败 {fail_count} 条。")


# ===== CLI =====
def main():
    """命令行入口：python job_indexer.py <jd_folder>"""
    parser = argparse.ArgumentParser(description="Job Indexer — 批量解析 JD 并入库到本地 ChromaDB")
    parser.add_argument("--jd_folder", default=DEFAULT_JOB_PATH, help="存放多份 JD .txt 文件的目录")
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"ChromaDB 持久化目录（默认 {DEFAULT_DB_PATH}）",
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION_NAME,
        help=f"Collection 名称（默认 {DEFAULT_COLLECTION_NAME}）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.jd_folder):
        print(f"[ERROR] 文件夹不存在：{args.jd_folder}")
        sys.exit(1)

    index_jobs(args.jd_folder, db_path=args.db_path, collection_name=args.collection)

    # ===== 读取测试：回读一条记录，验证入库结果 =====
    print("\n[TEST] 读取测试：从库中回读一条记录")
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection)
    total = collection.count()
    print(f"[TEST] 当前 collection 共 {total} 条记录")
    if total == 0:
        print("[TEST] collection 为空，跳过读取测试")
        return

    jd_test_len = 5
    sample = collection.get(limit=jd_test_len, include=["documents", "metadatas"])
    for i in range(jd_test_len):
        sample_id = sample["ids"][i]
        metadata = sample["metadatas"][i]
        document = sample["documents"][i]
        print(f"[TEST] job_id：{sample_id}")
        print(f"[TEST] 公司/岗位：{metadata.get('company')} - {metadata.get('title')}")
        print(f"[TEST] metadata 字段数：{len(metadata)}")
        print(f"[TEST] index_text 预览：\n{document}")


if __name__ == "__main__":
    main()
