"""
岗位知识库构建模块（Job Indexer）

批量解析 jd_folder 下的 .txt JD 文件，按「SQLite 事实源 + Chroma 索引」分层入库：
    - SQLite jobs 表：原文、完整 jd_profile、index_text 等全部业务数据（事实源）；
    - Chroma：embedding + document(index_text) + 最小可过滤 metadata（索引投影）。

流程（每个文件）：
    读取原文 -> 取画像（SQLite 已有则复用，否则 parse_jd，省 token）
        -> upsert SQLite（事实源始终刷新）
        -> 未在向量库则 embedding + 写入 Chroma
    -> 汇总日志

换 embedding 模型后用 scripts/rebuild_index.py 从 SQLite 重建 Chroma，无需重新解析。
"""

import os
import sys
import argparse
from typing import Optional

import chromadb

# 复用已有的 JD 解析器
from resume2job.parsing.jd_parser import parse_jd, derive_constraint_fields, infer_job_type
# 统一 Embedding（core 层单一事实来源）
from resume2job.core.llm import get_embedding
# jobs 表统一读写（SQLite 事实源，与运行时 jd_ingest_node 共用）
from resume2job.storage import jobs_store
from resume2job.storage.jobs_store import compute_jd_hash


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


# ===== Chroma metadata 构建（最小投影：只放检索 pre-filter 需要的标量）=====
def build_chroma_metadata(job_id: str, jd_profile: dict, source: str = "batch_indexed") -> dict:
    """构建写入 ChromaDB 的最小 metadata。

    架构约定：完整结构化画像存 SQLite（事实源，见 storage/jobs_store.py），
    Chroma metadata 只保留向量检索 pre-filter 必需的标量（city / direction /
    education_level）及少量展示/调试字段。Chroma 1.x metadata 值不接受 None，
    缺失字段直接不写入该 key。
    """
    if not isinstance(jd_profile, dict):
        jd_profile = {}
    location = jd_profile.get("location") if isinstance(jd_profile.get("location"), dict) else {}

    meta = {
        "job_id": job_id,
        "company": jd_profile.get("company"),
        "title": jd_profile.get("title"),
        "city": location.get("city"),
        "direction": jd_profile.get("direction"),
        "education_level": jd_profile.get("education_level"),
        "source": source,
    }
    return {k: v.strip() for k, v in meta.items()
            if isinstance(v, str) and v.strip()}


def lookup_ingested_jd_profile(jd_text: str) -> Optional[dict]:
    """按 JD 原文哈希在已入库岗位中查找并复用其结构化画像（零 token、跨链路一致性）。

    用途：用户直接粘贴的 JD 若与知识库中某条已入库 JD 文本完全一致，则复用入库时
    解析好的同一份 jd_profile，使「单 JD 评估」与「岗位推荐」两条链路看到**完全相同**
    的结构化画像，避免因重复解析产生字段差异导致同一 JD 评分不一致。

    实现：纯 SQLite 查询（jd_hash → jd_profile_json）。
    未命中或失败返回 None，由调用方回退到实时 parse_jd。
    """
    if not isinstance(jd_text, str) or not jd_text.strip():
        return None
    hit = jobs_store.get_profile_by_hash(compute_jd_hash(jd_text))
    return hit["jd_profile"] if hit else None


def upsert_job_record(job_id: str, jd_text: str, jd_profile: dict,
                      index_text: str, source: str = "batch_indexed") -> None:
    """把一条 JD 的完整业务数据写入 SQLite 事实源（幂等覆盖）。"""
    location = jd_profile.get("location") if isinstance(jd_profile.get("location"), dict) else {}
    skills = list(dict.fromkeys([*(jd_profile.get("hard_skills") or []),
                                 *(jd_profile.get("tools_or_frameworks") or [])]))
    jobs_store.upsert_job({
        "job_id": job_id,
        "company": jd_profile.get("company") or "unknown_company",
        "title": jd_profile.get("title") or "unknown_title",
        "city": location.get("city") or "",
        "direction": jd_profile.get("direction") or "",
        "education_level": jd_profile.get("education_level") or "",
        "jd_text": jd_text,
        "jd_hash": compute_jd_hash(jd_text),
        "jd_profile": jd_profile,
        "index_text": index_text,
        "requirements": jd_profile.get("responsibilities") or [],
        "skills": skills,
        "source": source,
        # 硬约束预过滤列（cities_json/job_types_json/min_degree_rank/city_status/education_status）
        **derive_constraint_fields(jd_profile),
    })


# ===== 硬约束预过滤列回填（旧库 ALTER 补列后跑一次，从 jd_profile 重算，纯 SQLite）=====
def backfill_constraint_fields() -> int:
    """为已入库岗位回填硬约束预过滤列（cities_json/job_types_json/min_degree_rank/city_status/
    education_status）。用 jd_text 在**内存**里重判 job_type（修旧「全职」等非三桶值），只写约束列、
    **不回写 jd_profile**（派生分类不持久化回事实源，避免污染 LLM 原始抽取）。返回条数。"""
    rows = jobs_store.all_jobs_min()
    n = 0
    for r in rows:
        prof = dict(r["jd_profile"])                       # 副本，不动原画像
        infer_job_type(prof, r.get("jd_text") or "")       # 内存重判（全职先查校招届再归社招，不持久化）
        jobs_store.update_constraint_fields(r["job_id"], **derive_constraint_fields(prof))
        n += 1
    print(f"[indexer] 硬约束预过滤列回填完成：{n} 条")
    return n


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
    """遍历 jd_folder 下 .txt 文件，逐条入库（SQLite 事实源 + Chroma 索引）。"""
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

        # 3) 读取原文
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

        # 4) 取画像：SQLite 已有则复用（零 token），否则 parse_jd
        existing = jobs_store.get_job(job_id)
        jd_profile = existing["jd_profile"] if existing and existing.get("jd_profile") else None
        if jd_profile:
            print(f"[INFO] 画像已在 SQLite，复用：{job_id}")
        else:
            try:
                jd_profile = parse_jd(jd_text)
            except Exception as e:
                print(f"[ERROR] 解析失败：{fname}，原因：{e}")
                fail_count += 1
                continue
            # 解析失败可区分（空文本 / 模型异常 / 输出非 JSON）：打印具体原因，不静默
            if not isinstance(jd_profile, dict) or not jd_profile or jd_profile.get("error"):
                reason = jd_profile.get("error") if isinstance(jd_profile, dict) and jd_profile.get("error") \
                    else "结果为空"
                print(f"[ERROR] 解析失败：{fname}（{reason}）")
                fail_count += 1
                continue

        # 5) 构造 index_text 并刷新 SQLite 事实源（幂等覆盖）
        index_text = build_index_text(jd_profile)
        if not index_text.strip():
            print(f"[ERROR] index_text 为空：{fname}")
            fail_count += 1
            continue
        try:
            upsert_job_record(job_id, jd_text, jd_profile, index_text)
        except Exception as e:
            print(f"[ERROR] 写入 SQLite 失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        # 6) 已在向量库：跳过向量化（省 token）
        if job_exists(collection, job_id):
            print(f"[SKIP] job_id 已存在（向量库）：{job_id}")
            skip_count += 1
            continue

        # 7) 获取向量并写入 Chroma（最小可过滤 metadata）
        try:
            vector = get_embedding(index_text)
        except Exception as e:
            print(f"[ERROR] Embedding 失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        metadata = build_chroma_metadata(job_id, jd_profile)
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

        company = metadata.get("company") or "未知公司"
        title = metadata.get("title") or "未知岗位"
        print(f"[OK] 入库成功（SQLite + 向量库）：{job_id} - {company} - {title}")
        ok_count += 1

    # 8) 汇总
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
