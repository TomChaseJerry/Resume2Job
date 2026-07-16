"""
岗位知识库构建模块（Job Indexer）

批量解析 jd_folder 下的 .txt JD 文件，按「SQLite 事实源 + Chroma 索引」分层入库：
    - SQLite jobs 表：原文、完整 jd_profile、index_text 等全部业务数据（事实源）；
    - Chroma：embedding + document(index_text) + 最小投影 metadata（job_id + 展示标量；
      硬约束改 SQLite eligibility 召回前预筛，详见 jobs_store / retriever 模块说明）。

流程（每个文件）：
    读取原文 -> 取画像（SQLite 已有**且 JD 原文哈希未变**则复用，否则 parse_jd，省 token）
        -> upsert SQLite（事实源始终刷新）
        -> 不在向量库 / JD 原文已变 则 embedding + 写入 Chroma（原文变了先删旧向量再重嵌入）
    -> 汇总日志

幂等：JD 原文不变时重跑全跳过；原文改了（同 job_id）会自动重解析 + 重嵌入，避免
「BM25/SQLite 看新 JD、dense 看旧向量」的双通道陈旧。
换 embedding 模型后用 scripts/rebuild_index.py 从 SQLite 重建 Chroma，无需重新解析。
"""

import os
import sys
import argparse
from typing import Optional

import chromadb

# 复用已有的 JD 解析器（infer_job_type/derive 供回填用；批量建库的 parse/embed 已下沉到 ingest.lifecycle）
from resume2job.parsing.jd_parser import derive_constraint_fields, infer_job_type
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


# ===== Chroma metadata 构建（最小投影：job_id 供 where $in 限定 + 少量展示标量）=====
def build_chroma_metadata(job_id: str, jd_profile: dict, source: str = "batch_indexed") -> dict:
    """构建写入 ChromaDB 的最小 metadata。

    架构约定：完整结构化画像存 SQLite（事实源，见 storage/jobs_store.py）。硬约束 pre-filter
    已移到 SQLite eligibility（召回前预筛 allowed_job_ids），Chroma metadata 只保留 job_id
    （供检索时 where job_id $in 限定范围）+ 少量展示/调试标量（city / direction / education_level）。
    Chroma 1.x metadata 值不接受 None，缺失字段直接不写入该 key。
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
    """把一条 JD 的完整业务数据写入 SQLite 事实源（幂等覆盖，含版本 / 质量戳）。

    经 ingest.models.JobRecord 装配，统一盖上 parser/index_text/embedding 版本戳与 quality_score，
    再交 jobs_store.upsert_job。批量建库主流程已改走 ingest.lifecycle.ingest_record；本函数保留为
    「仅写 SQLite 事实源、不碰向量」的便捷入口。
    """
    from resume2job.ingest.models import JobRecord
    from resume2job.ingest.validator import validate_job
    q = validate_job(jd_profile, jd_text)
    record = JobRecord.from_jd_profile(job_id, jd_text, jd_profile, index_text=index_text,
                                       source=source, quality_score=q.quality_score)
    jobs_store.upsert_job(record.to_store_dict())


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
    """遍历 jd_folder 下的 JD 文件，经统一接入层（ingest.lifecycle）逐条入库。

    历史上本函数自跑「读文件→parse→写 SQLite→写 Chroma」整条逻辑；现已收敛为
    LocalFileConnector + lifecycle.ingest_record，与运行时粘贴入库共用**同一条接入路径**，
    从根上消除两条路径写入漂移，并自动盖上版本戳 / 质量分。.txt/.md 以文件名 stem 作 job_id
    （幂等更新），.json 可携带 company/title/source_job_id/canonical_url 等提示。
    """
    from resume2job.ingest.connectors import LocalFileConnector
    from resume2job.ingest.lifecycle import ingest_all
    from resume2job.ingest.models import SOURCE_BATCH

    if not os.path.isdir(jd_folder):
        print(f"[WARN] 文件夹不存在：{jd_folder}")
        print("[DONE] 入库完成：created=0 updated=0 unchanged=0 duplicate=0 invalid=0 failed=0")
        return

    # Chroma collection：按 CLI 指定的 db_path / collection_name 打开（默认即统一向量库）
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    connector = LocalFileConnector(jd_folder, source=SOURCE_BATCH)
    summary = ingest_all(connector, collection=collection, verbose=True)
    c = summary["counts"]
    print(f"[DONE] 入库完成：created={c['created']} updated={c['updated']} "
          f"unchanged={c['unchanged']} duplicate={c['duplicate']} "
          f"invalid={c['invalid']} failed={c['failed']}")


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

    sample = collection.get(limit=5, include=["documents", "metadatas"])
    for i in range(len(sample.get("ids") or [])):   # 按实际回读条数遍历，库内不足 5 条也不越界
        sample_id = sample["ids"][i]
        metadata = sample["metadatas"][i]
        document = sample["documents"][i]
        print(f"[TEST] job_id：{sample_id}")
        print(f"[TEST] 公司/岗位：{metadata.get('company')} - {metadata.get('title')}")
        print(f"[TEST] metadata 字段数：{len(metadata)}")
        print(f"[TEST] index_text 预览：\n{document}")


if __name__ == "__main__":
    main()
