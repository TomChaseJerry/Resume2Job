# -*- coding: utf-8 -*-
"""
scripts/rebuild_index.py

向量索引迁移 / 重建工具（「SQLite 事实源 + Chroma 索引」分层的配套脚本）。

两种模式：
    --migrate（默认）一次性迁移旧库：
        1. 把旧 Chroma metadata 里的 jd_profile_json 等业务数据搬回 SQLite jobs 表
           （事实源补齐，零 token，不重新解析）；
        2. 把 Chroma metadata 收敛为最小投影（job_id / company / title /
           city / direction / education_level / source），不重新生成向量。
    --rebuild 全量重建向量库：
        从 SQLite 读 index_text 重新 embedding 并重写 Chroma。
        换 embedding 模型（RESUME2JOB_EMBEDDING_MODEL）后用这个，零解析成本。

用法：
    python scripts/rebuild_index.py             # 迁移旧库（幂等，可重复执行）
    python scripts/rebuild_index.py --rebuild   # 换 embedding 模型后全量重建
"""

import os
import sys
import json
import argparse

# 脚本位于 scripts/ 下，手动把项目根目录加进 sys.path 以便导入 resume2job 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import chromadb

from resume2job.storage.paths import CHROMA_DIR, COLLECTION_NAME
from resume2job.storage import jobs_store
from resume2job.storage.jobs_store import compute_jd_hash

# Windows 控制台中文输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _open_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client, client.get_or_create_collection(name=COLLECTION_NAME)


def _legacy_profile(meta: dict) -> dict:
    """从旧版 Chroma metadata 的 jd_profile_json 还原画像；无则空 dict。"""
    raw = (meta or {}).get("jd_profile_json")
    if isinstance(raw, str) and raw.strip():
        try:
            profile = json.loads(raw)
            if isinstance(profile, dict):
                return profile
        except json.JSONDecodeError:
            pass
    return {}


def _read_jd_text(job_id: str) -> str:
    """补齐 jd_text：SQLite 已有则用已有，否则尝试 JDs/<job_id>.txt（批量入库约定）。"""
    row = jobs_store.get_job(job_id)
    if row and row.get("jd_text"):
        return row["jd_text"]
    path = os.path.join(_PROJECT_ROOT, "JDs", f"{job_id}.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def migrate() -> int:
    """旧 Chroma metadata → SQLite 事实源；Chroma metadata 收敛为最小投影。"""
    from resume2job.retrieval.indexer import build_chroma_metadata, build_index_text

    _, collection = _open_collection()
    # embeddings 一并读出：metadata 收敛需要 delete+add 重写（Chroma 的
    # update(metadatas=...) 只做 key 合并，不会删除旧字段），向量原样保留不重算
    res = collection.get(include=["metadatas", "documents", "embeddings"])
    ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    docs = res.get("documents") or []
    embs = res.get("embeddings")
    embs = list(embs) if embs is not None else []
    if not ids:
        print("[migrate] 向量库为空，无需迁移。")
        return 0

    moved, normalized = 0, 0
    new_metas = []
    for i, job_id in enumerate(ids):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        document = docs[i] if i < len(docs) else ""

        # 1) 事实源补齐：SQLite 缺画像时，从旧 metadata 搬回
        row = jobs_store.get_job(job_id)
        profile = (row or {}).get("jd_profile") or _legacy_profile(meta)
        if profile and not ((row or {}).get("jd_profile")):
            jd_text = _read_jd_text(job_id)
            location = profile.get("location") if isinstance(profile.get("location"), dict) else {}
            skills = list(dict.fromkeys([*(profile.get("hard_skills") or []),
                                         *(profile.get("tools_or_frameworks") or [])]))
            jobs_store.upsert_job({
                "job_id": job_id,
                "company": profile.get("company") or "unknown_company",
                "title": profile.get("title") or "unknown_title",
                "city": location.get("city") or "",
                "direction": profile.get("direction") or "",
                "education_level": profile.get("education_level") or "",
                "jd_text": jd_text,
                "jd_hash": compute_jd_hash(jd_text),
                "jd_profile": profile,
                "index_text": document or build_index_text(profile),
                "requirements": profile.get("responsibilities") or [],
                "skills": skills,
                "source": meta.get("source") or "batch_indexed",
            })
            moved += 1
            print(f"[migrate] 画像迁回 SQLite：{job_id}")
        elif row and not row.get("index_text") and document:
            # 行已存在但缺 index_text（旧 jobs 表）：补 index_text 列
            merged = dict(row)
            merged["jd_profile"] = row["jd_profile"]
            merged["index_text"] = document
            merged["requirements"] = json.loads(row.get("requirements") or "[]")
            merged["skills"] = json.loads(row.get("skills") or "[]")
            jobs_store.upsert_job(merged)
            moved += 1
            print(f"[migrate] 补齐 index_text：{job_id}")

        # 2) metadata 最小化（含 jd_profile_json 的才需要重写）
        if "jd_profile_json" in meta or "hard_skills" in meta:
            normalized += 1
        new_metas.append(build_chroma_metadata(job_id, profile,
                                               source=meta.get("source") or "batch_indexed"))

    if normalized and len(embs) == len(ids):
        collection.delete(ids=ids)
        collection.add(
            ids=ids,
            embeddings=[list(e) for e in embs],
            documents=docs,
            metadatas=new_metas,
        )
        print(f"[migrate] Chroma metadata 已收敛为最小投影（delete+add 重写）：{normalized}/{len(ids)} 条")
    elif normalized:
        print("[migrate] 警告：未能读出 embeddings，跳过 metadata 收敛；可用 --rebuild 全量重建。")
    else:
        print("[migrate] Chroma metadata 已是最小投影，无需重写。")

    print(f"[migrate] 完成：迁移/补齐 {moved} 条，SQLite jobs 共 {jobs_store.count()} 条。")
    return 0


def rebuild() -> int:
    """从 SQLite 事实源全量重建 Chroma（换 embedding 模型后使用）。"""
    from resume2job.core.llm import get_embedding
    from resume2job.retrieval.indexer import build_chroma_metadata

    rows = jobs_store.all_jobs_for_index()
    if not rows:
        print("[rebuild] SQLite 无可用 index_text，请先运行 --migrate 或重新入库。")
        return 1

    client, _ = _open_collection()
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[rebuild] 已删除旧 collection：{COLLECTION_NAME}")
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ok, fail = 0, 0
    for r in rows:
        job_id = r["job_id"]
        full = jobs_store.get_job(job_id) or {}
        try:
            vector = get_embedding(r["index_text"])
            collection.add(
                ids=[job_id],
                embeddings=[vector],
                documents=[r["index_text"]],
                metadatas=[build_chroma_metadata(job_id, full.get("jd_profile") or {},
                                                 source=full.get("source") or "batch_indexed")],
            )
            ok += 1
            print(f"[rebuild] OK：{job_id}")
        except Exception as e:
            fail += 1
            print(f"[rebuild] FAIL：{job_id} — {e}")

    print(f"[rebuild] 完成：成功 {ok} 条，失败 {fail} 条。")
    return 0 if fail == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="向量索引迁移 / 重建（SQLite 事实源）")
    parser.add_argument("--rebuild", action="store_true",
                        help="从 SQLite 全量重新 embedding 并重写 Chroma（默认仅做迁移）")
    args = parser.parse_args()
    return rebuild() if args.rebuild else migrate()


if __name__ == "__main__":
    sys.exit(main())
