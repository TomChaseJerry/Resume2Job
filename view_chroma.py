"""
view_chroma.py

查看统一向量库（data/chroma_db，collection=jobs）的内容，用于核对入库结果。
只读，不修改任何数据。

用法：
    python view_chroma.py                  # 概览：总条数 + 每条 job_id / 公司 / 岗位 / 城市
    python view_chroma.py --limit 5        # 只看前 5 条
    python view_chroma.py --full           # 展开每条的 index_text 与全部 metadata
    python view_chroma.py --id jd_test_1  # 查看指定 job_id 的完整内容
"""

import os
import sys
import json
import argparse

import chromadb

from storage.paths import CHROMA_DIR, COLLECTION_NAME

# Windows 控制台中文输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _open_collection():
    """打开统一向量库的 jobs collection；目录不存在时返回 None。"""
    if not os.path.isdir(CHROMA_DIR):
        print(f"[ERROR] 向量库目录不存在：{CHROMA_DIR}")
        print("       先运行 python ingest_jds.py 入库。")
        return None
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _print_one(job_id: str, document: str, metadata: dict, full: bool) -> None:
    """打印单条记录：概览模式只打印关键字段，full 模式展开全部。"""
    meta = metadata or {}
    company = meta.get("company") or "未知公司"
    title = meta.get("title") or "未知岗位"
    city = meta.get("city") or "-"
    print(f"\n● job_id={job_id}")
    print(f"  公司/岗位：{company} - {title}    城市：{city}")
    print(f"  来源文件：{meta.get('source_file') or '-'}")

    if not full:
        return

    # full 模式：展开 index_text 与全部 metadata（jd_profile_json 单独美化）
    print("  --- index_text ---")
    print("  " + (document or "").replace("\n", "\n  "))
    print("  --- metadata ---")
    for k, v in meta.items():
        if k == "jd_profile_json":
            continue
        print(f"    {k}: {v}")
    raw = meta.get("jd_profile_json")
    if raw:
        try:
            profile = json.loads(raw)
            print("  --- jd_profile（解析后）---")
            print("  " + json.dumps(profile, ensure_ascii=False, indent=2).replace("\n", "\n  "))
        except Exception:
            print(f"    jd_profile_json: {raw}")


def main() -> int:
    parser = argparse.ArgumentParser(description="查看统一向量库（data/chroma_db）内容（只读）")
    parser.add_argument("--limit", type=int, default=0, help="最多展示多少条（0 表示全部）")
    parser.add_argument("--full", action="store_true", help="展开每条的 index_text 与全部 metadata")
    parser.add_argument("--id", default=None, help="只查看指定 job_id")
    args = parser.parse_args()

    collection = _open_collection()
    if collection is None:
        return 1

    total = collection.count()
    print("=" * 60)
    print(f"向量库：{CHROMA_DIR}")
    print(f"collection={COLLECTION_NAME}    总记录数：{total}")
    print("=" * 60)

    if total == 0:
        print("（向量库为空，先运行 python ingest_jds.py 入库）")
        return 0

    # ---- 单条查询 ----
    if args.id:
        res = collection.get(ids=[args.id], include=["documents", "metadatas"])
        ids = res.get("ids") or []
        if not ids:
            print(f"[未找到] job_id={args.id}")
            return 1
        _print_one(ids[0], (res.get("documents") or [""])[0], (res.get("metadatas") or [{}])[0], full=True)
        return 0

    # ---- 列表查询 ----
    get_kwargs = {"include": ["documents", "metadatas"]}
    if args.limit and args.limit > 0:
        get_kwargs["limit"] = args.limit
    res = collection.get(**get_kwargs)

    ids = res.get("ids") or []
    documents = res.get("documents") or []
    metadatas = res.get("metadatas") or []
    print(f"\n本次展示 {len(ids)} / {total} 条：")
    for i, job_id in enumerate(ids):
        doc = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        _print_one(job_id, doc, meta, full=args.full)

    return 0


if __name__ == "__main__":
    sys.exit(main())
