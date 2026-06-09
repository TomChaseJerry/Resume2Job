"""
view_jds.py

汇总 JDs 文件夹下所有岗位的关键信息（只读，不调用 LLM）。

数据来源不是重新解析 .txt，而是直接复用入库时已解析好的结构化画像：
批量入库（ingest_jds.py / job_indexer.py）会把每个 JD 的完整 jd_profile
以 jd_profile_json 形式存进向量库 metadata，且 job_id = 文件名去扩展名。
本脚本据此把每个 JD 文件映射回其已解析画像，零 token 汇总关键字段。

用法：
    python view_jds.py                       # 概览：每个 JD 的公司/岗位/方向/硬技能等关键字段
    python view_jds.py --full                # 额外展开每个 JD 的完整 jd_profile（JSON）
    python view_jds.py --jd_folder 路径      # 指定 JD 文件夹（默认项目下 JDs/）

未入库（向量库里查不到对应 job_id）的 JD 会被标注，提示先运行 python ingest_jds.py。
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

DEFAULT_JD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JDs")


def _open_collection():
    """打开统一向量库的 jobs collection；目录不存在时返回 None。"""
    if not os.path.isdir(CHROMA_DIR):
        print(f"[ERROR] 向量库目录不存在：{CHROMA_DIR}")
        print("       先运行 python ingest_jds.py 入库。")
        return None
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _load_profiles_by_job_id(collection) -> dict:
    """一次性读出全部记录，构建 {job_id: jd_profile(dict)} 映射。

    jd_profile 取自 metadata 的 jd_profile_json（入库时已解析），解析失败的条目跳过。
    """
    res = collection.get(include=["metadatas"])
    ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    mapping = {}
    for i, job_id in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        raw = (meta or {}).get("jd_profile_json")
        if not raw:
            continue
        try:
            profile = json.loads(raw)
            if isinstance(profile, dict):
                mapping[job_id] = profile
        except (json.JSONDecodeError, TypeError):
            continue
    return mapping


def _join(value, limit: int = 0) -> str:
    """把列表字段拼成可读字符串；limit>0 时截断并标注剩余数量。"""
    if not isinstance(value, list):
        return str(value) if value else "-"
    items = [str(x).strip() for x in value if str(x).strip()]
    if not items:
        return "-"
    if limit and len(items) > limit:
        return "、".join(items[:limit]) + f" …（共 {len(items)} 项）"
    return "、".join(items)


def _get_city(profile: dict) -> str:
    """城市可能在顶层 city，也可能在 location.city。"""
    city = profile.get("city")
    if city:
        return str(city)
    loc = profile.get("location") or {}
    if isinstance(loc, dict) and loc.get("city"):
        return str(loc["city"])
    return "-"


def _print_summary(job_id: str, source_file: str, profile: dict, full: bool) -> None:
    """打印单个 JD 的关键字段；full 模式额外展开完整 jd_profile。"""
    company = profile.get("company") or "未知公司"
    title = profile.get("title") or "未知岗位"

    print(f"\n● {source_file}（job_id={job_id}）")
    print(f"  公司/岗位：{company} - {title}    城市：{_get_city(profile)}")
    print(f"  方向：{profile.get('direction') or '-'}    业务场景：{profile.get('business_area') or '-'}")
    print(f"  学历要求：{profile.get('education_requirement') or '-'}    "
          f"经验要求：{profile.get('experience_requirement') or '-'}")
    print(f"  硬技能：{_join(profile.get('hard_skills'), limit=0)}")
    print(f"  工具框架：{_join(profile.get('tools_or_frameworks'), limit=0)}")
    print(f"  加分项：{_join(profile.get('preferred_skills'), limit=0)}")
    print(f"  领域关键词：{_join(profile.get('domain_keywords'), limit=0)}")

    resp = profile.get("responsibilities")
    if isinstance(resp, list) and resp:
        print(f"  岗位职责（{len(resp)} 条）：")
        for r in resp:
            print(f"    - {str(r).strip()}")

    if full:
        print("  --- 完整 jd_profile ---")
        print("  " + json.dumps(profile, ensure_ascii=False, indent=2).replace("\n", "\n  "))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="汇总 JDs 文件夹下所有岗位的关键信息（只读，不调用 LLM）"
    )
    parser.add_argument("--jd_folder", default=DEFAULT_JD_FOLDER,
                        help=f"JD 文件夹路径（默认 {DEFAULT_JD_FOLDER}）")
    parser.add_argument("--full", action="store_true",
                        help="额外展开每个 JD 的完整 jd_profile（JSON）")
    args = parser.parse_args()

    jd_folder = args.jd_folder
    if not os.path.isdir(jd_folder):
        print(f"[ERROR] JD 文件夹不存在：{jd_folder}")
        return 1

    collection = _open_collection()
    if collection is None:
        return 1

    profiles = _load_profiles_by_job_id(collection)

    # 按文件名排序，保证输出稳定
    txt_files = sorted(f for f in os.listdir(jd_folder) if f.lower().endswith(".txt"))

    print("=" * 60)
    print(f"JD 文件夹：{jd_folder}")
    print(f"向量库：{CHROMA_DIR}（collection={COLLECTION_NAME}）")
    print(f"JD 文件数：{len(txt_files)}    向量库已解析画像数：{len(profiles)}")
    print("=" * 60)

    if not txt_files:
        print("（该文件夹下没有 .txt 文件）")
        return 0

    matched = 0
    missing = []
    for fname in txt_files:
        job_id = os.path.splitext(fname)[0]
        profile = profiles.get(job_id)
        if profile is None:
            missing.append(fname)
            continue
        matched += 1
        _print_summary(job_id, fname, profile, full=args.full)

    print(f"\n——汇总：{matched}/{len(txt_files)} 个 JD 已入库并展示——")
    if missing:
        print(f"未入库（向量库中无对应 job_id，请先 python ingest_jds.py）：")
        for f in missing:
            print(f"  - {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
