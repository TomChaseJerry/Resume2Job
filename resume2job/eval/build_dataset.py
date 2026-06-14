# -*- coding: utf-8 -*-
"""检索评测集构造（synthetic eval set，支持单 query 多相关岗位）。

业界常用做法：从知识库文档**反向生成**检索 query（query 与生成它的源 JD 天然相关），
快速得到 query → relevant_doc 标注，规模可随 JD 库扩展自动重建。

单 query 多 relevant_job_ids（本文件核心）：
    真实场景里「搜大模型算法实习」不止源 JD 一个正确答案——腾讯/快手/阿里的同方向岗
    都该算相关。仅用源 JD 当唯一 gold 会低估召回质量（把本该召回的同类岗判为「召错」）。
    因此用 TREC 风格的 pooling 自动扩充相关集：
        对每个源 JD —— 检索召回相似岗位候选池 → LLM 批量判定哪些岗位与源 JD
        「属于求职者会同时投递的同类方向」→ 该源 JD 的 3 条 query 共享这个相关集。

流程：
    1. 从 SQLite 事实源遍历全部岗位（含结构化 jd_profile）；
    2. 每条 JD 生成 3 风格 query（keyword / natural / colloquial）；
    3. 检索 + LLM pooling 判定源 JD 的相关岗位集（含源自身）；
    4. 写入 eval/data/retrieval_dataset.jsonl，每行：
         {"query_id", "job_id"(源), "style", "query", "relevant_job_ids": [...]}

人工抽查：直接编辑该 jsonl —— 改写不自然的 query，或在 relevant_job_ids 增删岗位。
retrieval_eval 按文件现状评测，relevant 越准，指标越能反映真实召回质量。
"""

import os
import json
import argparse

from resume2job.core.config import CHAT_MODEL
from resume2job.core.llm import call_llm, safe_json_parse
from resume2job.storage import jobs_store

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_PATH = os.path.join(DATA_DIR, "retrieval_dataset.jsonl")

_STYLES = ("keyword", "natural", "colloquial")
_POOL_TOP_K = 8  # 相关性判定的候选池大小

# ---------------------------------------------------------------------------
# 1. query 生成（每个源 JD 三风格）
# ---------------------------------------------------------------------------
_QUERY_SYSTEM = (
    "你是一个检索评测数据构造器。给定一条实习岗位 JD 的结构化摘要，"
    "生成 3 条「正在找实习的学生可能输入的检索请求」，要求该 JD 是这些请求的理想答案。\n"
    "三条风格分别为：\n"
    "1. keyword：6~10 个空格分隔的技能/方向关键词；\n"
    "2. natural：一句完整的自然语言求职意向；\n"
    "3. colloquial：口语化、可含模糊措辞（如「会点」「想搞」）。\n"
    "禁止直接抄写公司名或岗位名（检索者通常不知道具体公司）。\n"
    "只输出 JSON：{\"keyword\": \"...\", \"natural\": \"...\", \"colloquial\": \"...\"}"
)


def _job_brief(profile: dict) -> dict:
    """岗位结构化摘要（喂给 LLM 做 query 生成 / 相关性判定，控制长度）。"""
    p = profile or {}
    loc = p.get("location") or {}
    return {
        "direction": p.get("direction"),
        "business_area": p.get("business_area"),
        "hard_skills": (p.get("hard_skills") or [])[:10],
        "domain_keywords": (p.get("domain_keywords") or [])[:8],
        "cities": loc.get("cities") or ([loc.get("city")] if loc.get("city") else []),
    }


def _generate_queries(job_id: str, profile: dict, index_text: str) -> dict:
    """对单条 JD 生成 3 风格 query。失败返回空 dict（该 JD 跳过）。"""
    brief = json.dumps(_job_brief(profile), ensure_ascii=False)
    try:
        raw = call_llm(
            _QUERY_SYSTEM,
            f"## JD 摘要\n{brief}\n\n## JD 检索文本\n{index_text[:1200]}",
            model=CHAT_MODEL, temperature=0.5,  # 评测 query 需多样性
        )
        parsed = safe_json_parse(raw) or {}
    except Exception as e:
        print(f"[WARN] {job_id} query 生成失败：{e}")
        return {}
    return {s: str(parsed.get(s) or "").strip() for s in _STYLES if str(parsed.get(s) or "").strip()}


# ---------------------------------------------------------------------------
# 2. 相关岗位 pooling（检索候选池 + LLM 批量判定）
# ---------------------------------------------------------------------------
_RELATED_SYSTEM = (
    "你是检索评测标注员。给定一个目标实习岗位与若干候选岗位，判断哪些候选岗位与"
    "目标岗位属于「求职者会同时投递的同类方向」——即方向相近、核心技能可迁移，"
    "对同一条求职请求都算合理答案。判定从严：方向明显不同（如大模型算法 vs 后端开发）"
    "不算相关。\n"
    "只输出 JSON：{\"related_job_ids\": [...]}，job_id 只能取自给定候选，无相关则空列表。"
)


def _find_related(source_id: str, source_profile: dict, index_text: str) -> list:
    """检索候选池 + LLM 判定，返回与源 JD 相关的其他 job_id 列表（不含源自身）。"""
    from resume2job.retrieval.retriever import search_jobs

    # 用源 JD 的检索文本召回相似岗位作为候选池（关掉 rerank 省调用，覆盖率优先）
    try:
        hits = search_jobs(index_text[:600], top_k=_POOL_TOP_K, mode="hybrid", use_rerank=False)
    except Exception as e:
        print(f"[WARN] {source_id} 候选池检索失败：{e}")
        return []

    pool = []
    for h in hits:
        jid = h.get("job_id")
        if not jid or jid == source_id:
            continue
        pool.append({"job_id": jid, "company": h.get("company"),
                     "title": h.get("title"), **_job_brief(h.get("jd_profile") or {})})
    if not pool:
        return []

    user = (
        f"## 目标岗位\n{json.dumps({'title': source_profile.get('title'), **_job_brief(source_profile)}, ensure_ascii=False)}\n\n"
        f"## 候选岗位（共 {len(pool)} 个）\n{json.dumps(pool, ensure_ascii=False)}"
    )
    try:
        parsed = safe_json_parse(call_llm(_RELATED_SYSTEM, user, model=CHAT_MODEL, temperature=0.0)) or {}
    except Exception as e:
        print(f"[WARN] {source_id} 相关性判定失败：{e}")
        return []

    valid = {p["job_id"] for p in pool}
    return [jid for jid in (parsed.get("related_job_ids") or []) if jid in valid]


# ---------------------------------------------------------------------------
# 3. 构造主流程
# ---------------------------------------------------------------------------
def build_dataset(output_path: str = DATASET_PATH, with_related: bool = True) -> int:
    """遍历岗位库构造评测集，返回写入的样本条数。

    with_related=True 时为每个源 JD 做相关岗位 pooling（单 query 多 relevant_job_ids）；
    False 则只用源 JD 自身作为唯一 gold（更快，召回指标偏保守）。
    """
    rows = jobs_store.all_jobs_for_index()
    if not rows:
        print("[ERROR] 岗位库为空，请先运行 scripts/ingest_jds.py 建库。")
        return 0

    profiles = {r["job_id"]: (jobs_store.get_job(r["job_id"]) or {}).get("jd_profile") or {}
                for r in rows}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            job_id, index_text = r["job_id"], r["index_text"]
            profile = profiles.get(job_id) or {}
            print(f"[build_dataset] 处理：{job_id}")

            queries = _generate_queries(job_id, profile, index_text)
            if not queries:
                continue

            relevant = [job_id]
            if with_related:
                related = _find_related(job_id, profile, index_text)
                relevant = sorted(set([job_id] + related))
                if related:
                    print(f"[build_dataset]   相关岗位：{related}")

            for style, query in queries.items():
                f.write(json.dumps({
                    "query_id": f"{job_id}__{style}",
                    "job_id": job_id,
                    "style": style,
                    "query": query,
                    "relevant_job_ids": relevant,
                }, ensure_ascii=False) + "\n")
                count += 1

    print(f"[build_dataset] 完成：{count} 条评测样本 → {output_path}")
    print("[build_dataset] 建议人工抽查：改写不自然的 query，核对 relevant_job_ids 的增删。")
    return count


def load_dataset(path: str = DATASET_PATH) -> list:
    """读取评测集 jsonl，返回样本 dict 列表。"""
    if not os.path.isfile(path):
        return []
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sample.get("query") and sample.get("relevant_job_ids"):
                samples.append(sample)
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构造检索评测集（LLM 反向生成 query + 相关岗位 pooling）")
    parser.add_argument("--output", default=DATASET_PATH, help="输出 jsonl 路径")
    parser.add_argument("--no-related", action="store_true",
                        help="不做相关岗位 pooling，仅用源 JD 作为唯一 gold（更快）")
    args = parser.parse_args()
    build_dataset(args.output, with_related=not args.no_related)
