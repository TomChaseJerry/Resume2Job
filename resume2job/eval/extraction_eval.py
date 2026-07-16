# -*- coding: utf-8 -*-
"""eval/extraction_eval.py —— 抽取层评测（简历 / JD 结构化抽取质量）。

为什么单独评抽取：JD 的 required_skills / 城市 / 学历 / 岗位类型若抽错，后面的技能匹配、检索 query、
排序特征、技能缺口、推荐报告会**全错**——不能只看最终推荐结果，要把准确率下沉到抽取这一层。

两种口径（按是否有 gold）：
    1. **无 gold（始终可跑，0 LLM）**：对已入库 jd_profile 统计「字段完整度」（各字段非空率）+ 质量分分布
       （复用 ingest.validator）。reparse=True 时重跑 parse_jd 测「JSON 合法率」（需 LLM）。
    2. **有 gold（可选）**：给定 {job_id → 期望 hard_skills/education_level/job_type/cities} 金标，
       算技能集合 Precision/Recall/F1 + 学历/类型/城市 Exact Match。

gold 文件（jsonl，每行一条）示例：
    {"job_id": "jd_test_1", "hard_skills": ["强化学习","python"], "education_level": "硕士",
     "job_type": "社招", "cities": ["北京"]}
"""

import os
import json
import argparse
from datetime import datetime

from resume2job.storage import jobs_store
from resume2job.parsing.jd_parser import job_cities, normalize_job_type
from resume2job.ingest.validator import validate_job

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# 评「完整度」的字段：取值器返回该字段是否非空
_JD_FIELDS = {
    "company": lambda p: bool(p.get("company") and "unknown" not in str(p.get("company")).lower()),
    "title": lambda p: bool(p.get("title")),
    "job_type": lambda p: bool(p.get("job_type")),
    "direction": lambda p: bool(p.get("direction")),
    "education_level": lambda p: bool(p.get("education_level")),
    "responsibilities": lambda p: bool(p.get("responsibilities")),
    "hard_skills": lambda p: bool(p.get("hard_skills")),
    "cities": lambda p: bool(job_cities(p)),
}


def _norm_set(items) -> set:
    return {str(x).strip().lower() for x in (items or []) if str(x).strip()}


def skill_prf(pred, gold) -> dict:
    """技能集合的 Precision / Recall / F1（大小写 / 空白归一后比对）。"""
    p, g = _norm_set(pred), _norm_set(gold)
    if not g:
        return {"precision": None, "recall": None, "f1": None}
    tp = len(p & g)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def field_completeness(profiles: list) -> dict:
    """各字段非空率（抽取完整度）。"""
    n = max(1, len(profiles))
    return {f: round(sum(1 for p in profiles if fn(p)) / n, 4) for f, fn in _JD_FIELDS.items()}


def _load_gold(path: str) -> dict:
    gold = {}
    if not path or not os.path.isfile(path):
        return gold
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("job_id"):
                gold[r["job_id"]] = r
    return gold


def _gold_metrics(profiles_by_id: dict, gold: dict) -> dict:
    """对有 gold 的岗位算技能 P/R/F1（宏平均）+ 学历/类型/城市 Exact Match。"""
    sp, sr, sf, edu_em, jt_em, city_em, n = [], [], [], [], [], [], 0
    for jid, g in gold.items():
        prof = profiles_by_id.get(jid)
        if not prof:
            continue
        n += 1
        prf = skill_prf(prof.get("hard_skills"), g.get("hard_skills"))
        if prf["f1"] is not None:
            sp.append(prf["precision"]); sr.append(prf["recall"]); sf.append(prf["f1"])
        if g.get("education_level") is not None:
            edu_em.append(1.0 if (prof.get("education_level") == g["education_level"]) else 0.0)
        if g.get("job_type") is not None:
            jt_em.append(1.0 if (normalize_job_type(prof.get("job_type")) == normalize_job_type(g["job_type"])) else 0.0)
        if g.get("cities"):
            city_em.append(1.0 if (_norm_set(job_cities(prof)) == _norm_set(g["cities"])) else 0.0)

    def _avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    return {
        "n_gold_matched": n,
        "skill_precision": _avg(sp), "skill_recall": _avg(sr), "skill_f1": _avg(sf),
        "education_exact_match": _avg(edu_em), "job_type_exact_match": _avg(jt_em),
        "cities_exact_match": _avg(city_em),
    }


def evaluate_jd_extraction(source: str = "store", gold_path: str = None,
                           jd_folder: str = None) -> dict:
    """评 JD 抽取。source='store'（用已入库 jd_profile，0 LLM）/ 'reparse'（重跑 parse_jd，需 LLM）。"""
    profiles_by_id = {}
    json_valid = None

    if source == "reparse":
        from resume2job.parsing.jd_parser import parse_jd
        folder = jd_folder or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "JDs")
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".txt")) if os.path.isdir(folder) else []
        valid = 0
        for fn in files:
            jid = os.path.splitext(fn)[0]
            with open(os.path.join(folder, fn), "r", encoding="utf-8") as f:
                prof = parse_jd(f.read())
            if isinstance(prof, dict) and not prof.get("error"):
                valid += 1
                profiles_by_id[jid] = prof
        json_valid = round(valid / max(1, len(files)), 4)
    else:
        for r in jobs_store.all_rows():
            prof = r.get("jd_profile")
            if isinstance(prof, dict) and prof:
                profiles_by_id[r["job_id"]] = prof

    profiles = list(profiles_by_id.values())
    # 质量分分布（复用 Stage 1 validator）
    q = [validate_job(p, "x" * 100).quality_score for p in profiles]
    report = {
        "source": source,
        "n_profiles": len(profiles),
        "json_valid_rate": json_valid,                 # 仅 reparse 模式有意义
        "field_completeness": field_completeness(profiles),
        "quality_score": {"min": min(q), "avg": round(sum(q) / len(q), 4), "max": max(q)} if q else {},
    }
    gold = _load_gold(gold_path) if gold_path else {}
    if gold:
        report["gold_metrics"] = _gold_metrics(profiles_by_id, gold)
    _save_report("extraction_eval", report)
    return report


def _save_report(name: str, report: dict) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORT_DIR, f"{name}_{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {name} {ts}\n\n```json\n{json.dumps(report, ensure_ascii=False, indent=2)}\n```\n")
    print(f"[{name}] 报告已保存：{path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="抽取层评测（JD 结构化抽取质量）")
    ap.add_argument("--source", choices=["store", "reparse"], default="store",
                    help="store=用已入库画像(0 LLM)；reparse=重跑 parse_jd(需 LLM)")
    ap.add_argument("--gold", default=None, help="gold jsonl 路径（可选，算 P/R/F1 + Exact Match）")
    args = ap.parse_args()
    rep = evaluate_jd_extraction(source=args.source, gold_path=args.gold)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
