# -*- coding: utf-8 -*-
"""eval/fairness_audit.py —— 曝光与数据质量审计（**不做人口统计学公平**）。

明确立场（治理展示）：当前**不**声称「用户群体公平」——没有可靠人口统计数据，也**不应**从简历推断性别 /
地域背景 / 学校层级。本模块做的是招聘推荐里真实、可度量的治理问题：
    1. 约束一致性：城市 / 岗位类型硬约束违例率、unknown 地点岗位被推荐率；
    2. 曝光分布   ：公司 / 方向 / 来源 / 城市曝光占比、Top-K 公司集中度（HHI / Top-1 份额）、同公司重复率；
    3. 数据质量   ：各来源缺字段率、各来源 / 城市 / 方向的质量分、各方向技能抽取失败率、过期岗位比例。

目的不是「证明系统已公平」，而是让你**发现问题**：某公司占了 70% Top-10 / 北京因数据多被过度曝光 /
unknown 地点岗被误推 / 某类岗位 JD 解析总失败。数据源 = jobs 表（目录）+ request_traces（曝光，Stage 2）。
全程 0 LLM。
"""

import os
import json
import argparse
import collections
from datetime import datetime

from resume2job.observability import events
from resume2job.storage import jobs_store
from resume2job.ingest.validator import validate_job
from resume2job.eval.ranking_eval import evaluate_operational, _loads, _freshness_days  # 复用曝光/约束计算

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

_DISCLAIMER = ("本审计为『曝光分布 + 约束一致性 + 数据质量』治理视角，**不做人口统计学公平判定**，"
               "也不从简历推断性别 / 地域 / 学校层级。用于发现集中度 / 偏斜 / 数据质量问题，而非声称系统已公平。")


def audit_catalog() -> dict:
    """目录侧数据质量审计（jobs 表全量）。"""
    rows = jobs_store.all_rows()
    n = len(rows)
    if not n:
        return {"n_jobs": 0}
    status = collections.Counter(r.get("status") or "active" for r in rows)
    by_source_q = collections.defaultdict(list)
    by_source_missing = collections.defaultdict(collections.Counter)
    by_direction_skillfail = collections.defaultdict(lambda: [0, 0])  # [无技能数, 总数]
    unknown_city = 0
    for r in rows:
        prof = r.get("jd_profile") or {}
        src = r.get("source") or "未知"
        rep = validate_job(prof, r.get("jd_text") or "")
        by_source_q[src].append(rep.quality_score)
        for w in rep.warnings:
            by_source_missing[src][w] += 1
        d = prof.get("direction") or "未知"
        by_direction_skillfail[d][1] += 1
        if not (prof.get("hard_skills") or prof.get("tools_or_frameworks")):
            by_direction_skillfail[d][0] += 1
        if (r.get("city_status") or "unknown") == "unknown":
            unknown_city += 1

    return {
        "n_jobs": n,
        "status_dist": dict(status),
        "expired_or_removed_ratio": round((status.get("expired", 0) + status.get("removed", 0)) / n, 4),
        "unknown_city_ratio": round(unknown_city / n, 4),
        "quality_by_source": {s: round(sum(q) / len(q), 4) for s, q in by_source_q.items()},
        "missing_field_rate_by_source": {
            s: {w: round(c / len(by_source_q[s]), 4) for w, c in cc.most_common(6)}
            for s, cc in by_source_missing.items()
        },
        "skill_extraction_fail_rate_by_direction": {
            d: round(fail / tot, 4) for d, (fail, tot) in by_direction_skillfail.items() if tot
        },
    }


def audit_unknown_location_recs(traces: list) -> dict:
    """unknown 地点岗位被推荐率：指定了城市约束时，最终岗位里 city_status=unknown 的占比（地点未明确仍被推）。"""
    constrained_pairs = unknown_recs = 0
    rows_cache = {}
    for t in traces:
        hc = (t.get("query_plan") or {}).get("hard_constraints") or {}
        if not hc.get("city"):
            continue
        ids = [j.get("job_id") for j in (t.get("final_ranked_jobs") or []) if j.get("job_id")]
        if not ids:
            continue
        missing = [j for j in ids if j not in rows_cache]
        if missing:
            rows_cache.update(jobs_store.get_jobs_by_ids(missing))
        for jid in ids:
            constrained_pairs += 1
            if (rows_cache.get(jid) or {}).get("city_status") == "unknown":
                unknown_recs += 1
    return {
        "n_constrained_pairs": constrained_pairs,
        "unknown_location_rec_rate": round(unknown_recs / constrained_pairs, 4) if constrained_pairs else None,
    }


def run_fairness_audit() -> dict:
    traces = events.load_traces()
    op = evaluate_operational(traces)
    report = {
        "disclaimer": _DISCLAIMER,
        "constraint_consistency": {
            **(op.get("constraint_violation") or {}),
            **audit_unknown_location_recs(traces),
        },
        "exposure": {
            **(op.get("exposure") or {}),
            "company_concentration": op.get("company_concentration") or {},
        },
        "data_quality": audit_catalog(),
        "n_requests_audited": op.get("n_requests", 0),
    }
    _save_report("fairness_audit", report)
    return report


def _save_report(name: str, report: dict) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORT_DIR, f"{name}_{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {name} {ts}\n\n> {_DISCLAIMER}\n\n```json\n{json.dumps(report, ensure_ascii=False, indent=2)}\n```\n")
    print(f"[{name}] 报告已保存：{path}")
    return path


if __name__ == "__main__":
    argparse.ArgumentParser(description="曝光与数据质量审计（非人口统计学公平）").parse_args()
    rep = run_fairness_audit()
    print(json.dumps(rep, ensure_ascii=False, indent=2))
