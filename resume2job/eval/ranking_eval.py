# -*- coding: utf-8 -*-
"""eval/ranking_eval.py —— 检索 / 排序评测（在相关性指标之上加运营 / 约束 / 成本指标）。

现有 retrieval_eval 已覆盖相关性（Recall@K / MRR / nDCG，合成评测集 + search_jobs）。本模块**补充**
真实请求链路的运营指标——它们回答「排序变好是否以约束违例 / 头部集中 / 时延成本飙升为代价」：
    - 约束违例率   ：最终岗位是否违反该请求的城市 / 岗位类型硬约束（应≈0，eligibility 预筛已保证；非 0 即 bug）；
    - 公司集中度   ：单请求 Top-K 同公司重复率 + 全局公司曝光头部占比（HHI / Top-1 份额）；
    - 新鲜岗位占比 ：最终岗位中较新（collected_at 在 N 天内）的比例；
    - 方向覆盖     ：最终岗位的方向分布；
    - 各阶段时延   ：每个节点的平均耗时；
    - 单请求成本   ：平均 token / 平均总时延。

数据源 = `data/request_traces.jsonl`（Stage 2，真实请求）。运营指标 0 LLM；--relevance 时另跑 retrieval_eval（需 API）。
"""

import os
import json
import argparse
import collections
from datetime import datetime

from resume2job.observability import events
from resume2job.storage import jobs_store
from resume2job.parsing.jd_parser import normalize_job_type

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
FRESH_DAYS = 30


def _loads(v, default):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return json.loads(v)
        except Exception:
            return default
    return default


def _final_ids(trace: dict) -> list:
    return [j.get("job_id") for j in (trace.get("final_ranked_jobs") or []) if j.get("job_id")]


def evaluate_operational(traces: list) -> dict:
    """从真实 request trace 算运营 / 约束 / 成本指标。"""
    traces = [t for t in traces if _final_ids(t)]
    if not traces:
        return {"n_requests": 0}

    # 预取所有最终岗位的元数据
    all_ids = sorted({jid for t in traces for jid in _final_ids(t)})
    rows = jobs_store.get_jobs_by_ids(all_ids)

    city_viol = jt_viol = pair_count = 0
    dup_rates, fresh_flags = [], []
    company_exposure = collections.Counter()
    direction_exposure = collections.Counter()
    source_exposure = collections.Counter()
    city_exposure = collections.Counter()
    lat_by_node = collections.defaultdict(list)
    tokens, totals = [], []

    for t in traces:
        ids = _final_ids(t)
        hc = (t.get("query_plan") or {}).get("hard_constraints") or {}
        want_city = hc.get("city") or None
        want_jt = normalize_job_type(hc.get("job_type")) if hc.get("job_type") else None
        companies = []
        for jid in ids:
            r = rows.get(jid) or {}
            pair_count += 1
            cities = _loads(r.get("cities_json"), [])
            jtypes = _loads(r.get("job_types_json"), [])
            # 城市违例：指定了城市、岗位有明确城市、且不含该城市、且城市非「待确认」
            if want_city and cities and r.get("city_status") != "unknown" and want_city not in cities:
                city_viol += 1
            # 类型违例：指定了类型、岗位类型数组非空、且不含该类型
            if want_jt and jtypes and want_jt not in jtypes:
                jt_viol += 1
            companies.append(r.get("company") or "未知")
            company_exposure[r.get("company") or "未知"] += 1
            direction_exposure[r.get("direction") or "未知"] += 1
            source_exposure[r.get("source") or "未知"] += 1
            for c in (cities or ["未知"]):
                city_exposure[c] += 1
            fr = _freshness_days(t.get("created_at"), r.get("collected_at") or r.get("created_at"))
            fresh_flags.append(1.0 if (fr is not None and fr <= FRESH_DAYS) else 0.0)
        if companies:
            dup_rates.append(1 - len(set(companies)) / len(companies))
        lat = t.get("latency_ms") or {}
        for node, ms in lat.items():
            if node != "total" and isinstance(ms, (int, float)):
                lat_by_node[node].append(ms)
        tu = (t.get("token_usage") or {}).get("total_tokens")
        if isinstance(tu, (int, float)):
            tokens.append(tu)
        if isinstance(lat.get("total"), (int, float)):
            totals.append(lat["total"])

    n = len(traces)
    return {
        "n_requests": n,
        "constraint_violation": {
            "city_violation_rate": round(city_viol / max(1, pair_count), 4),
            "job_type_violation_rate": round(jt_viol / max(1, pair_count), 4),
            "n_candidate_pairs": pair_count,
        },
        "company_concentration": {
            "avg_same_company_dup_rate": round(sum(dup_rates) / len(dup_rates), 4) if dup_rates else 0.0,
            "top1_company_share": _top_share(company_exposure),
            "hhi": _hhi(company_exposure),
            "top5_companies": company_exposure.most_common(5),
        },
        "fresh_job_ratio": round(sum(fresh_flags) / len(fresh_flags), 4) if fresh_flags else None,
        "exposure": {
            "by_direction": dict(direction_exposure.most_common(10)),
            "by_source": dict(source_exposure),
            "by_city": dict(city_exposure.most_common(10)),
        },
        "latency_ms_by_stage": {k: round(sum(v) / len(v), 1) for k, v in lat_by_node.items()},
        "cost_per_request": {
            "avg_total_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
            "avg_total_latency_ms": round(sum(totals) / len(totals), 1) if totals else None,
        },
    }


def _freshness_days(later_iso, earlier_iso):
    if not later_iso or not earlier_iso:
        return None
    try:
        a = datetime.fromisoformat(str(later_iso))
        b = datetime.fromisoformat(str(earlier_iso))
        return (a - b).total_seconds() / 86400.0
    except Exception:
        return None


def _top_share(counter: collections.Counter) -> float:
    total = sum(counter.values())
    return round(counter.most_common(1)[0][1] / total, 4) if total else 0.0


def _hhi(counter: collections.Counter) -> float:
    """Herfindahl 指数：Σ(share²)，1=全集中于一家，→0=分散。"""
    total = sum(counter.values())
    if not total:
        return 0.0
    return round(sum((c / total) ** 2 for c in counter.values()), 4)


def run_ranking_eval(with_relevance: bool = False, top_k: int = 5) -> dict:
    report = {"operational": evaluate_operational(events.load_traces())}
    if with_relevance:
        from resume2job.eval.retrieval_eval import run_retrieval_eval
        report["relevance"] = run_retrieval_eval(top_k=top_k)
    _save_report("ranking_eval", report)
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
    ap = argparse.ArgumentParser(description="检索 / 排序评测（运营 + 约束 + 成本；可选相关性）")
    ap.add_argument("--relevance", action="store_true", help="另跑 retrieval_eval 相关性指标（需 API）")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
    rep = run_ranking_eval(with_relevance=args.relevance, top_k=args.top_k)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
