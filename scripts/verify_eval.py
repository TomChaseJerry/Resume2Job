# -*- coding: utf-8 -*-
"""
scripts/verify_eval.py —— Stage 4（resume2job/eval 四层评测）的可复盘验收脚本。

四层：抽取(extraction) / Planner / 排序运营(ranking) / 曝光审计(fairness)。**默认全程 0 LLM**
（读已入库画像 + planner_traces + request_traces + jobs 目录）。相关性指标(Recall/MRR/nDCG)由既有
retrieval_eval 负责，需 API，本脚本不跑（用 `python -m resume2job.eval.run_eval --retrieval`）。

用法：python scripts/verify_eval.py
"""

import os
import sys
import json
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def hr(t): print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)
def ok(m): print("  [OK] " + m)
def info(m): print("  - " + m)


def main():
    argparse.ArgumentParser(description="Stage 4 eval 四层评测验收（0 LLM）").parse_args()

    hr("1. 抽取层 extraction_eval（JD 抽取完整度 / 质量；用已入库画像，0 LLM）")
    from resume2job.eval.extraction_eval import evaluate_jd_extraction
    ext = evaluate_jd_extraction(source="store")
    info(f"画像数={ext['n_profiles']}；字段完整度={ext['field_completeness']}")
    info(f"质量分={ext['quality_score']}")
    assert ext["n_profiles"] > 0 and "hard_skills" in ext["field_completeness"]
    ok("抽取完整度 + 质量分布已出（gold 可选：传 --gold 算技能 P/R/F1 + Exact Match）")

    hr("2. Planner 层 planner_eval（挖 planner_traces，0 LLM）")
    from resume2job.eval.planner_eval import run_planner_eval
    pl = run_planner_eval()
    if pl.get("summary"):
        s = pl["summary"]
        info(f"trace数={s['n_traces']}；intent分布={s['intent_dist']}；规则兜底率={s['rule_fallback_rate']}；澄清率={s['clarify_rate']}")
        info(f"工具请求率={s['tool_request_rate']}")
        ok("意图/动作分布 + 规则兜底率 + 澄清率 + 工具请求率已出（gold 可选：算意图/路由/槽位/澄清准确率）")
    else:
        info("暂无 planner_traces（跑若干 run_turn / chat.py 即可产生）")

    hr("3. 排序运营 ranking_eval（挖 request_traces，0 LLM）")
    from resume2job.eval.ranking_eval import run_ranking_eval
    rk = run_ranking_eval(with_relevance=False)
    op = rk["operational"]
    if op.get("n_requests"):
        info(f"请求数={op['n_requests']}；约束违例={op['constraint_violation']}")
        info(f"公司集中度={op['company_concentration']}")
        info(f"各阶段时延(ms)={op['latency_ms_by_stage']}；单请求成本={op['cost_per_request']}")
        assert "city_violation_rate" in op["constraint_violation"]
        ok("约束违例率 + 公司集中度 + 新鲜度 + 各阶段时延 + 单请求成本已出（相关性指标用 --relevance 另跑，需 API）")
    else:
        info("暂无 request_traces（跑一次 run_turn 即可产生）")

    hr("4. 曝光与数据质量审计 fairness_audit（0 LLM；非人口统计学公平）")
    from resume2job.eval.fairness_audit import run_fairness_audit
    fa = run_fairness_audit()
    info("立场：" + fa["disclaimer"][:46] + "…")
    dq = fa["data_quality"]
    info(f"目录：岗位{dq.get('n_jobs')}、过期率={dq.get('expired_or_removed_ratio')}、unknown城市率={dq.get('unknown_city_ratio')}")
    info(f"各来源缺字段率={dq.get('missing_field_rate_by_source')}")
    info(f"约束一致性={fa['constraint_consistency']}")
    assert "disclaimer" in fa and "data_quality" in fa and "constraint_consistency" in fa
    ok("曝光分布 + 约束一致性 + 数据质量审计已出，且明确不做人口统计学公平判定")

    hr("验收完成")
    print("  报告落盘：resume2job/eval/reports/{extraction,planner,ranking,fairness}_*.md")
    print("  一键入口：python -m resume2job.eval.run_eval --extraction --planner --ranking --fairness")


if __name__ == "__main__":
    main()
