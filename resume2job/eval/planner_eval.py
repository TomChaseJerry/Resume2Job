# -*- coding: utf-8 -*-
"""eval/planner_eval.py —— Planner 层评测（意图理解 / 路由 / 槽位 / 澄清 / 工具调用）。

后训练最重要的不是「回答自不自然」，而是**有没有走对系统路径**（用户只想看岗位介绍，Planner 却调高德 API
就是明显错误）。本模块评 Planner 是否正确理解用户并进入正确路径。

数据源 = `data/planner_traces.jsonl`（Stage 之前就在采的 Planner 决策 trace：user_query → planner_output
(intent/hard_constraints/soft_preferences) → plan (session_action/need_*/clarify) → decided_by/clarified）。

两种口径：
    1. **无 gold（始终可跑，0 LLM）**：分布与一致性——intent / session_action 分布、规则兜底率
       （decided_by=rule_fallback，高=NLU LLM 常失败）、澄清率、各工具请求率。
    2. **有 gold（可选）**：按 user_query 对齐金标，算 intent / 路由(session_action) / 槽位(city/job_type/
       education) 准确率、澄清准确率、工具「该调/误调」率。
gold jsonl 示例：{"query":"帮我找北京的Agent实习","intent":"RECOMMEND","session_action":"RETRIEVE",
                "hard_constraints":{"city":"北京","job_type":"实习"},"clarify":false}
"""

import os
import json
import argparse
import collections
from datetime import datetime

from resume2job.storage.paths import DATA_DIR

TRACES_PATH = os.path.join(DATA_DIR, "planner_traces.jsonl")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
_TOOL_FLAGS = ("need_commute", "need_learning_plan", "need_interview")
_SLOTS = ("city", "job_type", "education")


def load_planner_traces(path: str = TRACES_PATH) -> list:
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def summarize(traces: list) -> dict:
    """无 gold 的分布与一致性指标。"""
    n = max(1, len(traces))
    intents = collections.Counter((t.get("planner_output") or {}).get("intent") for t in traces)
    actions = collections.Counter((t.get("plan") or {}).get("session_action") for t in traces)
    decided = collections.Counter(t.get("decided_by") for t in traces)
    clarify = sum(1 for t in traces if (t.get("plan") or {}).get("clarify") or t.get("clarified"))
    tool_rates = {}
    for flag in _TOOL_FLAGS:
        tool_rates[flag] = round(sum(1 for t in traces if (t.get("plan") or {}).get(flag)) / n, 4)
    return {
        "n_traces": len(traces),
        "intent_dist": dict(intents),
        "session_action_dist": dict(actions),
        "decided_by_dist": dict(decided),
        "rule_fallback_rate": round(decided.get("rule_fallback", 0) / n, 4),
        "clarify_rate": round(clarify / n, 4),
        "tool_request_rate": tool_rates,
    }


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
            if r.get("query"):
                gold[r["query"].strip()] = r
    return gold


def evaluate_with_gold(traces: list, gold: dict) -> dict:
    """按 user_query 对齐金标，算意图 / 路由 / 槽位 / 澄清准确率（宏平均）。"""
    by_query = {(t.get("user_query") or "").strip(): t for t in traces}
    intent_hits, action_hits, clarify_hits, n = [], [], [], 0
    slot_hits = {s: [] for s in _SLOTS}
    for q, g in gold.items():
        t = by_query.get(q.strip())
        if not t:
            continue
        n += 1
        out = t.get("planner_output") or {}
        plan = t.get("plan") or {}
        if g.get("intent") is not None:
            intent_hits.append(1.0 if out.get("intent") == g["intent"] else 0.0)
        if g.get("session_action") is not None:
            action_hits.append(1.0 if plan.get("session_action") == g["session_action"] else 0.0)
        if "clarify" in g:
            clarify_hits.append(1.0 if bool(plan.get("clarify")) == bool(g["clarify"]) else 0.0)
        ghc = g.get("hard_constraints") or {}
        phc = out.get("hard_constraints") or {}
        for s in _SLOTS:
            if s in ghc:
                slot_hits[s].append(1.0 if (phc.get(s) or None) == (ghc.get(s) or None) else 0.0)

    def _avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    return {
        "n_gold_matched": n,
        "intent_accuracy": _avg(intent_hits),
        "route_accuracy": _avg(action_hits),
        "clarify_accuracy": _avg(clarify_hits),
        "slot_accuracy": {s: _avg(slot_hits[s]) for s in _SLOTS},
    }


def run_planner_eval(traces_path: str = TRACES_PATH, gold_path: str = None) -> dict:
    traces = load_planner_traces(traces_path)
    if not traces:
        print(f"[planner_eval] 无 planner_traces：{traces_path}（先跑若干 run_turn / chat.py 产生决策 trace）")
        return {}
    report = {"summary": summarize(traces)}
    gold = _load_gold(gold_path) if gold_path else {}
    if gold:
        report["gold_metrics"] = evaluate_with_gold(traces, gold)
    _save_report("planner_eval", report)
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
    ap = argparse.ArgumentParser(description="Planner 层评测（意图 / 路由 / 槽位 / 澄清）")
    ap.add_argument("--traces", default=TRACES_PATH)
    ap.add_argument("--gold", default=None, help="gold jsonl（可选，算准确率）")
    args = ap.parse_args()
    rep = run_planner_eval(traces_path=args.traces, gold_path=args.gold)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
