# -*- coding: utf-8 -*-
"""
scripts/export_planner_traces.py

把 planner 每轮决策 trace（resume2job/agent/planner/trace_logger.py 落的
data/planner_traces.jsonl）导出为后训练数据闭环的两类产物：

    1. 决策质量统计：意图分布、job_source 分布、LLM vs 规则兜底比例、澄清率、
       assist_actions 频次、平均置信度——监控 planner 在真实流量上的行为。
    2. SFT 训练样本（messages 格式）：user_query → PlannerOutput(JSON)，
       可按置信度过滤、排除澄清轮、区分是否纳入规则兜底样本。

定位：后训练数据闭环的「采集 → 导出」环节（第一阶段 trace_logger 只写不用，
这里消费）。本脚本是纯离线数据处理，不调用任何 LLM / 外部 API。

用法：
    python scripts/export_planner_traces.py --stats-only          # 只看统计
    python scripts/export_planner_traces.py --out data/sft.jsonl  # 导出 SFT 样本
    python scripts/export_planner_traces.py --min-confidence 0.7 --include-rule
"""

import os
import sys
import json
import argparse
from collections import Counter

# 脚本位于 scripts/ 下，手动把项目根目录加进 sys.path 以便导入 resume2job 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# SFT system 指令：复用 NLU 模块的 system prompt，保证训练目标与线上一致
from resume2job.agent.planner.nlu_extractor import _SYSTEM as _NLU_SYSTEM
from resume2job.storage.paths import DATA_DIR

DEFAULT_TRACE_PATH = os.path.join(DATA_DIR, "planner_traces.jsonl")
# SFT 输出字段：只保留语义决策（不含执行计划 plan，那是确定性映射、无需学）
_SFT_FIELDS = ("intent", "job_source", "request_more", "assist_actions",
               "hard_constraints", "soft_preferences", "commute",
               "selected_item_ref", "missing_slots")


def load_traces(path: str) -> list:
    """读取 trace jsonl，返回记录列表。文件不存在返回 []。"""
    if not os.path.isfile(path):
        print(f"[WARN] trace 文件不存在：{path}（planner 尚未产生 trace？）")
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compute_stats(rows: list) -> dict:
    """统计 planner 在真实流量上的决策分布。"""
    intents, sources, decided, assists = (Counter() for _ in range(4))
    request_more = clarified = 0
    confidences = []
    for r in rows:
        out = r.get("planner_output") or {}
        intents[out.get("intent")] += 1
        sources[out.get("job_source")] += 1
        decided[r.get("decided_by")] += 1
        if out.get("request_more"):
            request_more += 1
        for a in (out.get("assist_actions") or []):
            assists[a] += 1
        if r.get("clarified"):
            clarified += 1
        if isinstance(out.get("confidence"), (int, float)):
            confidences.append(float(out["confidence"]))
    n = len(rows)
    return {
        "total": n,
        "intent": dict(intents),
        "job_source": dict(sources),
        "assist_actions": dict(assists),
        "request_more_count": request_more,
        "decided_by": dict(decided),          # llm vs rule_fallback：兜底率反映 LLM 可靠度
        "clarify_rate": round(clarified / n, 3) if n else 0.0,
        "rule_fallback_rate": round(decided.get("rule_fallback", 0) / n, 3) if n else 0.0,
        "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
    }


def to_sft_samples(rows: list, min_confidence: float, include_rule: bool,
                   include_clarify: bool) -> list:
    """转 SFT 样本（messages）。默认只取 LLM 决策、非澄清、置信度达标的「干净正样本」。"""
    samples = []
    for r in rows:
        out = r.get("planner_output") or {}
        if not include_rule and r.get("decided_by") != "llm":
            continue
        if not include_clarify and r.get("clarified"):
            continue
        conf = out.get("confidence")
        if isinstance(conf, (int, float)) and conf < min_confidence:
            continue
        target = {k: out.get(k) for k in _SFT_FIELDS if k in out}
        samples.append({
            "messages": [
                {"role": "system", "content": _NLU_SYSTEM},
                {"role": "user", "content": r.get("user_query") or ""},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
            ]
        })
    return samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="导出 planner trace：决策统计 + SFT 样本")
    parser.add_argument("--trace", default=DEFAULT_TRACE_PATH, help="trace jsonl 路径")
    parser.add_argument("--out", default=None, help="SFT 样本输出 jsonl；省略则只打印统计")
    parser.add_argument("--stats-only", action="store_true", help="只输出统计，不导样本")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="SFT 样本置信度下限")
    parser.add_argument("--include-rule", action="store_true", help="纳入规则兜底样本")
    parser.add_argument("--include-clarify", action="store_true", help="纳入澄清轮样本")
    args = parser.parse_args(argv)

    rows = load_traces(args.trace)
    stats = compute_stats(rows)
    print("===== planner 决策质量统计 =====")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.stats_only or not args.out:
        return 0

    samples = to_sft_samples(rows, args.min_confidence, args.include_rule, args.include_clarify)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n[导出] SFT 样本 {len(samples)}/{len(rows)} 条 → {args.out}"
          f"（min_confidence={args.min_confidence}, include_rule={args.include_rule},"
          f" include_clarify={args.include_clarify}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
