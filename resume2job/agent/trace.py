"""
agent/trace.py

逐节点链路追踪（回归测试 / 调试用），是 pipeline.py 端到端验收的「中间路径可见性」基础设施。

设计要点：
    1. 零侵入——不改任何业务节点。依赖 LangGraph 的 app.stream(stream_mode="updates")
       逐节点产物，本模块只把每个节点的产物 diff 成「改动字段 + 关键信号快照」，
       串成一条 TurnTrace；
    2. 单一枢纽——summarize_state 同时服务于 trace 快照、节点级断言、报告渲染，
       三处共用一套关键信号定义，避免口径漂移；
    3. 可落盘、可渲染——TurnTrace 既能 dump 成 JSON 供失败复盘，也能渲染成
       Markdown 时序表并入验收报告。

本文件只做追踪与呈现，不含任何业务逻辑，也不参与图的执行决策。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 关键信号快照（枢纽函数）
# ---------------------------------------------------------------------------
def summarize_state(s: Dict[str, Any]) -> Dict[str, Any]:
    """从（任意阶段的）State 抽取一组体积可控、可读的关键信号。

    不放整份 State（candidate_jobs / messages 等可能很大），只取链路诊断真正需要的字段。
    trace 快照、节点级断言（Checker.at）、报告时序表三处共用本函数，保证口径一致。
    """
    mrs = s.get("match_results") or []
    top_score = None
    if mrs:
        top_score = (mrs[0].get("match_score") or {}).get("rank_score")
    return {
        "intent": s.get("intent"),
        "plan": s.get("plan"),
        "retrieval_config": s.get("retrieval_config"),
        "commute_intent": s.get("commute_intent"),
        "profile_source": s.get("profile_source"),
        "n_candidates": len(s.get("candidate_jobs") or []),
        "n_jd_profiles": len(s.get("jd_profiles") or []),
        "n_match_results": len(mrs),
        "top_score": top_score,
        "ingested_job_id": s.get("ingested_job_id"),
        "jd_is_duplicate": s.get("jd_is_duplicate"),
        "n_learning_stages": len((s.get("learning_plan") or {}).get("stages") or []),
        "n_interview_q": len((s.get("interview_prep") or {}).get("questions") or []),
        "n_commute_results": len(s.get("commute_results") or []),
        "has_final_response": bool((s.get("final_response") or "").strip()),
        "n_errors": len(s.get("errors") or []),
    }


def changed_keys(prev: Dict[str, Any], cur: Dict[str, Any]) -> List[str]:
    """对比前后两份 State，返回值发生变化的字段名（排序后）。

    用于标注「这个节点写了哪些字段」。对大列表做深比较，测试规模下开销可接受。
    """
    keys = set(prev) | set(cur)
    out = []
    for k in keys:
        if k == "messages":  # 历史消息体积大且节点不改，跳过比较
            continue
        if prev.get(k) != cur.get(k):
            out.append(k)
    return sorted(out)


# ---------------------------------------------------------------------------
# TurnTrace：一轮执行的逐节点链路
# ---------------------------------------------------------------------------
class TurnTrace:
    """记录单轮（一次 run_turn）的逐节点产物，可落盘、可渲染为报告时序表。"""

    def __init__(self, trace_id: str, session_id: str, user_query: str):
        self.trace_id = trace_id
        self.session_id = session_id
        self.user_query = user_query
        self.steps: List[Dict[str, Any]] = []
        self.total_elapsed: float = 0.0

    def record(self, node: str, elapsed_s: float, changed: List[str],
               summary: Dict[str, Any], errors_added: List[str]) -> None:
        """追加一个节点步（由 run_turn_traced 在每个 stream chunk 后调用）。"""
        self.steps.append({
            "seq": len(self.steps) + 1,
            "node": node,
            "elapsed_s": round(elapsed_s, 3),
            "changed_keys": changed,
            "summary": summary,
            "errors_added": list(errors_added or []),
        })
        self.total_elapsed += elapsed_s

    def summary_at(self, node: str) -> Optional[Dict[str, Any]]:
        """返回指定节点执行后的关键信号快照（取最后一次出现）。

        供 Checker.at(node, ...) 做节点级断言；节点未执行则返回 None。
        """
        for step in reversed(self.steps):
            if step["node"] == node:
                return step["summary"]
        return None

    def nodes(self) -> List[str]:
        """按执行顺序返回节点名序列。"""
        return [step["node"] for step in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "total_elapsed_s": round(self.total_elapsed, 3),
            "node_path": " → ".join(self.nodes()),
            "steps": self.steps,
        }

    def dump_json(self, path: str) -> None:
        """把整条 trace 落盘为 JSON（失败复盘用）。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    # ---------- 报告渲染 ----------
    def as_markdown_table(self) -> str:
        """渲染为 Markdown 时序表：seq / 节点 / 耗时 / 改动字段 / 关键信号。"""
        lines = [
            f"路径：`{' → '.join(self.nodes())}`（总耗时 {self.total_elapsed:.1f}s）",
            "",
            "| seq | 节点 | 耗时(s) | 改动字段 | 关键信号 |",
            "|---|---|---|---|---|",
        ]
        for step in self.steps:
            signals = self._fmt_signals(step["summary"])
            changed = ", ".join(step["changed_keys"]) or "—"
            err_flag = f" ⚠️+{len(step['errors_added'])}err" if step["errors_added"] else ""
            lines.append(
                f"| {step['seq']} | {step['node']}{err_flag} | "
                f"{step['elapsed_s']:.1f} | {changed} | {signals} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _fmt_signals(summary: Dict[str, Any]) -> str:
        """挑选非空 / 有意义的信号拼成一行，避免把整个 summary 铺进表格。"""
        rc = summary.get("retrieval_config") or {}
        parts = []
        if summary.get("intent"):
            parts.append(f"intent={summary['intent']}")
        if rc.get("city_filter"):
            parts.append(f"city={rc['city_filter']}")
        if summary.get("profile_source"):
            parts.append(f"profile={summary['profile_source']}")
        if summary.get("n_candidates"):
            parts.append(f"召回={summary['n_candidates']}")
        if summary.get("n_jd_profiles"):
            parts.append(f"jd={summary['n_jd_profiles']}")
        if summary.get("n_match_results"):
            parts.append(f"评分={summary['n_match_results']}")
        if summary.get("top_score") is not None:
            parts.append(f"top={summary['top_score']}")
        if summary.get("ingested_job_id"):
            parts.append(f"ingest={summary['ingested_job_id']}")
        if summary.get("commute_intent"):
            ci = summary["commute_intent"]
            parts.append(f"通勤={ci.get('preferred_transport')}/{ci.get('max_commute_minutes')}min")
        if summary.get("n_learning_stages"):
            parts.append(f"学习={summary['n_learning_stages']}阶段")
        if summary.get("n_interview_q"):
            parts.append(f"面试={summary['n_interview_q']}题")
        if summary.get("n_commute_results"):
            parts.append(f"通勤结果={summary['n_commute_results']}")
        if summary.get("has_final_response"):
            parts.append("final_resp✓")
        if summary.get("n_errors"):
            parts.append(f"errors={summary['n_errors']}")
        return "，".join(parts) or "—"
