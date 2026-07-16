# -*- coding: utf-8 -*-
"""observability/replay.py —— 历史请求回放与前后对比（回归评测的基础）。

用途：改了 Query 改写 / 换 embedding / 换 reranker / 上线 LambdaMART / 改硬约束逻辑 / 微调 Planner 后，
拿一批历史请求重新跑一遍，对比「同一请求」升级前后的结果是否回归（Top-K 变化、分数变化、时延 / token /
错误率变化），而不是只凭几条例子感觉「变好了」。

实现：从 request_traces.jsonl 取出旧 trace 的 user_query + session_id，重新跑一轮 run_turn（**会真实调用
API**，并在该 session 追加一轮对话——回放是开发期工具，副作用可接受），得到新 trace，再 diff_traces 对比。

注意：回放复用原 session 的缓存画像 / 上下文以尽量复现；若原 session 状态已变，结果可能与首次不完全一致，
diff 反映的是「当前系统 + 当前 session 状态」下重跑的差异。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from resume2job.observability import events


# ---------------------------------------------------------------------------
# 回放
# ---------------------------------------------------------------------------
def replay_request(request_id: str, app=None, session_id: Optional[str] = None) -> Tuple[dict, Optional[dict]]:
    """回放单条历史请求，返回 (旧 trace, 新 trace)。新 trace 为 None 表示重跑未产出 trace。"""
    old = events.get_trace(request_id)
    if not old:
        raise ValueError(f"找不到 request_id={request_id} 的历史 trace")

    if app is None:
        from resume2job.agent.graph import build_graph
        app = build_graph()

    from resume2job.agent.graph import run_turn
    sess = session_id or old.get("session_id") or "replay"
    final = run_turn(app, old.get("user_query") or "", session_id=sess)
    new_id = (final or {}).get("request_id")
    new = events.get_trace(new_id) if new_id else None
    return old, new


def replay_batch(request_ids: List[str], app=None) -> dict:
    """回放一批请求，返回逐条 diff + 汇总。"""
    if app is None:
        from resume2job.agent.graph import build_graph
        app = build_graph()
    diffs = []
    for rid in request_ids:
        try:
            old, new = replay_request(rid, app=app)
            diffs.append(diff_traces(old, new))
        except Exception as e:
            diffs.append({"request_id": rid, "error": str(e)})
    return {"n": len(diffs), "diffs": diffs, "summary": _aggregate(diffs)}


# ---------------------------------------------------------------------------
# 对比
# ---------------------------------------------------------------------------
def _ranked_map(trace: dict) -> dict:
    """final_ranked_jobs → {job_id: {rank, rank_score}}。"""
    out = {}
    for j in (trace or {}).get("final_ranked_jobs") or []:
        jid = j.get("job_id")
        if jid:
            out[jid] = {"rank": j.get("rank"), "rank_score": j.get("rank_score")}
    return out


def _ranked_ids(trace: dict) -> list:
    return [j.get("job_id") for j in (trace or {}).get("final_ranked_jobs") or [] if j.get("job_id")]


def diff_traces(old: dict, new: Optional[dict], k: int = 5) -> dict:
    """对比两条 trace 的检索/排序结果与成本。返回结构化 diff（含可读 summary）。"""
    if not new:
        return {"request_id": (old or {}).get("request_id"), "error": "新 trace 缺失（重跑未产出）"}

    old_ids, new_ids = _ranked_ids(old), _ranked_ids(new)
    old_top, new_top = old_ids[:k], new_ids[:k]
    old_set, new_set = set(old_top), set(new_top)
    inter = old_set & new_set
    union = old_set | new_set
    overlap = len(inter) / len(union) if union else 1.0

    om, nm = _ranked_map(old), _ranked_map(new)
    rank_changes = []
    for jid in inter:
        o, n = om.get(jid, {}), nm.get(jid, {})
        if o.get("rank") != n.get("rank") or o.get("rank_score") != n.get("rank_score"):
            rank_changes.append({"job_id": jid, "old_rank": o.get("rank"), "new_rank": n.get("rank"),
                                 "old_score": o.get("rank_score"), "new_score": n.get("rank_score")})

    def _tok(t):
        return ((t or {}).get("token_usage") or {}).get("total_tokens") or 0

    def _lat(t):
        return ((t or {}).get("latency_ms") or {}).get("total") or 0

    diff = {
        "request_id": old.get("request_id"),
        "user_query": old.get("user_query"),
        "topk": k,
        "old_top": old_top, "new_top": new_top,
        "topk_overlap": round(overlap, 3),
        "added": sorted(new_set - old_set),
        "removed": sorted(old_set - new_set),
        "rank_changes": rank_changes,
        "token_delta": _tok(new) - _tok(old),
        "latency_delta_ms": round(_lat(new) - _lat(old), 1),
        "errors_delta": len((new or {}).get("errors") or []) - len((old or {}).get("errors") or []),
        "model_versions_changed": (old.get("model_versions") != new.get("model_versions")),
        "index_version_changed": (old.get("index_version") != new.get("index_version")),
    }
    diff["regressed"] = bool(diff["removed"]) or overlap < 1.0
    diff["summary"] = render_diff(diff)
    return diff


def render_diff(diff: dict) -> str:
    """把单条 diff 渲染成一行人类可读摘要。"""
    if diff.get("error"):
        return f"[{diff.get('request_id')}] 回放失败：{diff['error']}"
    parts = [
        f"Top{diff['topk']} 重合={diff['topk_overlap']:.0%}",
        f"+{diff['added']}" if diff["added"] else "",
        f"-{diff['removed']}" if diff["removed"] else "",
        f"名次/分变化={len(diff['rank_changes'])}",
        f"tokenΔ={diff['token_delta']:+d}",
        f"时延Δ={diff['latency_delta_ms']:+.0f}ms",
    ]
    if diff.get("model_versions_changed"):
        parts.append("模型版本已变")
    if diff.get("index_version_changed"):
        parts.append("索引版本已变")
    return f"[{diff.get('request_id')}] " + "，".join(p for p in parts if p)


def _aggregate(diffs: List[dict]) -> dict:
    """批量回放汇总。"""
    valid = [d for d in diffs if not d.get("error")]
    if not valid:
        return {"valid": 0, "errors": len(diffs)}
    n = len(valid)
    avg_overlap = sum(d["topk_overlap"] for d in valid) / n
    regressed = [d["request_id"] for d in valid if d.get("regressed")]
    return {
        "valid": n,
        "errors": len(diffs) - n,
        "avg_topk_overlap": round(avg_overlap, 3),
        "n_regressed": len(regressed),
        "regressed_ids": regressed,
        "total_token_delta": sum(d["token_delta"] for d in valid),
        "avg_latency_delta_ms": round(sum(d["latency_delta_ms"] for d in valid) / n, 1),
    }
