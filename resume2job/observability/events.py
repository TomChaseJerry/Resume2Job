# -*- coding: utf-8 -*-
"""observability/events.py —— 请求级链路追踪（每次推荐一条 request trace）。

回答「用户这次推荐结果到底经历了什么」：一次 run_turn = 一个 request_id，沿途各阶段（规划 / 召回各通道 /
融合 / 精排 / 评分 / 工具 / LLM token 与时延）把快照写进**当前请求的 RequestTrace**，结束时落盘。
区别于既有两套 trace：planner_traces 只覆盖「规划输入→输出」；agent/trace.py 是 pipeline 验收用的逐节点
State diff。本模块是**生产链路**的全量可观测 + 回放/对比的基础（replay.py / Stage 3 ranking 特征都吃它）。

注入方式（与用户确认的方案一致）：**ContextVar 记录器**。run_turn 用 request_scope 开一个 trace 设进
contextvar；各处调 record_*（无活跃 trace 时全部 no-op，故 pipeline / eval / 直接调用模块均不受影响）。
core/llm.py 包一层自动记 token+时延；节点经 run_node 包一层记每节点时延 + 设「当前节点」供 LLM 调用归属。

落盘：完整 trace 追加到 data/request_traces.jsonl（不可变事实）；另写 SQLite request_traces 摘要表（可查询 +
承载事后回填的 user_feedback）。任何记录 / 落盘失败都被吞掉，绝不影响主链路。
"""

from __future__ import annotations

import os
import json
import time
import sqlite3
import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from resume2job.storage.paths import SQLITE_PATH, DATA_DIR

JSONL_PATH = os.path.join(DATA_DIR, "request_traces.jsonl")
DB_PATH = SQLITE_PATH

# 单条 trace 里每个候选列表的最大长度（防极端情况下 trace 膨胀）
_MAX_CANDIDATES = 100

# 当前请求的 trace / 当前执行的节点名（跨节点 + LLM 调用共享）
_current: contextvars.ContextVar[Optional["RequestTrace"]] = contextvars.ContextVar(
    "r2j_request_trace", default=None)
_current_node: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "r2j_current_node", default=None)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# RequestTrace
# ---------------------------------------------------------------------------
class RequestTrace:
    """一次请求的完整链路快照（可落盘、可被 replay 对比）。"""

    def __init__(self, request_id: str, session_id: str, user_query: str,
                 model_versions: Optional[dict] = None, index_version: Optional[dict] = None):
        self.request_id = request_id
        self.session_id = session_id
        self.user_query = user_query
        self.created_at = _utc_now_iso()
        self.model_versions = model_versions or {}
        self.index_version = index_version or {}

        self.query_plan: Dict[str, Any] = {}
        self.retrieval: Dict[str, Any] = {
            "queries": {}, "constraint_filter": {}, "allowed_count": None,
            "dense_candidates": [], "bm25_candidates": [],
            "rrf_candidates": [], "rerank": [],
        }
        self.rank_features: List[dict] = []
        self.final_ranked_jobs: List[dict] = []
        self.tool_calls: List[dict] = []
        self.llm_calls: List[dict] = []
        self.latency_ms: Dict[str, float] = {}
        self.errors: List[dict] = []
        self.user_feedback: Optional[str] = None
        self._t0 = time.perf_counter()

    # ---- 累加器 ----
    def add_latency(self, node: str, ms: float) -> None:
        self.latency_ms[node] = round(self.latency_ms.get(node, 0.0) + ms, 1)

    def add_llm_call(self, call: dict) -> None:
        self.llm_calls.append(call)

    def _token_usage(self) -> dict:
        prompt = sum(int(c.get("prompt_tokens") or 0) for c in self.llm_calls)
        completion = sum(int(c.get("completion_tokens") or 0) for c in self.llm_calls)
        total = sum(int(c.get("total_tokens") or 0) for c in self.llm_calls)
        by_node: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        for c in self.llm_calls:
            t = int(c.get("total_tokens") or 0)
            by_node[c.get("node") or "?"] = by_node.get(c.get("node") or "?", 0) + t
            by_kind[c.get("kind") or "?"] = by_kind.get(c.get("kind") or "?", 0) + t
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
                "n_calls": len(self.llm_calls), "by_node": by_node, "by_kind": by_kind}

    def to_dict(self) -> dict:
        lat = dict(self.latency_ms)
        lat["total"] = round((time.perf_counter() - self._t0) * 1000, 1)
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "created_at": self.created_at,
            "query_plan": self.query_plan,
            "retrieval": self.retrieval,
            "rank_features": self.rank_features,
            "final_ranked_jobs": self.final_ranked_jobs,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "token_usage": self._token_usage(),
            "latency_ms": lat,
            "model_versions": self.model_versions,
            "index_version": self.index_version,
            "errors": self.errors,
            "user_feedback": self.user_feedback,
        }


# ---------------------------------------------------------------------------
# 生命周期：开启 / 获取 / 结束
# ---------------------------------------------------------------------------
def current() -> Optional[RequestTrace]:
    return _current.get()


def start_request(request_id: str, session_id: str, user_query: str,
                  model_versions: Optional[dict] = None,
                  index_version: Optional[dict] = None) -> RequestTrace:
    tr = RequestTrace(request_id, session_id, user_query, model_versions, index_version)
    _current.set(tr)
    return tr


def finish_request() -> Optional[RequestTrace]:
    """落盘当前 trace 并清空 contextvar。返回该 trace（便于调用方取 request_id）。"""
    tr = _current.get()
    if tr is not None:
        _flush(tr)
        _current.set(None)
    return tr


@contextmanager
def request_scope(request_id: str, session_id: str, user_query: str,
                  model_versions: Optional[dict] = None, index_version: Optional[dict] = None):
    """run_turn 用：进入时开 trace，退出时落盘（即使异常也落盘）。"""
    tr = start_request(request_id, session_id, user_query, model_versions, index_version)
    try:
        yield tr
    finally:
        try:
            finish_request()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 节点包裹（每节点时延 + 设「当前节点」供 LLM 归属）
# ---------------------------------------------------------------------------
def run_node(name: str, fn, state):
    """在 build_graph 里包住每个节点：记录该节点耗时，并把 name 设为「当前节点」。

    无活跃 trace 时只透传执行（零开销旁路），故 pipeline / 直接 invoke 不受影响。
    """
    tr = _current.get()
    if tr is None:
        return fn(state)
    node_tok = _current_node.set(name)
    t0 = time.perf_counter()
    try:
        return fn(state)
    finally:
        try:
            tr.add_latency(name, (time.perf_counter() - t0) * 1000)
        except Exception:
            pass
        _current_node.reset(node_tok)


# ---------------------------------------------------------------------------
# 记录 API（均 no-op if 无活跃 trace）
# ---------------------------------------------------------------------------
def _cap(items: list) -> list:
    return list(items)[:_MAX_CANDIDATES]


def record_query_plan(plan: dict) -> None:
    tr = _current.get()
    if tr is not None and isinstance(plan, dict):
        tr.query_plan = plan


def record_constraint_filter(info: dict, allowed_count: Optional[int] = None) -> None:
    tr = _current.get()
    if tr is None:
        return
    tr.retrieval["constraint_filter"] = info or {}
    if allowed_count is not None:
        tr.retrieval["allowed_count"] = allowed_count


def record_retrieval_queries(queries: dict) -> None:
    tr = _current.get()
    if tr is not None and isinstance(queries, dict):
        tr.retrieval["queries"] = {k: v for k, v in queries.items() if v}


def _hit_score(hit: dict) -> float:
    for k in ("retrieval_score", "bm25_score", "vector_score", "rrf_score", "score"):
        v = hit.get(k)
        if isinstance(v, (int, float)):
            return round(float(v), 6)
    return 0.0


def record_channel_hits(channel: str, hits: list) -> None:
    """累积某通道（dense / bm25）的命中（多路 Query 共用同一通道时按 job_id 取最优分、最小名次）。"""
    tr = _current.get()
    if tr is None or not hits:
        return
    key = f"{channel}_candidates"
    bucket = {row["job_id"]: row for row in tr.retrieval.get(key, [])}
    for rank, h in enumerate(hits):
        jid = h.get("job_id")
        if not jid:
            continue
        score = _hit_score(h)
        cur = bucket.get(jid)
        if cur is None or score > cur.get("score", -1):
            bucket[jid] = {"job_id": jid, "score": score, "rank": rank}
        elif rank < cur.get("rank", 1 << 30):
            cur["rank"] = rank
    tr.retrieval[key] = _cap(sorted(bucket.values(), key=lambda r: r["score"], reverse=True))


def record_rrf(merged: list) -> None:
    tr = _current.get()
    if tr is None or not merged:
        return
    tr.retrieval["rrf_candidates"] = _cap([
        {"job_id": h.get("job_id"), "rrf_score": round(float(h.get("retrieval_score") or 0.0), 6), "rank": i}
        for i, h in enumerate(merged) if h.get("job_id")
    ])


def record_rerank(candidates: list) -> None:
    tr = _current.get()
    if tr is None or not candidates:
        return
    tr.retrieval["rerank"] = _cap([
        {"job_id": h.get("job_id"), "rerank_score": round(float(h.get("rerank_score") or 0.0), 6), "rank": i}
        for i, h in enumerate(candidates) if h.get("job_id") and "rerank_score" in h
    ])


def record_rank_features(features: list) -> None:
    tr = _current.get()
    if tr is not None and features:
        tr.rank_features = _cap(list(features))


def record_final_ranked(jobs: list) -> None:
    tr = _current.get()
    if tr is not None:
        tr.final_ranked_jobs = _cap(list(jobs or []))


def record_tool_call(name: str, latency_ms: float, ok: bool = True, summary: Optional[dict] = None) -> None:
    tr = _current.get()
    if tr is not None:
        tr.tool_calls.append({"tool": name, "latency_ms": round(latency_ms, 1),
                              "ok": ok, "summary": summary or {}})


def record_llm_call(kind: str, model: str, usage: Any, latency_ms: float) -> None:
    """core/llm.py 调用：记一次 LLM/embedding 调用的 token + 时延，归属到「当前节点」。"""
    tr = _current.get()
    if tr is None:
        return
    pt = ct = tt = None
    if usage is not None:
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        if isinstance(usage, dict):  # 兼容 dict 形态
            pt, ct, tt = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
    try:
        tr.add_llm_call({
            "node": _current_node.get(), "kind": kind, "model": model,
            "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt,
            "latency_ms": round(latency_ms, 1),
        })
    except Exception:
        pass


def record_error(node: str, message: str) -> None:
    tr = _current.get()
    if tr is not None:
        tr.errors.append({"node": node, "message": str(message)[:500], "at": _utc_now_iso()})


def record_state_errors(errors: list) -> None:
    """run_turn 结束前把 final_state 的 errors 一次性快照进 trace（去重补充）。"""
    tr = _current.get()
    if tr is None or not errors:
        return
    seen = {e.get("message") for e in tr.errors}
    for msg in errors:
        if msg not in seen:
            tr.errors.append({"node": "state", "message": str(msg)[:500], "at": _utc_now_iso()})


# ---------------------------------------------------------------------------
# 版本快照（请求开始时取一次）
# ---------------------------------------------------------------------------
def snapshot_model_versions() -> dict:
    from resume2job.core import config
    return {
        "chat_model": config.CHAT_MODEL,
        "planner_model": config.PLANNER_MODEL,
        "score_model": config.SCORE_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "use_rerank": config.USE_RERANK,
    }


def snapshot_index_version() -> dict:
    from resume2job.ingest import versions
    out = {
        "embedding_version": versions.embedding_version(),
        "index_text_version": versions.INDEX_TEXT_VERSION,
        "parser_version": versions.PARSER_VERSION,
    }
    try:
        from resume2job.storage import jobs_store
        out["corpus_signature"] = jobs_store.corpus_signature()
        out["job_count"] = jobs_store.count()
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS request_traces (
    request_id      TEXT PRIMARY KEY,
    session_id      TEXT,
    created_at      TEXT,
    user_query      TEXT,
    intent          TEXT,
    session_action  TEXT,
    n_final         INTEGER,
    total_latency_ms REAL,
    total_tokens    INTEGER,
    n_llm_calls     INTEGER,
    n_tool_calls    INTEGER,
    n_errors        INTEGER,
    user_feedback   TEXT
);
"""


def _flush(tr: RequestTrace) -> None:
    """完整 trace → JSONL；摘要 → SQLite。失败静默。"""
    data = tr.to_dict()
    # 1) JSONL（完整事实）
    try:
        os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[observability] 警告：写 request_traces.jsonl 失败：{e}")
    # 2) SQLite 摘要（可查询 + 承载 user_feedback 回填）
    try:
        qp = data.get("query_plan") or {}
        tu = data.get("token_usage") or {}
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_CREATE_SQL)
            conn.execute(
                "INSERT OR REPLACE INTO request_traces "
                "(request_id, session_id, created_at, user_query, intent, session_action, "
                " n_final, total_latency_ms, total_tokens, n_llm_calls, n_tool_calls, n_errors, user_feedback) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (tr.request_id, tr.session_id, tr.created_at, tr.user_query,
                 qp.get("intent"), qp.get("session_action"),
                 len(data.get("final_ranked_jobs") or []),
                 (data.get("latency_ms") or {}).get("total"),
                 tu.get("total_tokens"), tu.get("n_calls"),
                 len(data.get("tool_calls") or []), len(data.get("errors") or []),
                 tr.user_feedback),
            )
            conn.commit()
    except Exception as e:
        print(f"[observability] 警告：写 request_traces 表失败：{e}")


def get_feedback(request_id: str) -> Optional[str]:
    """读 SQLite 摘要表里**事后回填**的 user_feedback（JSONL trace 落盘时该字段尚为 null）。

    ranking.dataset 用它把反馈合并回 trace 作弱标签——feedback 产生在请求之后、权威在 SQLite。
    """
    if not request_id:
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT user_feedback FROM request_traces WHERE request_id=?", (request_id,)
            ).fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def record_feedback(request_id: str, feedback: Optional[str]) -> bool:
    """事后回填用户反馈（saved / skipped / not_interested / applied …）到 SQLite 摘要表。

    user_feedback 是后训练 / 排序学习的弱标签来源；产生在请求之后，故单独回填（JSONL 不可变，
    反馈以 SQLite 为权威，按 request_id 关联）。返回是否更新成功。
    """
    if not request_id:
        return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_CREATE_SQL)
            cur = conn.execute("UPDATE request_traces SET user_feedback=? WHERE request_id=?",
                               (feedback, request_id))
            conn.commit()
            return (cur.rowcount or 0) > 0
    except Exception as e:
        print(f"[observability] 警告：回填 feedback 失败：{e}")
        return False


# ---------------------------------------------------------------------------
# 读取（replay / 审计用）
# ---------------------------------------------------------------------------
def load_traces(path: Optional[str] = None) -> List[dict]:
    """读全部 trace（JSONL）。"""
    p = path or JSONL_PATH
    out: List[dict] = []
    if not os.path.exists(p):
        return out
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"[observability] 警告：读 request_traces.jsonl 失败：{e}")
    return out


def get_trace(request_id: str, path: Optional[str] = None) -> Optional[dict]:
    """按 request_id 取最近一条 trace（同 id 多条时取最后写入的）。"""
    found = None
    for t in load_traces(path):
        if t.get("request_id") == request_id:
            found = t
    return found
