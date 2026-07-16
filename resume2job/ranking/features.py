# -*- coding: utf-8 -*-
"""ranking/features.py —— 统一排序特征表（每个「请求 × 候选岗位」一行特征）。

现状：召回/评分各阶段的信号（BM25 / dense / RRF / rerank 分、技能覆盖、方向/通勤加分、学历门槛、
岗位质量、新鲜度）分散在检索层、评分层、ingest 层。本模块把它们**汇聚成一张标准表格**，是
LambdaMART / 双塔等排序学习的输入，也是 ranking_eval / fairness_audit 的分析底座。

数据来源（职责分离）：
    - **Stage 2 的 request trace**（observability）是原始事实：retrieval 各通道候选+分、rank_features
      （评分明细，含 skill_score 字典里的 matched/missing/preferred）、query_plan（硬约束）；
    - **jobs 表**（Stage 1）补 jd_quality_score / 新鲜度（collected_at）/ 城市 / preferred_skills 分母；
    - 本模块只做**特征工程**（join + 派生覆盖率/匹配标志），不重新跑检索/评分、不调 LLM。

一行特征 = 一个 (request_id, job_id) 对。FEATURE_NAMES 是喂给排序模型的有序数值向量 schema。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional

from resume2job.storage import jobs_store

# 排序模型输入的有序数值特征（feature_vector 按此顺序）。新增特征往后追加，勿改既有顺序。
FEATURE_NAMES: List[str] = [
    # 检索通道分（Stage 2 trace.retrieval）
    "bm25_score", "dense_score", "rrf_score", "rerank_score",
    # 评分层信号（Stage 2 trace.rank_features）
    "skill_score", "project_score", "match_score", "direction_bonus", "commute_bonus", "rank_score",
    # 派生：技能覆盖
    "required_skill_coverage", "preferred_skill_coverage", "missing_required_skill_count",
    # 派生：硬/软约束匹配标志
    "direction_match", "city_match", "education_match",
    # 岗位侧元数据（Stage 1 jobs 表）
    "jd_quality_score", "job_freshness_days",
]

# education_gate → 数值（用于 education_match 特征）
_EDU_GATE_VALUE = {"satisfied": 1.0, "indeterminate": 0.5, "insufficient": 0.0}


def _iso_days_between(later_iso: Optional[str], earlier_iso: Optional[str]) -> Optional[float]:
    """later - earlier 的天数；任一缺失 / 不可解析返回 None。"""
    if not later_iso or not earlier_iso:
        return None
    try:
        a = datetime.fromisoformat(str(later_iso))
        b = datetime.fromisoformat(str(earlier_iso))
        return round((a - b).total_seconds() / 86400.0, 2)
    except Exception:
        return None


def _channel_score_map(trace: dict, channel_key: str, score_key: str) -> Dict[str, float]:
    """trace.retrieval[channel_key] → {job_id: score}。"""
    out: Dict[str, float] = {}
    for row in (trace.get("retrieval") or {}).get(channel_key) or []:
        jid = row.get("job_id")
        if jid is not None:
            v = row.get(score_key)
            if isinstance(v, (int, float)):
                out[jid] = float(v)
    return out


def _skill_dict(rf: dict) -> dict:
    """rank_features 行里的 skill_score 可能是完整技能字典（含 matched/missing/preferred）或标量。"""
    s = rf.get("skill_score")
    return s if isinstance(s, dict) else {}


def _scalar(v):
    """从「可能是字典(.score)或标量」里取数值。"""
    if isinstance(v, dict):
        v = v.get("score")
    return float(v) if isinstance(v, (int, float)) else None


def build_features_from_trace(trace: dict) -> List[dict]:
    """把一条 request trace 装配成若干「(request_id, job_id) 特征行」。

    候选取自 trace.rank_features（已评分的候选池）；逐行 join 检索通道分 + jobs 表元数据 + 派生特征。
    返回 [{request_id, query_id, group_id, job_id, company, title, match_level, label_hint, <FEATURE_NAMES...>}]。
    """
    if not isinstance(trace, dict):
        return []
    rid = trace.get("request_id")
    qp = trace.get("query_plan") or {}
    want_city = ((qp.get("hard_constraints") or {}).get("city") or None)

    rank_features = trace.get("rank_features") or []
    if not rank_features:
        return []

    # 检索各通道分（按 job_id）
    dense = _channel_score_map(trace, "dense_candidates", "score")
    bm25 = _channel_score_map(trace, "bm25_candidates", "score")
    rrf = _channel_score_map(trace, "rrf_candidates", "rrf_score")
    rerank = _channel_score_map(trace, "rerank", "rerank_score")

    # jobs 表元数据（质量分 / 新鲜度 / 城市 / preferred 分母）批量取
    job_ids = [rf.get("job_id") for rf in rank_features if rf.get("job_id")]
    job_rows = jobs_store.get_jobs_by_ids(job_ids)
    ref_time = trace.get("created_at")  # 以请求时刻为基准算新鲜度

    rows: List[dict] = []
    for rf in rank_features:
        jid = rf.get("job_id")
        if not jid:
            continue
        jrow = job_rows.get(jid) or {}
        jd = jrow.get("jd_profile") or {}
        sk = _skill_dict(rf)

        matched = sk.get("matched_skills") or []
        missing = sk.get("missing_skills") or []
        preferred_matched = sk.get("preferred_matched_skills") or []
        n_required = len(matched) + len(missing)          # 核心必备 ≈ 命中 + 缺口
        jd_pref = [s for s in (jd.get("preferred_skills") or []) if isinstance(s, str)]

        required_cov = (len(matched) / n_required) if n_required else None
        preferred_cov = (len(preferred_matched) / len(jd_pref)) if jd_pref else None

        # 城市匹配：无城市约束 → 视为满足(1.0)；有约束 → 候选城市是否含之
        # jobs_store 行里 cities_json 是 JSON 字符串（_row_to_dict 只反序列化 jd_profile），按需解析。
        cities = jrow.get("cities_json") or []
        if isinstance(cities, str):
            try:
                cities = json.loads(cities)
            except Exception:
                cities = []
        if not want_city:
            city_match = 1.0
        else:
            city_match = 1.0 if want_city in (cities or []) else 0.0

        edu_gate = rf.get("education_gate")
        education_match = _EDU_GATE_VALUE.get(edu_gate, 0.5) if isinstance(edu_gate, str) else 0.5
        direction_bonus = rf.get("direction_bonus")
        direction_match = 1.0 if (isinstance(direction_bonus, (int, float)) and direction_bonus > 0) else 0.0

        freshness = _iso_days_between(ref_time, jrow.get("collected_at") or jrow.get("created_at"))
        qscore = jrow.get("quality_score")

        feat = {
            "bm25_score": bm25.get(jid),
            "dense_score": dense.get(jid),
            "rrf_score": rrf.get(jid),
            "rerank_score": rerank.get(jid),
            "skill_score": _scalar(rf.get("skill_score")),
            "project_score": _scalar(rf.get("project_score")),
            "match_score": _scalar(rf.get("match_score")),
            "direction_bonus": float(direction_bonus) if isinstance(direction_bonus, (int, float)) else None,
            "commute_bonus": float(rf["commute_bonus"]) if isinstance(rf.get("commute_bonus"), (int, float)) else None,
            "rank_score": _scalar(rf.get("rank_score")),
            "required_skill_coverage": round(required_cov, 4) if required_cov is not None else None,
            "preferred_skill_coverage": round(preferred_cov, 4) if preferred_cov is not None else None,
            "missing_required_skill_count": len(missing),
            "direction_match": direction_match,
            "city_match": city_match,
            "education_match": education_match,
            "jd_quality_score": float(qscore) if isinstance(qscore, (int, float)) else None,
            "job_freshness_days": freshness,
        }
        row = {
            "request_id": rid,
            "query_id": rid,           # 一次请求 = 一个 query 组
            "group_id": rid,
            "job_id": jid,
            "company": rf.get("company") or jrow.get("company"),
            "title": rf.get("title") or jrow.get("title"),
            "match_level": rf.get("match_level"),
            **feat,
        }
        rows.append(row)
    return rows


def feature_vector(row: dict, fill: float = 0.0) -> List[float]:
    """按 FEATURE_NAMES 顺序取出数值向量；None → fill（默认 0.0，缺通道分等价于「未命中该通道」）。"""
    out = []
    for name in FEATURE_NAMES:
        v = row.get(name)
        out.append(float(v) if isinstance(v, (int, float)) else float(fill))
    return out


def build_features_for_request(request_id: str) -> List[dict]:
    """便捷入口：按 request_id 从 trace 存储取 trace 并装配特征行。"""
    from resume2job.observability import events
    tr = events.get_trace(request_id)
    return build_features_from_trace(tr) if tr else []
