# -*- coding: utf-8 -*-
"""ranking/dataset.py —— Learning-to-Rank 数据集构建（query-candidate-label，带 group 与 label_source）。

LambdaMART 不是普通分类：它按「一个 query 对应一组候选」组织数据（group/qid），学组内相对排序。
本模块把 Stage 2 的 request trace → features.py 的特征行 → 加上**相关性标签**与**标签来源**，
导出标准 LtR 格式（JSONL + SVMlight），供后续训练 / 评测。**第一阶段只准备数据，不训练模型**。

label_source 很重要（用户规范）：人工标注的 3 分 ≠ 用户点击的 3 分 ≠ LLM 弱标签的 3 分；记录来源，
训练时可按来源赋不同权重，避免弱标签污染。本模块支持可插拔 labeler：
    - feedback_label_fn：用 request 级 user_feedback（saved/applied/skipped…）打标签——**弱信号**，
      请求级（同一请求所有候选同一档，组内无对比度），仅在缺更好标签时用，已如实标 label_source；
    - make_relevant_set_label_fn(relevant_ids)：候选 job_id ∈ 给定相关集 → 1 否则 0（组内有对比度），
      相关集可来自 eval/retrieval_dataset.jsonl 的 LLM pooling（weak_llm_label）或人工标注（human_annotated）；
    - unlabeled_label_fn：只导特征不打标签（推理 / 特征分析）。

真正能训练的强标签（组内有正有负）来自「相关集」路径或人工标注；feedback 路径是占位，待积累
**逐岗位**反馈后再增强（当前 trace 的 feedback 是请求级）。
"""

from __future__ import annotations

import os
import json
from typing import Callable, Dict, List, Optional, Tuple

from resume2job.ranking.features import build_features_from_trace, feature_vector, FEATURE_NAMES

# ---- label_source 取值（用户规范）----
HUMAN_ANNOTATED = "human_annotated"
USER_SAVED = "user_saved"
USER_APPLIED = "user_applied"
USER_NOT_INTERESTED = "user_not_interested"
WEAK_LLM_LABEL = "weak_llm_label"
RULE_LABEL = "rule_label"
UNLABELED = "unlabeled"

# request 级 feedback → (相关性档, label_source)
_FEEDBACK_LABEL = {
    "applied": (2, USER_APPLIED),
    "saved": (2, USER_SAVED),
    "clicked": (1, USER_SAVED),
    "not_interested": (0, USER_NOT_INTERESTED),
    "skipped": (0, USER_NOT_INTERESTED),
}

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL_PATH = os.path.join(_DATA_DIR, "ltr_dataset.jsonl")
SVMLIGHT_PATH = os.path.join(_DATA_DIR, "ltr_dataset.svmlight")

# labeler 签名：(trace, feature_row) -> (relevance_label 或 None, label_source)
LabelFn = Callable[[dict, dict], Tuple[Optional[int], str]]


def feedback_label_fn(trace: dict, row: dict) -> Tuple[Optional[int], str]:
    """请求级 feedback 打标签（弱信号，组内无对比度）。无 feedback → 不打标。"""
    fb = (trace or {}).get("user_feedback")
    if fb in _FEEDBACK_LABEL:
        return _FEEDBACK_LABEL[fb]
    return None, UNLABELED


def make_relevant_set_label_fn(relevant_ids, source: str = WEAK_LLM_LABEL) -> LabelFn:
    """工厂：候选 job_id ∈ relevant_ids → 1，否则 0（组内有对比度，可训练）。source 标注来源。"""
    rel = set(relevant_ids or [])

    def _fn(trace: dict, row: dict) -> Tuple[Optional[int], str]:
        return (1 if row.get("job_id") in rel else 0), source

    return _fn


def unlabeled_label_fn(trace: dict, row: dict) -> Tuple[Optional[int], str]:
    return None, UNLABELED


def relevant_map_from_eval_dataset(path: Optional[str] = None) -> Dict[str, list]:
    """从 eval/retrieval_dataset.jsonl 读 {query_id: relevant_job_ids}（供 gold labeler 用）。

    注意：eval 的 query_id 是合成的（job_id__style），与 request trace 的 query_id(=request_id) **不同命名空间**；
    要用它打标签，需先用这些 eval query 跑出对应 trace（属 Stage 4 ranking 评测口径），本函数只提供读取。
    """
    from resume2job.eval.build_dataset import load_dataset
    out: Dict[str, list] = {}
    for s in load_dataset(path) if path else load_dataset():
        qid = s.get("query_id")
        if qid:
            out[qid] = s.get("relevant_job_ids") or []
    return out


def build_rows(traces: Optional[List[dict]] = None, label_fn: LabelFn = feedback_label_fn) -> List[dict]:
    """把若干 trace → LtR 行列表。traces 缺省读全部 request trace。

    每行：{query_id, group_id, job_id, timestamp, relevance_label, label_source, features{}, feature_vector[]}。
    """
    if traces is None:
        from resume2job.observability import events
        traces = events.load_traces()
        # user_feedback 事后回填在 SQLite（JSONL 落盘时为 null），合并回 trace 供 feedback labeler 用
        for tr in traces:
            if tr.get("user_feedback") is None:
                fb = events.get_feedback(tr.get("request_id"))
                if fb:
                    tr["user_feedback"] = fb
    rows: List[dict] = []
    for tr in traces or []:
        feats = build_features_from_trace(tr)
        ts = tr.get("created_at")
        for fr in feats:
            label, source = label_fn(tr, fr)
            rows.append({
                "query_id": fr.get("query_id"),
                "group_id": fr.get("group_id"),
                "job_id": fr.get("job_id"),
                "timestamp": ts,
                "relevance_label": label,
                "label_source": source,
                "company": fr.get("company"),
                "title": fr.get("title"),
                "features": {k: fr.get(k) for k in FEATURE_NAMES},
                "feature_vector": feature_vector(fr),
            })
    return rows


def dataset_stats(rows: List[dict]) -> dict:
    """统计：行数 / 组数 / 已标注数 / 标签分布 / 来源分布。"""
    import collections
    groups = {r.get("group_id") for r in rows}
    labeled = [r for r in rows if r.get("relevance_label") is not None]
    return {
        "n_rows": len(rows),
        "n_groups": len(groups),
        "n_labeled": len(labeled),
        "label_dist": dict(collections.Counter(r["relevance_label"] for r in labeled)),
        "label_source_dist": dict(collections.Counter(r.get("label_source") for r in rows)),
        "n_features": len(FEATURE_NAMES),
    }


def export_jsonl(rows: List[dict], path: str = JSONL_PATH) -> int:
    """导出全部行为 JSONL（含特征 + 标签 + 来源 + 时间戳）。返回行数。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def export_svmlight(rows: List[dict], path: str = SVMLIGHT_PATH) -> int:
    """导出 SVMlight/LtR 文本格式：`<label> qid:<gid> 1:<f1> 2:<f2> ...`（每组按 group 连续）。

    仅导**已标注**行（训练需标签）；group_id 字符串映射为稳定递增整数 qid。返回导出的行数。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labeled = [r for r in rows if r.get("relevance_label") is not None]
    labeled.sort(key=lambda r: str(r.get("group_id")))   # 同组连续（LtR 要求）
    qid_map: Dict[str, int] = {}
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in labeled:
            gid = str(r.get("group_id"))
            qid = qid_map.setdefault(gid, len(qid_map) + 1)
            vec = r.get("feature_vector") or feature_vector(r.get("features") or {})
            feats = " ".join(f"{i + 1}:{v:g}" for i, v in enumerate(vec))
            f.write(f"{int(r['relevance_label'])} qid:{qid} {feats}\n")
            n += 1
    return n


def build_dataset(label_fn: LabelFn = feedback_label_fn,
                  traces: Optional[List[dict]] = None,
                  jsonl_path: str = JSONL_PATH, svmlight_path: str = SVMLIGHT_PATH) -> dict:
    """端到端：trace → 特征 + 标签 → 导出 JSONL + SVMlight，返回统计。"""
    rows = build_rows(traces=traces, label_fn=label_fn)
    n_jsonl = export_jsonl(rows, jsonl_path)
    n_svm = export_svmlight(rows, svmlight_path)
    stats = dataset_stats(rows)
    stats["exported_jsonl"] = n_jsonl
    stats["exported_svmlight"] = n_svm
    stats["jsonl_path"] = jsonl_path
    stats["svmlight_path"] = svmlight_path
    print(f"[ranking.dataset] 导出 {n_jsonl} 行（{stats['n_groups']} 组，已标注 {stats['n_labeled']}）→ {jsonl_path}")
    print(f"[ranking.dataset] SVMlight 训练格式 {n_svm} 行 → {svmlight_path}")
    print(f"[ranking.dataset] 标签来源分布：{stats['label_source_dist']}")
    return stats
