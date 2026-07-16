# -*- coding: utf-8 -*-
"""
scripts/verify_ranking.py —— Stage 3（resume2job/ranking）排序特征与 LtR 数据准备的可复盘验收脚本。

**全程不调 API**（ranking 只读 Stage 2 trace + Stage 1 jobs 表做特征工程，不跑检索/评分）。
默认：离线单测（合成 trace）+ 读真实 trace 抽特征 + 导出 LtR 数据集到 ranking/data/。

用法：
    python scripts/verify_ranking.py                  # 全部（含读真实 trace 导出数据集）
    python scripts/verify_ranking.py --no-export      # 不写 ranking/data/（只做内存校验）
"""

import os
import sys
import argparse
import tempfile
import shutil

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def hr(t): print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)
def ok(m): print("  [OK] " + m)
def info(m): print("  - " + m)

_SYN_TRACE = {
    "request_id": "syn-verify", "created_at": "2026-06-23T06:00:00+00:00",
    "query_plan": {"intent": "RECOMMEND", "hard_constraints": {"city": "北京"}},
    "retrieval": {
        "dense_candidates": [{"job_id": "jd_test_1", "score": 0.8}, {"job_id": "jd_test_2", "score": 0.6}],
        "bm25_candidates": [{"job_id": "jd_test_1", "score": 0.9}],
        "rrf_candidates": [{"job_id": "jd_test_1", "rrf_score": 0.03}, {"job_id": "jd_test_2", "rrf_score": 0.02}],
        "rerank": [{"job_id": "jd_test_1", "rerank_score": 0.95}],
    },
    "rank_features": [
        {"job_id": "jd_test_1", "company": "字节", "title": "算法",
         "skill_score": {"score": 80, "matched_skills": ["深度学习", "RAG"], "missing_skills": ["强化学习"], "preferred_matched_skills": ["PyTorch"]},
         "project_score": {"score": 75}, "match_score": 78, "direction_bonus": 6, "commute_bonus": 0,
         "rank_score": 84, "match_level": "recommended", "education_gate": "satisfied"},
        {"job_id": "jd_test_2", "company": "腾讯", "title": "算法",
         "skill_score": {"score": 60, "matched_skills": ["深度学习"], "missing_skills": ["分布式"], "preferred_matched_skills": []},
         "project_score": {"score": 50}, "match_score": 56, "direction_bonus": 0, "commute_bonus": 0,
         "rank_score": 56, "match_level": "maybe", "education_gate": "indeterminate"},
    ],
    "user_feedback": "saved",
}


def section_imports():
    hr("1. 模块导入 & 计算图自检（无 API）—— 证明 ranking 接线无导入环")
    from resume2job.agent.graph import build_graph
    build_graph()
    from resume2job.ranking import FEATURE_NAMES  # noqa: F401
    ok(f"ranking(features/dataset/ltr) 可导入；LangGraph 可编译；排序特征维度 = {len(FEATURE_NAMES)}")


def section_features():
    hr("2. 统一排序特征表（features.py；无 API）—— 由 trace + jobs 表 join 出每候选一行特征")
    from resume2job.ranking import build_features_from_trace, feature_vector, FEATURE_NAMES
    rows = build_features_from_trace(_SYN_TRACE)
    r1 = next(r for r in rows if r["job_id"] == "jd_test_1")
    info(f"jd_test_1 特征：{{{', '.join(f'{k}={r1.get(k)}' for k in FEATURE_NAMES)}}}")
    assert r1["bm25_score"] == 0.9 and r1["rrf_score"] == 0.03 and r1["rerank_score"] == 0.95
    assert abs(r1["required_skill_coverage"] - 2 / 3) < 1e-3 and r1["missing_required_skill_count"] == 1
    assert r1["direction_match"] == 1.0 and r1["education_match"] == 1.0
    assert len(feature_vector(r1)) == len(FEATURE_NAMES)
    ok("检索通道分 + 技能覆盖 + 方向/学历匹配 + jd 质量/新鲜度 已 join；feature_vector 维度对齐")


def section_dataset():
    hr("3. LtR 数据集（dataset.py；无 API）—— group / label / label_source + JSONL/SVMlight 导出")
    from resume2job.ranking import (build_rows, dataset_stats, export_svmlight,
                                     feedback_label_fn, make_relevant_set_label_fn)
    fb = build_rows([_SYN_TRACE], feedback_label_fn)
    st = dataset_stats(fb)
    info(f"feedback 标签（请求级弱信号）stats：{st}")
    assert st["n_groups"] == 1 and st["label_source_dist"].get("user_saved") == 2

    gold = build_rows([_SYN_TRACE], make_relevant_set_label_fn({"jd_test_1"}))
    labels = {r["job_id"]: r["relevance_label"] for r in gold}
    info(f"相关集标签（组内有对比度，可训练）：{labels}，来源={gold[0]['label_source']}")
    assert labels == {"jd_test_1": 1, "jd_test_2": 0}

    tmp = tempfile.mkdtemp(prefix="ltr_v_")
    p = os.path.join(tmp, "d.svmlight")
    n = export_svmlight(gold, p)
    first = open(p, encoding="utf-8").readline().strip()
    info(f"SVMlight 首行：{first[:78]}")
    assert "qid:1" in first and n == 2
    shutil.rmtree(tmp, ignore_errors=True)
    ok("两种 labeler（feedback / 相关集）+ label_source 记录 + SVMlight 训练格式导出 均正常")


def section_ltr():
    hr("4. LambdaMART 接口占位 + 基线（ltr.py；无 API）")
    from resume2job.ranking import LambdaMARTRanker, baseline_ranking, build_features_from_trace, STAGE_NAME
    rows = build_features_from_trace(_SYN_TRACE)
    order = baseline_ranking(rows)
    info(f"非 ML 基线排序（按 rank_score）：{order}")
    assert order == ["jd_test_1", "jd_test_2"]
    raised = False
    try:
        LambdaMARTRanker().train("x.svmlight")
    except NotImplementedError:
        raised = True
    assert raised
    ok(f"baseline_ranking 可用（对照基线）；LambdaMARTRanker 训练/推理正确抛 NotImplementedError（阶段='{STAGE_NAME}' 占位，不实现）")


def section_real(export: bool):
    hr("5. 真实 trace → 特征 + 导出 LtR 数据集（读 Stage 2 trace；无 API）")
    from resume2job.observability import events
    from resume2job.ranking import build_features_from_trace, build_dataset, feedback_label_fn
    traces = events.load_traces()
    if not traces:
        info("库内暂无真实 trace —— 先跑一次 run_turn / scripts/verify_observability.py --with-api")
        return
    rows = build_features_from_trace(traces[-1])
    info(f"最近 trace({traces[-1].get('request_id')})：抽出 {len(rows)} 行特征")
    joined = sum(1 for r in rows if r.get("rrf_score") is not None and r.get("jd_quality_score") is not None)
    info(f"检索分 + jd 质量分 join 成功：{joined}/{len(rows)}（job_id 与检索/jobs 表对齐）")
    if rows:
        r = rows[0]
        info(f"样例 {r['job_id']}：bm25={r['bm25_score']} dense={r['dense_score']} rrf={r['rrf_score']} "
             f"rerank={r['rerank_score']} skill={r['skill_score']} rank={r['rank_score']} "
             f"quality={r['jd_quality_score']} freshness={r['job_freshness_days']}d")
    if export:
        stats = build_dataset(label_fn=feedback_label_fn)  # 全部真实 trace → ranking/data/
        ok(f"已导出 LtR 数据集：{stats['n_rows']} 行 / {stats['n_groups']} 组 → {stats['jsonl_path']}（+ .svmlight）")
        info(f"标签来源分布：{stats['label_source_dist']}（当前多为弱/未标注，待积累逐岗反馈或人工标注）")
    else:
        info("--no-export：跳过写 ranking/data/")


def main():
    ap = argparse.ArgumentParser(description="Stage 3 ranking 验收（全程无 API）")
    ap.add_argument("--no-export", action="store_true", help="不写 ranking/data/（只做内存校验）")
    args = ap.parse_args()
    section_imports()
    section_features()
    section_dataset()
    section_ltr()
    section_real(export=not args.no_export)
    hr("验收完成")
    print("  产物：ranking/data/ltr_dataset.jsonl（特征+标签+来源）+ ltr_dataset.svmlight（训练格式）")
    print("  说明：第一阶段只准备数据 + 基线对比；LambdaMART 训练为后续阶段（ltr.py 已占位接口）。")


if __name__ == "__main__":
    main()
