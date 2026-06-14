# -*- coding: utf-8 -*-
"""检索指标评测：四种配置横向对比 Recall@K / MRR / nDCG。

配置矩阵（同一评测集逐条 query 跑四遍）：
    vector          纯向量召回
    bm25            纯 BM25 召回
    hybrid          双通道 + RRF 融合
    hybrid+rerank   融合后再 gte-rerank 精排

指标定义（gold 为评测集的 relevant_job_ids，对 query 取宏平均）：
    Hit@1     ：首位是否命中任一 gold（命中 1，否则 0）；多 gold 下比 Recall@1 直观；
    Recall@5  ：Top-5 内命中的 gold 占全部 gold 的比例；
    MRR@10    ：第一个 gold 命中名次的倒数（1/rank），未命中记 0；
    nDCG@5    ：DCG@5 / IDCG@5，二值相关性（gold=1，其余=0）。

用法：
    python -m resume2job.eval.retrieval_eval [--top_k 5] [--modes vector bm25 hybrid hybrid+rerank]
"""

import os
import math
import json
import argparse
from datetime import datetime

from resume2job.retrieval.retriever import search_jobs
from resume2job.eval.build_dataset import load_dataset, DATASET_PATH

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# 配置名 → (mode, use_rerank)
CONFIGS = {
    "vector": ("vector", False),
    "bm25": ("bm25", False),
    "hybrid": ("hybrid", False),
    "hybrid+rerank": ("hybrid", True),
}


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def hit_at_k(ranked_ids: list, gold: set, k: int) -> float:
    """Hit@K：前 K 命中任一相关岗位即记 1，否则 0。

    多 gold 场景下比 Recall@1 更直观——Recall@1 上限是 1/|gold|（平均 4 个相关时
    天花板仅 0.25），而「首位是否命中」用命中率（0/1）表达更贴合「第一条准不准」。
    """
    if not gold:
        return 0.0
    return 1.0 if set(ranked_ids[:k]) & gold else 0.0


def recall_at_k(ranked_ids: list, gold: set, k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked_ids[:k]) & gold) / len(gold)


def mrr_at_k(ranked_ids: list, gold: set, k: int = 10) -> float:
    for rank, job_id in enumerate(ranked_ids[:k], start=1):
        if job_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list, gold: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, job_id in enumerate(ranked_ids[:k], start=1)
              if job_id in gold)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------
def evaluate_config(samples: list, mode: str, use_rerank: bool, top_k: int) -> dict:
    """对一个检索配置跑全量评测集，返回宏平均指标。"""
    metrics = {"hit@1": [], f"recall@{top_k}": [], "mrr@10": [],
               f"ndcg@{top_k}": []}
    for sample in samples:
        gold = set(sample["relevant_job_ids"])
        hits = search_jobs(sample["query"], top_k=max(top_k, 10),
                           mode=mode, use_rerank=use_rerank)
        ranked_ids = [h.get("job_id") for h in hits]
        metrics["hit@1"].append(hit_at_k(ranked_ids, gold, 1))
        metrics[f"recall@{top_k}"].append(recall_at_k(ranked_ids, gold, top_k))
        metrics["mrr@10"].append(mrr_at_k(ranked_ids, gold, 10))
        metrics[f"ndcg@{top_k}"].append(ndcg_at_k(ranked_ids, gold, top_k))

    n = max(1, len(samples))
    return {name: round(sum(vals) / n, 4) for name, vals in metrics.items()}


def run_retrieval_eval(top_k: int = 5, config_names: list = None,
                       dataset_path: str = DATASET_PATH) -> dict:
    """跑全部（或指定）检索配置的横向对比，打印表格并落盘报告。"""
    samples = load_dataset(dataset_path)
    if not samples:
        print(f"[ERROR] 评测集为空：{dataset_path}，请先运行 "
              "python -m resume2job.eval.build_dataset")
        return {}

    config_names = config_names or list(CONFIGS.keys())
    print(f"[retrieval_eval] 评测集 {len(samples)} 条，配置：{config_names}，top_k={top_k}")

    results = {}
    for name in config_names:
        if name not in CONFIGS:
            print(f"[WARN] 未知配置 {name}，跳过")
            continue
        mode, use_rerank = CONFIGS[name]
        print(f"\n[retrieval_eval] === 配置：{name} ===")
        results[name] = evaluate_config(samples, mode, use_rerank, top_k)
        print(f"[retrieval_eval] {name}: {results[name]}")

    _print_table(results)
    _save_report(results, len(samples), top_k)
    return results


def _print_table(results: dict) -> None:
    """终端打印对比表。"""
    if not results:
        return
    metric_names = list(next(iter(results.values())).keys())
    header = f"{'config':<16}" + "".join(f"{m:>12}" for m in metric_names)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(f"{name:<16}" + "".join(f"{metrics[m]:>12.4f}" for m in metric_names))
    print("=" * len(header))


def _save_report(results: dict, n_samples: int, top_k: int) -> str:
    """把评测结果保存为 Markdown 报告（含 JSON 原始数据）。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORT_DIR, f"retrieval_eval_{ts}.md")

    metric_names = list(next(iter(results.values())).keys()) if results else []
    lines = [
        f"# 检索评测报告 {ts}",
        "",
        f"- 评测集样本数：{n_samples}",
        f"- top_k：{top_k}",
        "",
        "| config | " + " | ".join(metric_names) + " |",
        "|---|" + "---|" * len(metric_names),
    ]
    for name, metrics in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{metrics[m]:.4f}" for m in metric_names) + " |")
    lines += ["", "```json", json.dumps(results, ensure_ascii=False, indent=2), "```", ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[retrieval_eval] 报告已保存：{path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检索指标评测（四配置横向对比）")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--modes", nargs="*", default=None,
                        help=f"评测配置子集，可选：{list(CONFIGS.keys())}")
    parser.add_argument("--dataset", default=DATASET_PATH)
    args = parser.parse_args()
    run_retrieval_eval(top_k=args.top_k, config_names=args.modes,
                       dataset_path=args.dataset)
