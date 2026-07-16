# -*- coding: utf-8 -*-
"""resume2job.ranking —— 排序特征与 Learning-to-Rank 数据准备（Stage 3）。

把召回 / 评分各阶段分散的信号汇聚成标准排序特征表，并构建 LtR 训练数据，为 LambdaMART / 双塔等
排序学习铺路。**第一阶段只做特征抽取 + 训练数据导出 + 基线对比，不训练模型**（ltr.py 为接口占位）。

    - features.py：统一排序特征（FEATURE_NAMES + build_features_from_trace），消费 Stage 2 trace + Stage 1 jobs 元数据；
    - dataset.py ：query-candidate-label 数据集（group_id + label_source），导出 JSONL + SVMlight；
    - ltr.py     ：LambdaMART 接口占位 + 链路位置 + 非 ML 基线排序（baseline_ranking）。
"""

from resume2job.ranking.features import (
    FEATURE_NAMES, build_features_from_trace, build_features_for_request, feature_vector,
)
from resume2job.ranking.dataset import (
    build_rows, build_dataset, dataset_stats, export_jsonl, export_svmlight,
    feedback_label_fn, make_relevant_set_label_fn, unlabeled_label_fn, relevant_map_from_eval_dataset,
)
from resume2job.ranking.ltr import LambdaMARTRanker, baseline_ranking, STAGE_NAME

__all__ = [
    "FEATURE_NAMES", "build_features_from_trace", "build_features_for_request", "feature_vector",
    "build_rows", "build_dataset", "dataset_stats", "export_jsonl", "export_svmlight",
    "feedback_label_fn", "make_relevant_set_label_fn", "unlabeled_label_fn", "relevant_map_from_eval_dataset",
    "LambdaMARTRanker", "baseline_ranking", "STAGE_NAME",
]
