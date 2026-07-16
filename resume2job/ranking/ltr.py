# -*- coding: utf-8 -*-
"""ranking/ltr.py —— LambdaMART 粗排（**接口占位，第一阶段不实现训练逻辑**）。

明确边界（按用户要求）：第一阶段只做「特征抽取（features.py）→ 训练数据导出（dataset.py）→ 基线排序对比」，
**不生成 LambdaMART / XGBoost 的训练与推理实现**。本文件只钉住两件事：
    1. LambdaMART 在检索-排序链路里的**位置**与**契约**（输入 = features.FEATURE_NAMES 向量，输出 = 排序分）；
    2. 一个**非 ML 基线排序**（按现有 rank_score）——供后续 ranking_eval 对比「LambdaMART 是否真的更好」。

链路中的位置（job_matching_and_ranking 规范）：
    BM25 + dense 召回 → RRF 融合 → **LambdaMART 粗排（本模块，待实现）** → qwen3-rerank 深度精排 → Top-K LLM 解释/报告

为什么需要它（待实现时的目标）：RRF 只按名次融合，学不会「城市严格匹配比 BM25 高一点更重要 / 技能缺口过多
应强烈降权 / rerank 高但学历不满足应过滤 / 项目证据比技能栏声明更可信」这类**特征间非线性权衡**；
LambdaMART 能从带 label 的数据（dataset.py 产出）里学到这些。实现时预期用 XGBoost 的 rank:ndcg 目标。
"""

from __future__ import annotations

from typing import List, Optional

from resume2job.ranking.features import FEATURE_NAMES  # 训练/推理的输入特征契约（顺序即向量维度）

# 模型在链路中的阶段标识（observability / eval 标注用）
STAGE_NAME = "ltr_coarse_rank"


class LambdaMARTRanker:
    """LambdaMART 粗排器**接口占位**。第一阶段不实现——train/predict/save/load 均抛 NotImplementedError。

    设计契约（供后续实现时遵循，勿在第一阶段写实现）：
        - 输入特征 = features.FEATURE_NAMES 的有序数值向量（dataset.export_svmlight 已是其训练格式）；
        - 训练目标 = listwise（rank:ndcg），按 group_id(qid) 分组、relevance_label 为标签；
        - 训练时按 label_source 赋样本权重（人工 > 用户行为 > LLM 弱标签 > 规则）；
        - predict 输出每个候选的排序分，用于在 RRF 之后、rerank 之前做粗排截断。
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.feature_names = list(FEATURE_NAMES)
        self._model = None  # 第一阶段不加载/不训练

    def train(self, svmlight_path: str, **kwargs):
        raise NotImplementedError(
            "LambdaMART 训练为后续阶段内容；第一阶段只用 ranking.dataset 导出训练数据。"
            "实现时按 group(qid) + relevance_label + label_source 权重训练（预期 XGBoost rank:ndcg）。")

    def predict(self, feature_vectors: List[List[float]]) -> List[float]:
        raise NotImplementedError("LambdaMART 推理为后续阶段内容（第一阶段未训练模型）。")

    def save(self, path: str):
        raise NotImplementedError("第一阶段无模型可保存。")

    def load(self, path: str):
        raise NotImplementedError("第一阶段无模型可加载。")


def baseline_ranking(feature_rows: List[dict]) -> List[str]:
    """非 ML 基线排序：按现有 rank_score 降序返回 job_id 列表（缺失分排最后）。

    这是当前生产系统的排序口径（评分层 rank_score），作为 LambdaMART 上线后的对照基线——
    后续 ranking_eval 比较「基线 vs LambdaMART」的 nDCG/MRR，判断学习排序是否带来真实提升。
    本函数纯 Python、无模型，第一阶段即可用。
    """
    def _key(r: dict):
        v = r.get("rank_score")
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    return [r.get("job_id") for r in sorted(feature_rows, key=_key, reverse=True) if r.get("job_id")]
