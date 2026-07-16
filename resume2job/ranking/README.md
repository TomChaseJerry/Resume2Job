# resume2job/ranking —— 排序特征与 Learning-to-Rank 数据准备（Stage 3）

把召回 / 评分各阶段分散的信号汇聚成一张标准**排序特征表**，并构建 **LtR 训练数据**（query-candidate-label），
为 LambdaMART / 双塔等排序学习铺路。

**边界（按要求）**：第一阶段只做「特征抽取 → 训练数据导出 → 基线对比」，**不实现 LambdaMART 训练 / 推理逻辑**（`ltr.py` 是接口占位）。

## 数据来源（职责分离）

- **Stage 2 request trace**（`observability`）= 原始事实：`retrieval` 各通道候选+分（bm25/dense/rrf/rerank）、
  `rank_features`（评分明细，含 skill_score 字典里的 matched/missing/preferred）、`query_plan`（硬约束）；
- **Stage 1 jobs 表** = jd 质量分 / 新鲜度（collected_at）/ 城市 / preferred 分母；
- **user_feedback**（SQLite）= 弱标签来源（事后回填，`dataset` 合并回 trace）。

本模块只做特征工程（join + 派生），**不重新跑检索/评分、不调 LLM**。

## 文件清单

| 文件 | 作用 | 关键符号 |
|---|---|---|
| `features.py` | 统一排序特征：一行 = 一个 (request_id, job_id) | `FEATURE_NAMES`、`build_features_from_trace`、`build_features_for_request`、`feature_vector` |
| `dataset.py` | LtR 数据集：group + label + label_source，导出 JSONL/SVMlight | `build_rows`、`build_dataset`、`feedback_label_fn`、`make_relevant_set_label_fn`、`export_svmlight`、`dataset_stats` |
| `ltr.py` | LambdaMART **接口占位** + 非 ML 基线 | `LambdaMARTRanker`(train/predict 抛 NotImplementedError)、`baseline_ranking` |

## 特征（FEATURE_NAMES，18 维，顺序即向量维度）

```
bm25_score dense_score rrf_score rerank_score              # 检索通道分（Stage 2）
skill_score project_score match_score direction_bonus commute_bonus rank_score   # 评分层信号
required_skill_coverage preferred_skill_coverage missing_required_skill_count    # 派生：技能覆盖
direction_match city_match education_match                 # 派生：硬/软约束匹配
jd_quality_score job_freshness_days                        # 岗位侧元数据（Stage 1）
```
缺通道分（如未 rerank）→ feature_vector 填 0.0。

## 标签与来源（label_source）

人工标注 ≠ 用户行为 ≠ LLM 弱标签，训练时应按来源赋权。可插拔 labeler：
- `feedback_label_fn`（默认）：用 request 级 `user_feedback`（saved/applied→2、not_interested→0）。**弱信号**：
  请求级、同组所有候选同一档、**组内无对比度**（LtR 学不到东西），仅占位，待**逐岗位**反馈再增强；
- `make_relevant_set_label_fn(relevant_ids)`：候选 ∈ 相关集→1 否则 0，**组内有对比度可训练**；相关集可取
  `eval/retrieval_dataset.jsonl` 的 LLM pooling（`weak_llm_label`）或人工标注（`human_annotated`）；
- `unlabeled_label_fn`：只导特征不打标。

> 强标签（组内有正有负）来自相关集 / 人工标注路径。要用 eval gold 打标签，需先用那些 eval query 跑出对应
> trace（query 命名空间不同）——这属 Stage 4 ranking 评测口径。

## 怎么运行 / 在哪看

```bash
python scripts/verify_ranking.py          # 验收（无 API）：合成单测 + 真实 trace 抽特征 + 导出数据集
python scripts/verify_ranking.py --no-export   # 不写 ranking/data/

# 代码里：
from resume2job.ranking import build_features_for_request, build_dataset, make_relevant_set_label_fn
build_dataset()                                # 全部 trace → ranking/data/ltr_dataset.{jsonl,svmlight}
```

产物：`ranking/data/ltr_dataset.jsonl`（特征+标签+来源+时间戳）+ `ltr_dataset.svmlight`（`<label> qid:<group> i:val` 训练格式）。

## LambdaMART 的位置（待实现）

```
BM25 + dense 召回 → RRF → [LambdaMART 粗排（ltr.py 占位）] → qwen3-rerank 精排 → Top-K LLM 报告
```
为什么需要：RRF 只按名次融合，学不会「城市严格匹配 > BM25 略高 / 技能缺口过多强降权 / 项目证据 > 技能栏声明」
这类特征间非线性权衡——这交给 LambdaMART 从带标签数据里学。`baseline_ranking`（按现有 rank_score）是其上线后的对照基线。
