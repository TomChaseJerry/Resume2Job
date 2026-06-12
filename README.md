# Resume2Job —— 基于 LangGraph 的 Agentic RAG 实习岗位智能推荐系统

输入一份简历 PDF（和/或一条岗位 JD），系统自动完成：简历结构化 → 岗位混合检索 →
多维匹配评分 → 技能差距分析 → 推荐报告，并按用户诉求由 LLM Tool Calling 自主追加
通勤计算 / 学习计划 / 模拟面试题。支持多轮对话记忆（画像缓存 + 会话历史）。

## 技术要点

| 能力 | 实现 |
|---|---|
| Agent 编排 | LangGraph StateGraph 多节点工作流 + 条件路由（plan-and-execute） |
| Function Calling | 规划节点单次结构化输出：意图分类 + 检索过滤条件 + 通勤约束抽取 |
| Tool Calling | 评分完成后 LLM `bind_tools` 自主决定调用通勤/学习计划/面试题工具 |
| 混合检索 | LLM 多 Query 改写 → BM25（jieba + BM25Okapi）+ 向量（Chroma）双通道 → RRF 融合 |
| 重排序 | gte-rerank 交叉编码器精排（召回-粗排-精排三段式） |
| 自动化评测 | LLM 反向构造评测集；Recall@K / MRR / nDCG 四配置对比；LLM-as-judge 报告质量 |
| 多轮记忆 | SQLite 用户画像缓存（跨轮免重传简历）+ 会话历史注入规划上下文 |
| 存储分层 | SQLite 为事实源（全部业务数据），Chroma 只存向量 + 最小过滤标量；检索后按 job_id 回填（hydration），换 embedding 模型可从 SQLite 零成本重建索引 |
| 知识库 | 粘贴 JD 两层去重（哈希精确 + 语义近邻）自动入库，与批量建库同一写入路径 |
| 健壮性 | 所有 LLM 环节配确定性规则兜底；单点失败不阻断工作流（errors 透传） |

## 工作流

```
START → planner（FC：task_type + 过滤条件 + 通勤意图）
      → resume_parser → profile_cache（画像缓存）
      ├─ job_retriever（BM25+向量+RRF+rerank）──┐
      ├─ jd_input → jd_ingest（去重入库）───────┤
      └────────────────────────────────────────→ jd_analyzer
      → match_scorer（规则+LLM 多维评分 / 技能差距 / 推荐报告）
      → enhancements（Tool Calling：通勤 / 学习计划 / 3 道面试题）
      → END
```

## 目录结构

```
resume2job/
├── core/          config（模型/检索策略统一配置）、llm（LLM/Embedding 客户端 + JSON 工具）
├── parsing/       resume_parser（PDF→结构化画像）、jd_parser（JD→结构化 profile）
├── retrieval/     indexer（建库）、retriever（混合检索）、bm25 / fusion / rerank
├── scoring/       match_scorer（技能/项目/学历/方向四维评分）、skill_gap
├── generation/    recommendation（推荐报告）、learning_plan（单次调用）、interview（3 题）
├── agent/         state、graph、nodes/（planner / executor / enhancements）
├── storage/       paths、jobs_store（jobs 表事实源）、profile_cache、conversation_store、jd_ingest
├── tools/         commute（高德路线规划）
└── eval/          build_dataset、retrieval_eval、judge、run_eval
chat.py            命令行多轮对话入口
pipeline.py        5 场景端到端验收
scripts/           ingest_jds（建库）、rebuild_index（迁移/重建向量索引）、
                   view_chroma / view_jds（只读巡检）、check_api
```

## 快速开始

```bash
pip install -r requirements.txt
# 环境变量：DASHSCOPE_API_KEY（必需）、AMAP_API_KEY（通勤功能可选）

python scripts/ingest_jds.py        # 1. 把 JDs/ 下的 JD 解析入库
python chat.py                      # 2. 开聊
```

```text
你> 我的简历是 Resumes/resume_test_1.pdf，帮我推荐北京的大模型实习岗位
你> 看看 JDs/jd_test_5.txt 适合我吗，再出几道面试题
你> 对照这个岗位我还差哪些能力，给我一个一个月的学习计划
```

## 评测

```bash
python -m resume2job.eval.run_eval --build        # LLM 反向构造检索评测集（可人工抽查修订）
python -m resume2job.eval.run_eval --retrieval    # vector / bm25 / hybrid / hybrid+rerank 四配置对比
python -m resume2job.eval.run_eval --judge tmp/resume_test_1.json JDs/jd_test_1.txt
                                                  # 端到端报告生成 + LLM-as-judge 三维度打分
python pipeline.py                                # 5 场景端到端回归
```

检索评测输出 Recall@1/@5、MRR@10、nDCG@5 对比表并落盘 `resume2job/eval/reports/`。

## 配置

模型与检索策略集中在 `resume2job/core/config.py`，均可用环境变量覆盖：
`RESUME2JOB_CHAT_MODEL`、`RESUME2JOB_RETRIEVAL_MODE`（vector/bm25/hybrid）、
`RESUME2JOB_USE_RERANK`、`RESUME2JOB_RRF_K` 等。
