# Resume2Job —— 基于 LangGraph 的 Agentic RAG 实习岗位智能推荐系统

输入一份简历 PDF（和/或一条岗位 JD），系统自动完成：规划 → 简历结构化 → 岗位混合检索 →
两层匹配评分（基础适配分 + 偏好排序分）→ 技能差距分析 → 推荐报告；用户指定 job_id 时
（intent=ASSIST）追加通勤查询 / 学习计划 / 岗位定制面试练习题。支持多轮对话记忆
（画像缓存 + 会话历史 + 结构化指代 + 候选池复用）。

## 技术要点

| 能力 | 实现 |
|---|---|
| Agent 编排 | LangGraph StateGraph 多节点工作流 + 条件路由（plan-and-execute） |
| 规划器（planner 包） | 拆分为 nlu_extractor / rule_corrector / clarification / policy_orchestrator / context_builder / trace_logger 子模块；单次 Function Calling 抽结构化语义（intent=RECOMMEND/EVALUATE/ASSIST / job_source / request_more / assist_actions / hard·soft 约束 / 通勤），确定性规则纠错 + 任务式澄清 |
| 会话动作编排 | policy_orchestrator 据本轮语义 + 会话状态决定 session_action：硬约束变→重召回；仅软偏好变→复用池重排；换一批→取池中未展示；USER_JD/SELECTED/ASSIST→读缓存不重检索 |
| 工具执行 | learning_plan / interview 仅由 intent=ASSIST + assist_actions 驱动（需 SELECTED + 有效 job_id），enhancements 节点据 need_* 开关确定性执行通勤/学习计划/面试练习题 |
| 澄清优先 | 缺简历/缺 JD/指代不明/低置信度时 planner 判 clarify、追问而非跑半成品（短路业务链路） |
| 多轮指代 | 结构化 last_results（会话短期状态）须用 job_id 指代，SELECTED/ASSIST 复用候选池结果免重算 |
| 两层评分 | 基础适配分 match_score = 技能×0.55 + 项目×0.45；排序分 rank_score = match_score + 方向偏好分(≤6) + 通勤偏好分(≤4)。城市/学历/岗位类型为硬约束，召回前过滤 |
| 候选池复用 | 推荐时多召回评分一个候选池；换一批从未展示中取、改方向偏好仅重算 direction_bonus 重排，均不重检索/不重评分；仅硬约束变才重召回 |
| 混合检索 | LLM 2 路 Query（方向标签 / 实践细节）+ Query3（方向偏好召回）→ BM25（jieba + BM25Okapi）+ 向量（Chroma）双通道 → RRF 融合；方向偏好仅助召回 + 评分层 direction_bonus，不进 where |
| 重排序 | gte-rerank 交叉编码器精排（召回-粗排-精排三段式） |
| 自动化评测 | LLM 反向构造评测集（多相关岗位 pooling）；Hit@1 / Recall@5 / MRR@10 / nDCG@5 四配置对比；LLM-as-judge 报告质量 |
| 多轮记忆 | 会话短期状态 SessionStore（Redis→SQLite→内存三级回退）+ SQLite 用户画像缓存 + 会话历史注入规划上下文 |
| 存储分层 | SQLite 为事实源（全部业务数据），Chroma 只存向量 + 最小过滤标量；检索后按 job_id 回填（hydration），换 embedding 模型可从 SQLite 零成本重建索引 |
| 知识库 | 粘贴 JD 两层去重（哈希精确 + 语义近邻）自动入库，与批量建库同一写入路径 |
| 健壮性 | 所有 LLM 环节配确定性规则兜底；单点失败不阻断工作流（errors 透传） |

## 工作流

```
START → planner（FC 抽语义 → 规则纠错 → 澄清 → 编排出 ExecutionPlan）
      │     ├─ clarify（缺槽/低置信）────────────────────────────→ END（追问用户）
      → resume_parser → profile_cache（画像缓存）
      ├─ job_retriever（BM25+向量+RRF+rerank）──┐
      ├─ jd_input → jd_ingest（去重入库）───────┤
      └────────────────────────────────────────→ jd_analyzer
      ├─ （候选池复用/换一批/SELECTED/ASSIST）─────────────────→ match_scorer（直达）
      → match_scorer（规则技能分 + LLM 项目分 → 两层评分 / 技能差距 / 推荐报告）
      → enhancements（need_* 开关：通勤 / 学习计划 / 岗位定制面试练习题）
      → END
```

> intent=ASSIST + assist_actions（LEARNING_PLAN / INTERVIEW_PREP，需 SELECTED + 有效 job_id）
> 才触发学习计划 / 面试练习题；推荐与 JD 适配报告默认含「匹配点 + 技能缺口」。

## 目录结构

```
resume2job/
├── core/          config（模型/检索策略统一配置）、llm（LLM/Embedding 客户端 + JSON 工具）
├── parsing/       resume_parser（PDF→结构化画像）、jd_parser（JD→结构化 profile）
├── retrieval/     indexer（建库）、retriever（混合检索）、bm25 / fusion / rerank
├── scoring/       match_scorer（技能/项目/学历/方向四维评分）、skill_gap
├── generation/    recommendation（推荐报告）、learning_plan（单次调用）、interview（3 题）
├── agent/         state、graph、trace、nodes/（executor / enhancements）、
│                  planner/（schema / context_builder / nlu_extractor / rule_corrector /
│                           clarification / policy_orchestrator / trace_logger / node）
├── storage/       paths、jobs_store（jobs 表事实源）、session_store（会话短期状态三级回退）、
│                  preference_store（长期软偏好+时间衰减）、profile_cache、conversation_store、jd_ingest
├── tools/         commute（高德路线规划）
└── eval/          build_dataset、retrieval_eval、judge、run_eval
chat.py            命令行多轮对话入口
pipeline.py        5 场景端到端验收
scripts/           ingest_jds（建库）、rebuild_index（迁移/重建向量索引）、
                   export_planner_traces（决策统计 + SFT 样本导出）、
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
