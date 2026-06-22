# Resume2Job 端到端验收报告

> 生成时间：2026-06-19 12:29:15　|　脚本：`python pipeline.py`

## 测试原则

1. **场景即用户旅程**：每个场景通过 `run_turn`（真实图执行）验证一条完整链路，不直接调业务函数；
2. **机制隐式验证**：画像缓存、对话历史注入等基础机制在多轮场景（S3）内以断言覆盖，不单独立场景；
3. **一场景多断言**：核验链路上各关键节点产物（规划抽取、检索回填、report_views 驱动增强、澄清短路、输出结构），失败精确到断言；
4. **数据隔离**：验收目录拷贝真实库后运行，入库/画像/会话写操作不污染 `data/`；
5. **职责分离**：检索质量指标（Recall@K / MRR / nDCG）由 `resume2job/eval` 负责，本验收只管链路正确性。

## 场景矩阵

| 场景 | 用户旅程 | 覆盖能力 |
|---|---|---|
| S1 岗位推荐全链路 | 传简历 +「推荐北京的大模型实习岗位」 | planner FC 抽 intent=RECOMMEND + 条件、混合检索（BM25+向量+RRF+rerank）、SQLite 回填、城市约束、未请求时不开增强视图 |
| S2 JD 评估 + 全增强 | 传简历 + JD +「适合吗？给学习计划和面试题」 | intent=EVALUATE 链路、jd_ingest 去重入库、report_views 驱动多增强、skill_gap / 学习计划 / 3 道面试题 |
| S3 多轮记忆 + 通勤 + 指代 | 轮① 评估 JD；轮② 不传简历「我住中关村，公交地铁1小时内」求推荐；轮③「第二个出面试题」 | 画像缓存命中、通勤意图抽取与路线/时长进报告、**last_results 结构化指代（「第二个」→job_id）+ FOLLOWUP 复用** |
| S4 城市硬约束提前终止 | 「推荐深圳的实习岗位」（库中无深圳岗） | 城市硬约束（检索级联不丢城市）、零浪费提前终止、final_response 告知并询问放宽 |
| S5 澄清机制 | ①无简历无缓存求推荐 ②有简历缺 JD 求评估 | 缺关键槽位时 planner 判 clarify、追问而非跑半成品、业务节点短路 |

## 本次运行结果

**总计：5 / 5 通过**

### S1 岗位推荐全链路 — ✅ PASS（331.4s）

召回 3 岗，首位 final_score=67

- ✓ intent=RECOMMEND（实际 RECOMMEND）
- ✓ FC planner 抽出 city=北京（实际 北京）
- ✓ [planner] planner 节点抽出 city=北京
- ✓ [job_retriever] 检索节点召回候选 ≥1
- ✓ [match_scorer] 评分节点产出结果 ≥1
- ✓ 召回并评分岗位 ≥1（实际 3）
- ✓ 每个岗位都有 final_score
- ✓ 每个岗位都有推荐报告
- ✓ 候选 jd_profile 来自 SQLite 回填（字段完整）
- ✓ 全部岗位城市均为北京（硬约束生效）
- ✓ 未请求时不生成学习计划
- ✓ 未请求时不生成面试题

<details><summary>🔍 链路追踪（trace_id=`accept_s1#1`）</summary>

路径：`planner → resume_parser → profile_cache → job_retriever → jd_analyzer → match_scorer → enhancements`（总耗时 331.4s）

| seq | 节点 | 耗时(s) | 改动字段 | 关键信号 |
|---|---|---|---|---|
| 1 | planner | 2.2 | commute_constraint, commute_intent, hard_constraints, intent, plan, retrieval_config, soft_preferences | intent=RECOMMEND，city=北京，通勤=transit/Nonemin |
| 2 | resume_parser | 83.6 | resume_profile | intent=RECOMMEND，city=北京，通勤=transit/Nonemin |
| 3 | profile_cache | 0.0 | profile_id, profile_source | intent=RECOMMEND，city=北京，profile=new_upload，通勤=transit/Nonemin |
| 4 | job_retriever | 18.1 | candidate_jobs | intent=RECOMMEND，city=北京，profile=new_upload，召回=3，通勤=transit/Nonemin |
| 5 | jd_analyzer | 0.0 | jd_profiles | intent=RECOMMEND，city=北京，profile=new_upload，召回=3，jd=3，通勤=transit/Nonemin |
| 6 | match_scorer | 227.5 | match_results, skill_gaps | intent=RECOMMEND，city=北京，profile=new_upload，召回=3，jd=3，评分=3，top=67，通勤=transit/Nonemin |
| 7 | enhancements | 0.0 | — | intent=RECOMMEND，city=北京，profile=new_upload，召回=3，jd=3，评分=3，top=67，通勤=transit/Nonemin |

</details>

### S2 JD 评估 + 全增强 — ✅ PASS（193.9s）

score=76, 学习计划 2 阶段, 面试题 3 道

- ✓ intent=EVALUATE（实际 EVALUATE）
- ✓ jd_ingest 入库/去重命中（job_id=jd_test_5）
- ✓ [jd_ingest] jd_ingest 节点产出 job_id
- ✓ [enhancements] enhancements 节点按 report_views 执行学习计划 + 3 道面试题
- ✓ 单 JD 评估产出 1 条结果（实际 1）
- ✓ 推荐报告非空
- ✓ skill_gap 有逐项分析
- ✓ report_views 驱动学习计划（阶段数 2）
- ✓ 面试题恰好 3 道（实际 3）

<details><summary>🔍 链路追踪（trace_id=`accept_s2#1`）</summary>

路径：`planner → resume_parser → profile_cache → jd_input → jd_ingest → jd_analyzer → match_scorer → enhancements`（总耗时 193.9s）

| seq | 节点 | 耗时(s) | 改动字段 | 关键信号 |
|---|---|---|---|---|
| 1 | planner | 1.0 | commute_constraint, commute_intent, intent, plan | intent=EVALUATE，通勤=transit/Nonemin |
| 2 | resume_parser | 67.2 | resume_profile | intent=EVALUATE，通勤=transit/Nonemin |
| 3 | profile_cache | 0.0 | profile_id, profile_source | intent=EVALUATE，profile=new_upload，通勤=transit/Nonemin |
| 4 | jd_input | 0.0 | jd_profiles | intent=EVALUATE，profile=new_upload，jd=1，通勤=transit/Nonemin |
| 5 | jd_ingest | 0.0 | ingested_job_id, jd_is_duplicate | intent=EVALUATE，profile=new_upload，jd=1，ingest=jd_test_5，通勤=transit/Nonemin |
| 6 | jd_analyzer | 0.0 | — | intent=EVALUATE，profile=new_upload，jd=1，ingest=jd_test_5，通勤=transit/Nonemin |
| 7 | match_scorer | 71.9 | match_results, skill_gaps | intent=EVALUATE，profile=new_upload，jd=1，评分=1，top=76，ingest=jd_test_5，通勤=transit/Nonemin |
| 8 | enhancements | 53.7 | interview_prep, learning_plan | intent=EVALUATE，profile=new_upload，jd=1，评分=1，top=76，ingest=jd_test_5，通勤=transit/Nonemin，学习=2阶段，面试=3题 |

</details>

### S3 多轮记忆 + 通勤 — ✅ PASS（766.1s）

首位通勤：地铁/公交约 32 分钟（路线：地铁10号线外环），符合通勤要求；指代第二个→腾讯大模型算法工程师

- ✓ 轮① 画像保存 new_upload（实际 new_upload）
- ✓ 轮② 复用缓存画像（实际 cached）
- ✓ [profile_cache] 轮② profile_cache 命中缓存
- ✓ [planner] planner 抽出通勤意图（60min / transit）
- ✓ 通勤地址抽取（实际 北京市海淀区中关村）
- ✓ 时间上限换算 60 分钟（实际 60）
- ✓ 交通方式=transit（实际 transit）
- ✓ 通勤场景扩大召回 top_k≥10
- ✓ 召回并评分岗位 ≥1（实际 10）
- ✓ 通勤计算结果非空
- ✓ 报告含【通勤】段
- ✓ 报告含通勤时间
- ✓ 报告含通勤路线（公交地铁换乘串）
- ✓ [planner] planner 识别为 FOLLOWUP 追问（非新检索）
- ✓ 「第二个」解析出 job_id（实际 腾讯大模型算法工程师）
- ✓ 指代命中轮②结果的第二个岗位
- ✓ 针对指代岗位出 3 道面试题（实际 3）

<details><summary>🔍 链路追踪 轮1（trace_id=`accept_s3#1`）</summary>

路径：`planner → resume_parser → profile_cache → jd_input → jd_ingest → jd_analyzer → match_scorer → enhancements`（总耗时 178.6s）

| seq | 节点 | 耗时(s) | 改动字段 | 关键信号 |
|---|---|---|---|---|
| 1 | planner | 1.1 | commute_constraint, commute_intent, intent, plan | intent=EVALUATE，通勤=transit/Nonemin |
| 2 | resume_parser | 93.8 | resume_profile | intent=EVALUATE，通勤=transit/Nonemin |
| 3 | profile_cache | 0.0 | profile_id, profile_source | intent=EVALUATE，profile=new_upload，通勤=transit/Nonemin |
| 4 | jd_input | 0.0 | jd_profiles | intent=EVALUATE，profile=new_upload，jd=1，通勤=transit/Nonemin |
| 5 | jd_ingest | 0.0 | ingested_job_id, jd_is_duplicate | intent=EVALUATE，profile=new_upload，jd=1，ingest=jd_test_5，通勤=transit/Nonemin |
| 6 | jd_analyzer | 0.0 | — | intent=EVALUATE，profile=new_upload，jd=1，ingest=jd_test_5，通勤=transit/Nonemin |
| 7 | match_scorer | 60.3 | match_results, skill_gaps | intent=EVALUATE，profile=new_upload，jd=1，评分=1，top=87，ingest=jd_test_5，通勤=transit/Nonemin |
| 8 | enhancements | 23.3 | learning_plan | intent=EVALUATE，profile=new_upload，jd=1，评分=1，top=87，ingest=jd_test_5，通勤=transit/Nonemin，学习=2阶段 |

</details>

<details><summary>🔍 链路追踪 轮2（trace_id=`accept_s3#2`）</summary>

路径：`planner → resume_parser → profile_cache → job_retriever → jd_analyzer → match_scorer → enhancements`（总耗时 546.6s）

| seq | 节点 | 耗时(s) | 改动字段 | 关键信号 |
|---|---|---|---|---|
| 1 | planner | 5.3 | commute_constraint, commute_intent, hard_constraints, intent, plan, retrieval_config, soft_preferences | intent=RECOMMEND，city=北京，通勤=transit/60min |
| 2 | resume_parser ⚠️+1err | 0.0 | errors | intent=RECOMMEND，city=北京，通勤=transit/60min，errors=1 |
| 3 | profile_cache | 0.0 | plan, profile_id, profile_source, resume_profile | intent=RECOMMEND，city=北京，profile=cached，通勤=transit/60min，errors=1 |
| 4 | job_retriever | 25.8 | candidate_jobs | intent=RECOMMEND，city=北京，profile=cached，召回=10，通勤=transit/60min，errors=1 |
| 5 | jd_analyzer | 0.0 | jd_profiles | intent=RECOMMEND，city=北京，profile=cached，召回=10，jd=10，通勤=transit/60min，errors=1 |
| 6 | match_scorer | 513.6 | match_results, skill_gaps | intent=RECOMMEND，city=北京，profile=cached，召回=10，jd=10，评分=10，top=89，通勤=transit/60min，errors=1 |
| 7 | enhancements | 1.9 | commute_results, final_response, match_results | intent=RECOMMEND，city=北京，profile=cached，召回=10，jd=10，评分=10，top=89，通勤=transit/60min，通勤结果=10，final_resp✓，errors=1 |

</details>

<details><summary>🔍 链路追踪 轮3（trace_id=`accept_s3#3`）</summary>

路径：`planner → resume_parser → profile_cache → jd_analyzer → match_scorer → enhancements`（总耗时 40.8s）

| seq | 节点 | 耗时(s) | 改动字段 | 关键信号 |
|---|---|---|---|---|
| 1 | planner | 1.4 | commute_constraint, commute_intent, intent, match_results, plan, selected_item_ref | intent=EVALUATE，评分=1，top=87，通勤=transit/Nonemin |
| 2 | resume_parser ⚠️+1err | 0.0 | errors | intent=EVALUATE，评分=1，top=87，通勤=transit/Nonemin，errors=1 |
| 3 | profile_cache | 0.0 | plan, profile_id, profile_source, resume_profile | intent=EVALUATE，profile=cached，评分=1，top=87，通勤=transit/Nonemin，errors=1 |
| 4 | jd_analyzer | 0.0 | — | intent=EVALUATE，profile=cached，评分=1，top=87，通勤=transit/Nonemin，errors=1 |
| 5 | match_scorer ⚠️+1err | 0.0 | errors | intent=EVALUATE，profile=cached，评分=1，top=87，通勤=transit/Nonemin，errors=2 |
| 6 | enhancements | 39.4 | interview_prep | intent=EVALUATE，profile=cached，评分=1，top=87，通勤=transit/Nonemin，面试=3题，errors=2 |

</details>

### S4 城市硬约束提前终止 — ✅ PASS（147.4s）

提前终止：知识库暂无「武汉」的实习岗位，本轮未做推荐。要不要看看不限城市的岗位？回复「不限城市」即可。

- ✓ FC planner 抽出 city=武汉（实际 武汉）
- ✓ [job_retriever] 检索节点召回=0 并就地写出告知文案（提前终止）
- ✓ 评分节点未对错误城市产出结果（零浪费）
- ✓ match_results 为空（未对错误城市浪费评分调用）
- ✓ 明确告知暂无武汉岗位（实际：知识库暂无「武汉」的实习岗位，本轮未做推荐。要不要看看不限城市的岗位？回复「不限城市」即可。）
- ✓ 提示用户可放宽到不限城市

<details><summary>🔍 链路追踪（trace_id=`accept_s4#1`）</summary>

路径：`planner → resume_parser → profile_cache → job_retriever → jd_analyzer → match_scorer → enhancements`（总耗时 147.4s）

| seq | 节点 | 耗时(s) | 改动字段 | 关键信号 |
|---|---|---|---|---|
| 1 | planner | 1.1 | commute_constraint, commute_intent, hard_constraints, intent, plan, retrieval_config | intent=RECOMMEND，city=武汉，通勤=transit/Nonemin |
| 2 | resume_parser | 132.0 | resume_profile | intent=RECOMMEND，city=武汉，通勤=transit/Nonemin |
| 3 | profile_cache | 0.0 | profile_id, profile_source | intent=RECOMMEND，city=武汉，profile=new_upload，通勤=transit/Nonemin |
| 4 | job_retriever | 14.3 | final_response | intent=RECOMMEND，city=武汉，profile=new_upload，通勤=transit/Nonemin，final_resp✓ |
| 5 | jd_analyzer | 0.0 | — | intent=RECOMMEND，city=武汉，profile=new_upload，通勤=transit/Nonemin，final_resp✓ |
| 6 | match_scorer | 0.0 | — | intent=RECOMMEND，city=武汉，profile=new_upload，通勤=transit/Nonemin，final_resp✓ |
| 7 | enhancements | 0.0 | — | intent=RECOMMEND，city=武汉，profile=new_upload，通勤=transit/Nonemin，final_resp✓ |

</details>

### S5 澄清机制 — ✅ PASS（1.9s）

缺简历/缺 JD 均触发澄清、未跑业务链路

- ✓ [planner] ①planner 判定需澄清
- ✓ ①澄清时不产出岗位结果
- ✓ ①追问上传简历
- ✓ ①未跑检索节点（澄清短路）
- ✓ ②planner 判定需澄清
- ✓ ②澄清时不产出岗位结果
- ✓ ②给出澄清问题

<details><summary>🔍 链路追踪 轮1（trace_id=`accept_s5a#1`）</summary>

路径：`planner → resume_parser → profile_cache`（总耗时 0.9s）

| seq | 节点 | 耗时(s) | 改动字段 | 关键信号 |
|---|---|---|---|---|
| 1 | planner | 0.9 | commute_constraint, commute_intent, final_response, intent, plan | intent=RECOMMEND，通勤=transit/Nonemin，final_resp✓ |
| 2 | resume_parser | 0.0 | — | intent=RECOMMEND，通勤=transit/Nonemin，final_resp✓ |
| 3 | profile_cache ⚠️+1err | 0.0 | errors | intent=RECOMMEND，通勤=transit/Nonemin，final_resp✓，errors=1 |

</details>

<details><summary>🔍 链路追踪 轮2（trace_id=`accept_s5b#1`）</summary>

路径：`planner → resume_parser → profile_cache`（总耗时 1.0s）

| seq | 节点 | 耗时(s) | 改动字段 | 关键信号 |
|---|---|---|---|---|
| 1 | planner | 1.0 | commute_constraint, commute_intent, final_response, intent, plan | intent=EVALUATE，通勤=transit/Nonemin，final_resp✓ |
| 2 | resume_parser | 0.0 | — | intent=EVALUATE，通勤=transit/Nonemin，final_resp✓ |
| 3 | profile_cache | 0.0 | profile_id, profile_source, resume_profile | intent=EVALUATE，profile=cached，通勤=transit/Nonemin，final_resp✓ |

</details>

## 运行环境

- 主模型：`deepseek-v4-pro`　规划：`qwen-flash`　评分：`qwen-plus`
- 检索：mode=`hybrid`，rerank=`True`（`qwen3-rerank`）
- Embedding：`text-embedding-v3`

## 已知限制

- 依赖联网与 `DASHSCOPE_API_KEY`；S3 的通勤断言依赖 `AMAP_API_KEY`（缺失时降级为通勤说明断言）；
- 意图分类 / 条件抽取 / 工具决策由 LLM 完成，存在小概率非确定性误判——单场景偶发 FAIL 可重跑确认；
- 检索召回质量不在本验收范围，见 `resume2job/eval` 的指标评测报告。
