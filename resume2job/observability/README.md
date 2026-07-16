# resume2job/observability —— 请求级链路可观测性（Stage 2）

回答三个问题：

1. **这次推荐到底经历了什么？** → `events.py`：每次 `run_turn` = 一个 `request_id`，沿途各阶段
   （规划 / 召回各通道 / 融合 / 精排 / 评分 / 工具 / 每次 LLM 的 token 与时延）快照写进一条 `RequestTrace`，结束落盘。
2. **升级前后同一请求结果怎么变？** → `replay.py`：回放历史请求 + `diff_traces` 对比 Top-K / 名次分 / token / 时延 / 版本变化。
3. **拿日志做训练 / 展示会泄露 PII 吗？** → `redaction.py`：脱敏（保留技能 / 教育 / 偏好，移除姓名 / 邮箱 / 手机 / 住址等身份信息）。

与既有两套 trace 的分工：`planner_traces` 只覆盖「规划输入→输出」；`agent/trace.py` 是 pipeline 验收用的逐节点 State diff；
本模块是**生产链路**的全量可观测 + 回放 / 对比 + 后训练数据闭环的基础（Stage 3 ranking 特征、Stage 4 eval 都吃它）。

## 注入方式（ContextVar 记录器）

- `run_turn` 分配 `request_id`，用 `events.request_scope(...)` 开一个 trace 设进 contextvar；退出时落盘。
- 各处调 `events.record_*(...)` 写快照——**无活跃 trace 时全部 no-op**，故 `run_turn_traced` / pipeline / eval / 直接调用模块均不受影响。
- `core/llm.py` 的 `call_llm` / `get_embedding` 包一层自动记 token + 时延；`match_scorer` 的内联 project 评分调用也单独挂了钩。
- 每个图节点经 `events.run_node` 包一层记**每节点时延** + 设「当前节点」，供 LLM 调用归属到节点。
- **线程注意**：contextvars 不会自动传播到 `ThreadPoolExecutor`；`_narrate_batch`（评分叙述并发）用 `copy_context()` 把上下文复制进工作线程，否则那批 LLM token 会丢。

## 文件清单

| 文件 | 作用 | 关键符号 |
|---|---|---|
| `events.py` | RequestTrace + contextvar 记录器 + 落盘 + 读取 | `request_scope`、`run_node`、`record_*`、`record_feedback`、`get_trace`、`load_traces`、`snapshot_model_versions`、`snapshot_index_version` |
| `replay.py` | 回放历史请求 + 前后对比 | `replay_request`、`replay_batch`、`diff_traces`、`render_diff` |
| `redaction.py` | 脱敏（纯规则，0 LLM） | `redact_text`、`redact_obj`、`redact_resume_profile`、`redact_trace` |

## trace 结构（落 data/request_traces.jsonl 的一条）

```
request_id, session_id, user_query, created_at
query_plan            { intent, session_action, hard_constraints, soft_preferences, clarify, decided_by }
retrieval             { queries{q1,q2,q3}, constraint_filter, allowed_count,
                        dense_candidates[], bm25_candidates[], rrf_candidates[], rerank[] }   # 各阶段 job_id + 分数
rank_features[]       # 每个候选的评分特征（skill/project/direction/commute/rank_score/match_level/education_gate…）
final_ranked_jobs[]   # 最终呈现名次（job_id, rank, rank_score, match_level）
tool_calls[]          # commute / learning_plan / interview（名称 + 时延 + 概要）
llm_calls[]           # 每次调用：node, kind(chat/embedding), model, prompt/completion/total_tokens, latency_ms
token_usage           # 汇总：总量 + by_node + by_kind
latency_ms            # 每节点 + total
model_versions        # chat/planner/score/judge/rerank/embedding + 检索模式
index_version         # embedding/index_text/parser 版本 + corpus_signature + job_count
errors[]              # 节点级错误
user_feedback         # saved / skipped / not_interested / applied …（事后回填，权威在 SQLite）
```

## 落盘

- `data/request_traces.jsonl`：完整 trace（不可变事实，append）。
- SQLite `request_traces` 表：可查询摘要（request_id / intent / n_final / total_tokens / total_latency_ms / feedback）+ 承载 `user_feedback` 事后回填。

## 怎么运行 / 在哪看

```bash
# 验收体检（默认离线，不调 API、不改业务库）
python scripts/verify_observability.py
# 额外真实采一条（需联网）
python scripts/verify_observability.py --with-api

# 生产里：任意 run_turn / chat.py 一轮对话都会自动产出一条 trace
# 回填反馈：from resume2job.observability import record_feedback; record_feedback(request_id, "saved")
# 回放对比：from resume2job.observability.replay import replay_request, diff_traces
```

结果在 `data/request_traces.jsonl` 与 SQLite `request_traces` 表。

## 已知缺口（Stage 2 范围内的取舍）

- **planner 的 token 未计**：planner 走 LangChain `get_chat_llm(...).with_structured_output().invoke()`，在 `core/llm` 之外被调用，
  当前只采到 planner 节点**时延**、未采 token（qwen-flash、输出短，成本占比很小）。如需，后续可挂 LangChain callback 补齐。
- replay 默认复用原 session 重跑（复现缓存画像 / 上下文），会在该 session 追加一轮——开发期回归工具，副作用可接受。
