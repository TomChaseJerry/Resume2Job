# resume2job/eval —— 分层评测体系

把「效果评估」拆到每一层，而不是只看最终 Recall——这样能判断问题出在**抽取 / Planner / 检索排序**哪一环。

## 评测层（Stage 4 新增 4 个 + 既有 2 个）

| 模块 | 评什么 | 数据源 | 是否需 API |
|---|---|---|---|
| `extraction_eval.py` | 抽取层：JD 抽取完整度 / 质量 / JSON 合法率；(gold) 技能 P/R/F1 + Exact Match | 已入库 jd_profile（或 reparse JDs/） | 否（reparse 需） |
| `planner_eval.py` | Planner：意图 / 路由 / 槽位 / 澄清；规则兜底率、工具请求率；(gold) 准确率 | `data/planner_traces.jsonl` | 否 |
| `ranking_eval.py` | 排序运营：约束违例率、公司集中度、新鲜度、各阶段时延、单请求成本 | `data/request_traces.jsonl`（Stage 2） | 否（--relevance 需） |
| `fairness_audit.py` | 曝光分布 + 约束一致性 + 数据质量（**非人口统计学公平**） | jobs 目录 + request_traces | 否 |
| `retrieval_eval.py` | 检索相关性：Recall@K / MRR / nDCG（四配置对比） | `eval/data/retrieval_dataset.jsonl` + search_jobs | 是 |
| `judge.py` | 生成层：LLM-as-judge（忠实性 / 有用性 / 证据性） | 端到端 | 是 |

## 一键入口

```bash
# 0 LLM 的四层（读已有数据 + 目录）
python -m resume2job.eval.run_eval --extraction --planner --ranking --fairness
# 单独 + gold（算准确率）
python -m resume2job.eval.run_eval --extraction --gold eval/data/jd_gold.jsonl
python -m resume2job.eval.run_eval --planner --gold eval/data/planner_gold.jsonl
# 需 API 的相关性 / 生成层
python -m resume2job.eval.run_eval --build --retrieval        # 构造评测集 + 四配置相关性
python -m resume2job.eval.run_eval --judge resume.json jd.txt # 端到端 + LLM-judge
python -m resume2job.eval.run_eval --extraction --extraction-reparse  # 重跑 parse_jd 测 JSON 合法率

# 一键验收（0 LLM，含断言）
python scripts/verify_eval.py
```

报告统一落 `resume2job/eval/reports/<层>_<时间戳>.md`。

## gold 文件格式（可选，jsonl 每行一条）

```jsonc
// extraction gold
{"job_id":"jd_test_1","hard_skills":["强化学习","python"],"education_level":"硕士","job_type":"社招","cities":["北京"]}
// planner gold
{"query":"帮我找北京的Agent实习","intent":"RECOMMEND","session_action":"RETRIEVE","hard_constraints":{"city":"北京","job_type":"实习"},"clarify":false}
```

无 gold 时各层仍输出分布 / 完整度 / 一致性等可跑指标；有 gold 才算 P/R/F1 / 准确率。

## fairness_audit 的立场（治理展示）

**不**声称用户群体公平，**不**从简历推断性别 / 地域 / 学校层级。它做的是：① 约束一致性（城市 / 类型违例、
unknown 地点误推）；② 曝光分布（公司 / 方向 / 来源 / 城市占比、Top-K 集中度 HHI、同公司重复）；
③ 数据质量（各来源缺字段率、各方向技能抽取失败率、过期 / unknown 城市比例）。目的是**发现**集中度 /
偏斜 / 数据质量问题（如某公司占 70% Top-10、某方向 JD 解析总失败），而非证明系统已公平。
