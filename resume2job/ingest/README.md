# resume2job/ingest —— 岗位数据接入与生命周期管理（Stage 1）

把「岗位数据**从哪来、如何更新、是否过期、如何进入索引**」从检索 / 评分链路里抽出来，独立成一层。
解决四类现实问题：① 数据更新时 SQLite / Chroma / BM25 三处一致；② 岗位有生命周期（过期 / 下线不再被推荐）；
③ 每条记录可追溯版本（解析器 / index_text / embedding）；④ 低质量 JD 不污染召回池。

## 一句话数据流

```
connector.fetch() → RawJobPayload
        → normalizer（信封清洗：换行/BOM/空白）
        → parse_jd（LLM 解析，字段级归一都在这里）
        → validator（质量闸门：is_valid + quality_score + warnings）
        → lifecycle 去重（精确 → URL/外部ID → 语义）
        → JobRecord（盖版本戳）→ jobs_store.upsert_job（SQLite 事实源）
        → embedding → Chroma（向量索引）
```

唯一入口是 **`lifecycle.ingest_record()`**；批量建库（`indexer.index_jobs`）和运行时粘贴入库
（`jd_ingest_node`）现在都走它，所以两条路径永远写出一致的数据。

## 文件清单

| 文件 | 作用 | 关键符号 |
|---|---|---|
| `models.py` | 三个数据结构（接入边界的类型契约） | `RawJobPayload`（来源契约）、`JobRecord`（标准记录，含 `from_jd_profile`/`from_store_row`/`to_store_dict`）、`IngestResult`（接入结果） |
| `versions.py` | 三类版本号的单一事实源（改逻辑时手动递增） | `PARSER_VERSION`、`INDEX_TEXT_VERSION`、`embedding_version()` |
| `validator.py` | 岗位**质量**校验（纯规则，0 LLM；区别于 jd_parser 的**结构**校验） | `validate_job(jd_profile, jd_text) → QualityReport{is_valid, quality_score, warnings}` |
| `normalizer.py` | 入库前**信封层**清洗（换行/BOM/空白）；字段级语义归一 re-export jd_parser，不重复实现 | `clean_text`、`normalize_raw_payload` |
| `lifecycle.py` | **接入编排核心**：去重 / 新建 / 更新 / 幂等跳过 / 过期下线 / 回填 | `ingest_record`、`ingest_all`、`mark_expired`/`mark_removed`/`reactivate`、`sweep_stale`、`backfill_lifecycle_fields` |
| `connectors/base.py` | 接入器抽象基类 | `Connector.fetch()` |
| `connectors/local_file.py` | 本地 txt/md/json（文件名 stem 作 job_id，幂等更新） | `LocalFileConnector` |
| `connectors/csv_file.py` | CSV 批量导入（自动识别中英文列名） | `CSVConnector` |
| `connectors/user_paste.py` | 用户粘贴的一段 JD（无 job_id → 全量去重） | `UserPasteConnector`、`payload_from_text` |
| `connectors/official_career.py` | 官方招聘页增量同步（**Phase 2 占位**，fetch 抛 NotImplementedError；BOSS 同理未来接入） | `OfficialCareerConnector` |

## `ingest_record` 的两种去重模式

- **身份模式**（payload 带 `job_id`，如本地文件以文件名作 id）：按 job_id 幂等。哈希未变 → `unchanged`
  （零 token）；变了 → `updated`（重解析 + 删旧向量重嵌入）；不存在 → `created`。
- **全量去重模式**（无 `job_id`，如用户粘贴）：精确（公司+标题+原文哈希）→ `canonical_url` → `source_job_id`
  → 语义（向量近邻 > 0.92）逐层判重；都未命中才 `created`。

`strict=True` 时质量不达标（`is_valid=False`）的岗位被拦截为 `invalid` 不入库；默认非严格（入库并记 `quality_score`）。

## jobs 表新增的 10 列（`storage/jobs_store.py`）

`status`（active/expired/removed，仅 active 可召回）、`source_job_id`、`canonical_url`、`content_hash`
（index_text 内容指纹，驱动重嵌入判断）、`collected_at`、`last_verified_at`、`parser_version`、
`embedding_version`、`index_text_version`、`quality_score`。`get_eligible_jobs` 已加 `status='active'` 闸门。

## 怎么运行 / 在哪看结果

```bash
# 1) 验收体检（推荐先跑，全程不调 API、不改真实库）
python scripts/verify_ingest.py
# 1b) 额外跑真实端到端（隔离临时库，需联网 + DASHSCOPE_API_KEY，真实库不受影响）
python scripts/verify_ingest.py --with-api

# 2) 真实建库 / 增量入库（会调 embedding/解析 API，写真实库）
python scripts/ingest_jds.py

# 3) 浏览入库结果
python scripts/view_jds.py
```

结果都打印在终端；持久化数据在 `data/resume2job.db` 的 `jobs` 表。

## 旧库升级须知

导入 `jobs_store` 时 `init_db()` 会自动幂等 ALTER 补齐新列；新列的值用
`from resume2job.ingest.lifecycle import backfill_lifecycle_fields; backfill_lifecycle_fields()`
回填（纯 SQLite + 规则校验，不调 LLM、不重嵌入）。本机真实库已回填过。
`embedding_version` 回填假定当前 embedding 模型；若曾换模型，需 `scripts/rebuild_index.py` 重建向量。
