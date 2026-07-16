# -*- coding: utf-8 -*-
"""ingest/versions.py — 解析 / 索引 / Embedding 版本号（单一事实源）。

为什么需要版本号：一条 JD 会流经「LLM 解析 → jd_profile → SQLite → index_text → BM25/Chroma」。
若解析逻辑、index_text 拼接方式或 embedding 模型升级了，旧库里的画像 / 检索文本 / 向量就「陈旧」
但表面看不出来。给每条记录盖上当时的版本戳，就能回答「这条记录由哪版解析器 / 哪个 embedding
模型生成」，从而：
    - 升级解析器后，用 SQL 找出 parser_version 落后的记录，批量重解析；
    - 换 embedding 模型后，按 embedding_version 找出需重嵌入的向量（scripts/rebuild_index）；
    - 观测层（Stage 2）记录 prompt_version / model_version / index_version 时复用同一组版本。

约定：改动对应逻辑时**手动**递增下面的常量——让版本变化成为一次有意识的决定，而非自动漂移。
"""

from resume2job.core.config import EMBEDDING_MODEL

# JD 解析器版本：改动 jd_parser 的 SYSTEM_PROMPT / 抽取规则 / normalize_* 后处理时递增。
PARSER_VERSION = "2026.06"

# index_text 拼接版本：改动 indexer.build_index_text 的字段 / 顺序 / 分隔符时递增
# （它同时决定 BM25 语料与送入 embedding 的文本，变了即意味着检索语义变了）。
INDEX_TEXT_VERSION = "1"


def embedding_version() -> str:
    """当前 embedding 版本 = 模型名（换模型即换版本，向量维度 / 语义随之改变）。

    取 config.EMBEDDING_MODEL，使「改 config 换模型」与「记录里的版本戳」天然一致。
    """
    return EMBEDDING_MODEL
