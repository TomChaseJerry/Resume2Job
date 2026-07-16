# -*- coding: utf-8 -*-
"""ingest/normalizer.py — 入库前的「原始记录」规范化（信封层）。

职责边界（重要，避免与 jd_parser 重复造轮子）：
    - **本模块**：清洗 connector 产出的「原始记录信封」——统一换行 / 去 BOM / 去首尾空白、
      裁剪 company/title 提示字段、规范来源标识。即「把任意来源的脏输入收敛成干净的 jd_text + 元数据」，
      发生在 **parse_jd 之前**。
    - **字段级语义归一**（学历→本/硕/博、城市去「市」后缀、技能原子化与同义归并、job_type 三桶、
      可替代技能组）**一律委托 jd_parser**（parse_jd 内部已统一处理）。本模块**不复制**这些逻辑，
      仅在文末 re-export 这些函数，作为「归一化入口」的统一可发现点（下游想做字段归一时从这里取）。

**清洗刻意保守**：clean_text 只做不改变内容指纹（jd_hash）稳定性的轻量变换。NFKC 全角折叠 / 逐行去尾
空白 / 压空行等激进归一会让既有库里每条记录的哈希失配、触发整库重解析+重嵌入——那应作为一次显式的
「renormalize 迁移」单独进行，而非每次接入隐式触发。
"""

from __future__ import annotations

from dataclasses import replace

from resume2job.ingest.models import RawJobPayload


def clean_text(text: str) -> str:
    """规范化一段原始文本的「信封」：统一换行、去开头 BOM、去首尾空白。

    刻意保守，保证「同一来源文本 → 同一 jd_hash」稳定且与历史入库口径一致：
      - CRLF / CR → LF（粘贴文本常带 CRLF；文件经 open() 已是 LF，此步幂等）；
      - 去掉开头 BOM（utf-8 读出的 \\ufeff 不会被 str.strip() 清掉，会污染首字段 / 哈希）；
      - 整体 strip。
    """
    if not isinstance(text, str) or not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = s.lstrip("﻿")
    return s.strip()


def _clean_field(v):
    """裁剪单个字符串元数据字段；空串 → None。"""
    if not isinstance(v, str):
        return v
    s = v.strip()
    return s or None


def normalize_raw_payload(payload: RawJobPayload) -> RawJobPayload:
    """清洗 connector 产出的原始记录信封，返回**新的** RawJobPayload（不就地修改）。

    只做信封层轻量清洗（文本换行 / BOM / 空白 + 元数据裁剪），不触碰字段语义——
    语义归一在随后的 parse_jd 内完成。
    """
    return replace(
        payload,
        raw_jd=clean_text(payload.raw_jd),
        company=_clean_field(payload.company),
        title=_clean_field(payload.title),
        source=(_clean_field(payload.source) or "unknown_source"),
        source_job_id=_clean_field(payload.source_job_id),
        canonical_url=_clean_field(payload.canonical_url),
    )


# ===== 字段级归一化的统一入口（委托 jd_parser，本模块不重复实现）=====
# 下游若需在解析之外单独做字段归一，从这里导入，保证与解析层口径一致。
from resume2job.parsing.jd_parser import (  # noqa: E402  (re-export 放文末，避免与上文清洗逻辑混淆)
    normalize_city,
    job_cities,
    normalize_job_type,
    degree_min_rank_and_status,
    split_compound_skill,
)

__all__ = [
    "clean_text",
    "normalize_raw_payload",
    # 委托 jd_parser 的字段级归一（re-export）
    "normalize_city",
    "job_cities",
    "normalize_job_type",
    "degree_min_rank_and_status",
    "split_compound_skill",
]
