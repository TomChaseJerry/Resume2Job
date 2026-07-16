# -*- coding: utf-8 -*-
"""ingest/connectors/csv_file.py — CSV 批量接入器。

适合「整理好的岗位表」批量导入。自动识别常见列名（中英文），也可用 text_column 显式指定 JD 正文列。
无 id 列时不指定 job_id（→ 全量去重模式，按公司+标题+哈希 / 语义去重，重复行自动合并）。
"""

from __future__ import annotations

import csv
from typing import Iterator, Optional, Sequence

from resume2job.ingest.connectors.base import Connector
from resume2job.ingest.models import RawJobPayload, SOURCE_CSV

_TEXT_COLS = ("jd_text", "raw_jd", "jd", "text", "description", "content",
              "岗位描述", "职位描述", "jd原文", "描述")
_COMPANY_COLS = ("company", "公司", "企业", "公司名称")
_TITLE_COLS = ("title", "岗位", "职位", "岗位名称", "职位名称")
_URL_COLS = ("canonical_url", "url", "link", "链接", "网址")
_JOBID_COLS = ("job_id", "id", "编号")
_SOURCE_JOBID_COLS = ("source_job_id", "external_id", "来源id")


def _pick(headers: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """在表头里按候选列名（大小写不敏感）挑第一个命中的真实列名。"""
    lower = {h.strip().lower(): h for h in headers if isinstance(h, str)}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _v(row: dict, col: Optional[str]) -> Optional[str]:
    if not col:
        return None
    v = row.get(col)
    return v.strip() if isinstance(v, str) and v.strip() else None


class CSVConnector(Connector):
    """从 CSV 文件批量产出岗位。默认 utf-8-sig 编码（兼容 Excel 导出的 BOM）。"""

    source_name = SOURCE_CSV

    def __init__(self, path: str, *, source: str = SOURCE_CSV,
                 text_column: Optional[str] = None, encoding: str = "utf-8-sig"):
        self.path = path
        self.source = source
        self.text_column = text_column
        self.encoding = encoding

    def fetch(self) -> Iterator[RawJobPayload]:
        with open(self.path, "r", newline="", encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            text_col = self.text_column or _pick(headers, _TEXT_COLS)
            if not text_col:
                raise ValueError(
                    f"CSVConnector：未能识别 JD 正文列（表头={headers}）；"
                    f"请用 text_column 显式指定。"
                )
            company_col = _pick(headers, _COMPANY_COLS)
            title_col = _pick(headers, _TITLE_COLS)
            url_col = _pick(headers, _URL_COLS)
            jobid_col = _pick(headers, _JOBID_COLS)
            srcid_col = _pick(headers, _SOURCE_JOBID_COLS)

            for row in reader:
                raw = (row.get(text_col) or "").strip()
                if not raw:
                    continue
                yield RawJobPayload(
                    source=self.source,
                    raw_jd=raw,
                    job_id=_v(row, jobid_col),          # 有 id 列 → 身份模式；否则全量去重
                    company=_v(row, company_col),
                    title=_v(row, title_col),
                    canonical_url=_v(row, url_col),
                    source_job_id=_v(row, srcid_col),
                )
