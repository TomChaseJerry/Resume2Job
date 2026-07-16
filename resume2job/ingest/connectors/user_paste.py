# -*- coding: utf-8 -*-
"""ingest/connectors/user_paste.py — 用户粘贴 JD 接入器。

把聊天框里用户粘贴的一段 JD 原文包成一条 RawJobPayload。不带 job_id（→ 全量去重模式：
先精确去重再语义去重，确认是新岗位才生成 id），与 storage/jd_ingest 的运行时入库语义一致。
"""

from __future__ import annotations

from typing import Iterator, Optional

from resume2job.ingest.connectors.base import Connector
from resume2job.ingest.models import RawJobPayload, SOURCE_USER


def payload_from_text(jd_text: str, *, company: Optional[str] = None, title: Optional[str] = None,
                      canonical_url: Optional[str] = None, source_job_id: Optional[str] = None,
                      source: str = SOURCE_USER) -> RawJobPayload:
    """便捷构造：单段粘贴文本 → RawJobPayload（job_id 留空，走全量去重）。"""
    return RawJobPayload(source=source, raw_jd=jd_text or "", job_id=None,
                         company=company, title=title,
                         canonical_url=canonical_url, source_job_id=source_job_id)


class UserPasteConnector(Connector):
    """单条粘贴 JD 的接入器（fetch 产出 0 或 1 条）。"""

    source_name = SOURCE_USER

    def __init__(self, jd_text: str, *, company: Optional[str] = None, title: Optional[str] = None,
                 canonical_url: Optional[str] = None, source_job_id: Optional[str] = None,
                 source: str = SOURCE_USER):
        self.jd_text = jd_text or ""
        self.company = company
        self.title = title
        self.canonical_url = canonical_url
        self.source_job_id = source_job_id
        self.source = source

    def fetch(self) -> Iterator[RawJobPayload]:
        if self.jd_text.strip():
            yield payload_from_text(self.jd_text, company=self.company, title=self.title,
                                    canonical_url=self.canonical_url,
                                    source_job_id=self.source_job_id, source=self.source)
