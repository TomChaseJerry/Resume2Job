# -*- coding: utf-8 -*-
"""ingest/connectors/official_career.py — 官方招聘页增量同步接入器（Phase 1 占位）。

设计意图（留待 Phase 2 落地）：对**已获授权访问**的官方招聘页做增量同步——分页抓取、
详情解析、按 canonical_url / source_job_id 去重、按更新时间 / ETag 做增量。BOSS 等第三方
平台同理，作为未来 connector 接入；二者都只需实现 fetch() 产出 RawJobPayload，下游解析 /
去重 / 索引 / 推荐链路无需改动。

**当前阶段不实现任何抓取逻辑**：调用 fetch() 会抛 NotImplementedError，请改用
LocalFileConnector / CSVConnector / UserPasteConnector 接入合法来源的数据。
"""

from __future__ import annotations

from typing import Iterable

from resume2job.ingest.connectors.base import Connector
from resume2job.ingest.models import RawJobPayload, SOURCE_OFFICIAL


class OfficialCareerConnector(Connector):
    """官方招聘页增量同步（占位，Phase 2 实现）。"""

    source_name = SOURCE_OFFICIAL

    def __init__(self, base_url: str, *, authorized: bool = False, source: str = SOURCE_OFFICIAL):
        self.base_url = base_url
        self.authorized = authorized
        self.source = source

    def fetch(self) -> Iterable[RawJobPayload]:
        raise NotImplementedError(
            "OfficialCareerConnector 计划在 Phase 2 实现（授权页增量同步）；"
            "当前请使用 LocalFileConnector / CSVConnector / UserPasteConnector。"
        )
