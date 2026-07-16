# -*- coding: utf-8 -*-
"""ingest.connectors —— 各来源数据接入器（统一产出 RawJobPayload）。

实现：LocalFileConnector（txt/md/json）、CSVConnector、UserPasteConnector。
占位：OfficialCareerConnector（Phase 2）。
"""

from resume2job.ingest.connectors.base import Connector, read_text_file
from resume2job.ingest.connectors.local_file import LocalFileConnector
from resume2job.ingest.connectors.csv_file import CSVConnector
from resume2job.ingest.connectors.user_paste import UserPasteConnector, payload_from_text
from resume2job.ingest.connectors.official_career import OfficialCareerConnector

__all__ = [
    "Connector",
    "read_text_file",
    "LocalFileConnector",
    "CSVConnector",
    "UserPasteConnector",
    "payload_from_text",
    "OfficialCareerConnector",
]
