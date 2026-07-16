# -*- coding: utf-8 -*-
"""resume2job.ingest —— 岗位数据接入与生命周期管理。

把「岗位数据从哪来、如何更新、是否过期、如何进入索引」从检索 / 评分链路里抽出来，独立成层：

    采集（connectors）→ 规范化（normalizer）→ 校验（validator）→ 去重 / 更新 / 索引 / 下线（lifecycle）

数据结构（models）：RawJobPayload（来源契约）/ JobRecord（标准记录）/ IngestResult（接入结果）。
版本戳（versions）：parser / index_text / embedding 三类版本，解决「SQLite 新、Chroma 旧、BM25 旧」漂移。

轻量契约（models / connectors / validator / normalizer / versions）从本包直接导入；
接入编排（会拉起 Chroma）请从子模块导入：`from resume2job.ingest.lifecycle import ingest_record`。
"""

from resume2job.ingest.models import (
    RawJobPayload,
    IngestResult,
    JobRecord,
    STATUS_ACTIVE,
    STATUS_EXPIRED,
    STATUS_REMOVED,
    SOURCE_BATCH,
    SOURCE_USER,
    SOURCE_CSV,
    SOURCE_OFFICIAL,
)
from resume2job.ingest.validator import validate_job, QualityReport
from resume2job.ingest.normalizer import normalize_raw_payload, clean_text
from resume2job.ingest import versions
from resume2job.ingest.connectors import (
    Connector,
    LocalFileConnector,
    CSVConnector,
    UserPasteConnector,
    OfficialCareerConnector,
)

__all__ = [
    # 数据结构
    "RawJobPayload", "IngestResult", "JobRecord", "QualityReport",
    "STATUS_ACTIVE", "STATUS_EXPIRED", "STATUS_REMOVED",
    "SOURCE_BATCH", "SOURCE_USER", "SOURCE_CSV", "SOURCE_OFFICIAL",
    # 校验 / 归一 / 版本
    "validate_job", "normalize_raw_payload", "clean_text", "versions",
    # 接入器
    "Connector", "LocalFileConnector", "CSVConnector", "UserPasteConnector", "OfficialCareerConnector",
]
