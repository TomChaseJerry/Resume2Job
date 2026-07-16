# -*- coding: utf-8 -*-
"""ingest/connectors/base.py — 数据接入器抽象基类。

Connector 的唯一职责：把**某一种来源**的原始岗位数据转成统一的 RawJobPayload 流。
它**不**负责解析 / 归一 / 去重 / 入库——那些由 ingest.lifecycle 统一完成。这样换数据源时
只改 / 加一个 Connector，解析、技能抽取、去重、索引、推荐链路都不用动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from resume2job.ingest.models import RawJobPayload


class Connector(ABC):
    """数据接入器抽象基类：实现 fetch() 产出 RawJobPayload 流即可。"""

    #: 该来源的默认 source 标识（写入 jobs.source，供曝光 / 质量审计按来源分组）
    source_name: str = "unknown_source"

    @abstractmethod
    def fetch(self) -> Iterable[RawJobPayload]:
        """产出本来源的全部岗位（惰性 / 批量均可）。每条都是规范化前的原始信封。"""
        raise NotImplementedError


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """读取文本文件；失败返回空串（由调用方决定跳过 / 报告）。"""
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        print(f"[connector] 文件读取失败：{path}，原因：{e}")
        return ""
