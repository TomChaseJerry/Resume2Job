# -*- coding: utf-8 -*-
"""ingest/connectors/local_file.py — 本地文件接入器。

支持目录下的 .txt / .md（整文件即一条 JD 原文）与 .json（单对象或对象数组）。
.txt/.md 以**文件名 stem 作 job_id**（身份模式，幂等更新，沿用既有 JDs/*.txt 约定，
保证重复入库不产生新 id）；.json 可携带 company/title/source_job_id/canonical_url 等提示。
"""

from __future__ import annotations

import json
import os
from typing import Iterator

from resume2job.ingest.connectors.base import Connector, read_text_file
from resume2job.ingest.models import RawJobPayload, SOURCE_BATCH

_TEXT_EXTS = (".txt", ".md", ".markdown")
_JSON_EXTS = (".json",)
# .json 对象里「JD 正文」可能的键名（按优先级）
_JD_TEXT_KEYS = ("raw_jd", "jd_text", "jd", "text", "description", "content")


def _opt_str(v):
    return v.strip() if isinstance(v, str) and v.strip() else None


def _first_str(obj: dict, keys) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


class LocalFileConnector(Connector):
    """从本地目录读取 JD 文件。"""

    source_name = SOURCE_BATCH

    def __init__(self, folder: str, *, source: str = SOURCE_BATCH, recursive: bool = False):
        self.folder = folder
        self.source = source
        self.recursive = recursive

    def _iter_files(self) -> Iterator[str]:
        if self.recursive:
            for root, _, files in os.walk(self.folder):
                for f in sorted(files):
                    yield os.path.join(root, f)
        else:
            for f in sorted(os.listdir(self.folder)):
                p = os.path.join(self.folder, f)
                if os.path.isfile(p):
                    yield p

    def fetch(self) -> Iterator[RawJobPayload]:
        for path in self._iter_files():
            ext = os.path.splitext(path)[1].lower()
            stem = os.path.splitext(os.path.basename(path))[0]
            if ext in _TEXT_EXTS:
                text = read_text_file(path)
                if text and text.strip():
                    yield RawJobPayload(source=self.source, raw_jd=text, job_id=stem)
            elif ext in _JSON_EXTS:
                yield from self._from_json(path, stem)
            # 其它扩展名静默跳过

    def _from_json(self, path: str, stem: str) -> Iterator[RawJobPayload]:
        text = read_text_file(path)
        if not text.strip():
            return
        try:
            data = json.loads(text)
        except Exception:
            print(f"[LocalFileConnector] 跳过非法 JSON：{path}")
            return
        items = data if isinstance(data, list) else [data]
        multi = len(items) > 1
        for i, obj in enumerate(items):
            jid_suffix = f"{stem}_{i}" if multi else stem
            if isinstance(obj, str) and obj.strip():
                yield RawJobPayload(source=self.source, raw_jd=obj, job_id=jid_suffix)
                continue
            if not isinstance(obj, dict):
                continue
            raw = _first_str(obj, _JD_TEXT_KEYS)
            if not raw:
                continue
            jid = obj.get("job_id")
            yield RawJobPayload(
                source=self.source,
                raw_jd=raw,
                job_id=str(jid).strip() if isinstance(jid, (str, int)) and str(jid).strip() else jid_suffix,
                company=_opt_str(obj.get("company")),
                title=_opt_str(obj.get("title")),
                source_job_id=_opt_str(obj.get("source_job_id")),
                canonical_url=_opt_str(obj.get("canonical_url") or obj.get("url")),
                collected_at=_opt_str(obj.get("collected_at")),
            )
