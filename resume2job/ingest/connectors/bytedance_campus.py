# -*- coding: utf-8 -*-
"""字节跳动校园招聘官网岗位接入器。

只读取官网公开展示的校园正式岗位（recruitment_id=201），不访问登录、简历或投递接口。
官网岗位列表接口是前端内部 XHR，并非有稳定性承诺的开放 API；因此本接入器把超时、
分页上限和请求间隔都暴露为参数，接口变化时让异常显式上抛，由调用方决定降级策略。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from typing import Iterator, Optional

from resume2job.ingest.connectors.base import Connector
from resume2job.ingest.models import RawJobPayload


class ByteDanceCampusConnector(Connector):
    """从字节招聘官网读取校园正式岗位，统一转换为 ``RawJobPayload``。"""

    source_name = "bytedance_campus"
    base_url = "https://jobs.bytedance.com"
    positions_url = f"{base_url}/campus/position"
    csrf_url = f"{base_url}/api/v1/csrf/token"
    search_url = f"{base_url}/api/v1/search/job/posts"
    recruitment_id = "201"  # 校园正式；202 是实习，不在本接入器当前范围内

    def __init__(
        self,
        *,
        keyword: str = "",
        page_size: int = 50,
        max_pages: Optional[int] = None,
        request_interval: float = 0.5,
        timeout: float = 20.0,
    ):
        if not 1 <= page_size <= 100:
            raise ValueError("page_size 必须在 1..100 之间")
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages 必须为正整数或 None")
        if request_interval < 0:
            raise ValueError("request_interval 不能为负数")
        self.keyword = keyword.strip()
        self.page_size = page_size
        self.max_pages = max_pages
        self.request_interval = request_interval
        self.timeout = timeout
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar())
        )

    @property
    def _headers(self) -> dict:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": self.base_url,
            "Referer": self.positions_url,
            "portal-channel": "campus",
            "portal-platform": "pc",
            "website-path": "campus",
        }

    def _request_json(self, url: str, *, body: Optional[dict] = None,
                      extra_headers: Optional[dict] = None) -> dict:
        headers = {**self._headers, **(extra_headers or {})}
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(
                f"字节招聘接口 HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"字节招聘接口请求失败: {exc}") from exc
        if payload.get("code") != 0:
            raise RuntimeError(
                f"字节招聘接口返回错误 code={payload.get('code')}: "
                f"{payload.get('message') or payload.get('error') or 'unknown error'}"
            )
        return payload

    def _get_csrf_token(self) -> str:
        # 先访问职位页建立与官网前端一致的匿名会话，再领取短期 CSRF token。
        page_request = urllib.request.Request(
            self.positions_url,
            headers=self._headers,
            method="GET",
        )
        try:
            with self._opener.open(page_request, timeout=self.timeout):
                pass
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法访问字节校园招聘职位页: {exc}") from exc
        payload = self._request_json(self.csrf_url, body={})
        token = (payload.get("data") or {}).get("token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("字节招聘 CSRF 接口未返回 token")
        return token

    def _search_page(self, token: str, offset: int) -> tuple[list, int]:
        body = {
            "keyword": self.keyword,
            "limit": self.page_size,
            "offset": offset,
            "portal_type": 3,
            "portal_entrance": 1,
            "language": "zh",
            "recruitment_id_list": [self.recruitment_id],
            "job_category_id_list": [],
            "location_code_list": [],
            "subject_id_list": [],
            "tag_id_list": [],
            "storefront_id_list": [],
            "job_function_id_list": [],
        }
        payload = self._request_json(
            self.search_url,
            body=body,
            extra_headers={"x-csrf-token": token},
        )
        data = payload.get("data") or {}
        posts = data.get("job_post_list") or []
        return posts if isinstance(posts, list) else [], int(data.get("count") or 0)

    @staticmethod
    def _name(value) -> str:
        return value.get("name", "").strip() if isinstance(value, dict) else ""

    def _to_payload(self, item: dict, collected_at: str) -> Optional[RawJobPayload]:
        source_id = str(item.get("id") or "").strip()
        title = str(item.get("title") or "").strip()
        if not source_id or not title:
            return None
        description = str(item.get("description") or "").strip()
        requirement = str(item.get("requirement") or "").strip()
        cities = [self._name(x) for x in (item.get("city_list") or [])]
        cities = [x for x in cities if x]
        if not cities:
            city = self._name(item.get("city_info"))
            cities = [city] if city else []
        category = self._name(item.get("job_category"))
        raw_jd = "\n\n".join(
            x for x in (
                f"公司：字节跳动\n岗位：{title}\n岗位类型：校园招聘正式岗位"
                + (f"\n工作城市：{'、'.join(cities)}" if cities else "")
                + (f"\n职位类别：{category}" if category else "")
                + f"\n岗位链接：{self.positions_url}/{source_id}/detail",
                f"岗位职责：\n{description}" if description else "",
                f"任职要求：\n{requirement}" if requirement else "",
            ) if x
        )
        return RawJobPayload(
            source=self.source_name,
            raw_jd=raw_jd,
            job_id=f"bytedance_campus_{source_id}",
            company="字节跳动",
            title=title,
            source_job_id=source_id,
            canonical_url=f"{self.positions_url}/{source_id}/detail",
            collected_at=collected_at,
            extra={
                "recruitment_id": self.recruitment_id,
                "cities": cities,
                "job_category": category,
                "publish_time": item.get("publish_time"),
            },
        )

    def fetch(self) -> Iterator[RawJobPayload]:
        token = self._get_csrf_token()
        collected_at = datetime.now(timezone.utc).isoformat()
        offset = 0
        page = 0
        total = None
        while total is None or offset < total:
            posts, total = self._search_page(token, offset)
            if not posts:
                break
            for item in posts:
                if isinstance(item, dict):
                    payload = self._to_payload(item, collected_at)
                    if payload is not None:
                        yield payload
            page += 1
            offset += len(posts)
            if self.max_pages is not None and page >= self.max_pages:
                break
            if len(posts) < self.page_size:
                break
            if self.request_interval:
                time.sleep(self.request_interval)
