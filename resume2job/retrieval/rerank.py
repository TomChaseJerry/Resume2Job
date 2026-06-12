# -*- coding: utf-8 -*-
"""Rerank 精排模块（召回 → 粗排 → 精排 三段式架构的最后一段）。

混合召回（BM25 + 向量 + RRF）解决「找得全」，rerank 用交叉编码器逐对计算
query–document 相关性，解决「排得准」。调用 DashScope 原生 text-rerank 接口
（gte-rerank 系列，OpenAI 兼容端点不覆盖该能力），与 commute 模块一致只用
标准库 urllib，不引入第三方 HTTP 依赖。

失败兜底：任何网络 / 配额 / 解析异常都打印告警并原样返回输入命中列表，
绝不让精排环节中断检索主链路。
"""

import json
import urllib.request
from typing import List

from resume2job.core import config
from resume2job.core.llm import get_api_key

_RERANK_PATH = "/services/rerank/text-rerank/text-rerank"
_TIMEOUT_SECONDS = 15
# 控制送精排的文档长度，防止超长 document 触发接口限制
_MAX_DOC_CHARS = 2000


def rerank_hits(query_text: str, hits: List[dict], top_n: int = None) -> List[dict]:
    """对召回命中做 rerank 精排，返回按 rerank_score 降序的新列表。

    - 每个命中的 document（入库 index_text）作为候选文档；
    - 成功时为每个命中写入 rerank_score 并重排；
    - 失败时返回原列表（保持 RRF / 向量排序），并打印告警。
    """
    if not hits or not isinstance(query_text, str) or not query_text.strip():
        return hits

    documents = [(h.get("document") or "")[:_MAX_DOC_CHARS] for h in hits]
    try:
        api_key = get_api_key()
    except RuntimeError as e:
        print(f"[WARN] rerank 跳过：{e}")
        return hits

    payload = {
        "model": config.RERANK_MODEL,
        "input": {"query": query_text[:_MAX_DOC_CHARS], "documents": documents},
        "parameters": {
            "return_documents": False,
            "top_n": min(len(hits), top_n or len(hits)),
        },
    }

    try:
        req = urllib.request.Request(
            config.DASHSCOPE_NATIVE_URL + _RERANK_PATH,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[WARN] rerank 调用失败，保持原排序：{e}")
        return hits

    results = ((data.get("output") or {}).get("results")) or []
    if not results:
        print("[WARN] rerank 返回空结果，保持原排序。")
        return hits

    reranked = []
    for item in results:
        idx = item.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(hits)):
            continue
        hit = dict(hits[idx])
        hit["rerank_score"] = float(item.get("relevance_score") or 0.0)
        reranked.append(hit)

    if not reranked:
        return hits
    reranked.sort(key=lambda h: h["rerank_score"], reverse=True)
    return reranked
