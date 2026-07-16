# -*- coding: utf-8 -*-
"""resume2job.observability —— 请求级链路可观测性（Stage 2）。

回答三个问题：
    1. 这次推荐到底经历了什么？        → events.py（request_id + 各阶段候选/分数/工具/token/时延快照）
    2. 升级前后同一请求结果怎么变？      → replay.py（回放历史请求 + diff）
    3. 拿日志做训练/展示会泄露 PII 吗？  → redaction.py（脱敏：保留技能/教育/偏好，移除身份信息）

注入方式：ContextVar 记录器。run_turn 用 events.request_scope 开 trace；core/llm 包一层记 token+时延；
节点经 events.run_node 记每节点时延。无活跃 trace 时所有记录 no-op（pipeline / eval / 直接调用不受影响）。
"""

from resume2job.observability import events, redaction, replay
from resume2job.observability.events import (
    RequestTrace, request_scope, current, record_feedback, get_trace, load_traces,
)

__all__ = [
    "events", "redaction", "replay",
    "RequestTrace", "request_scope", "current", "record_feedback", "get_trace", "load_traces",
]
