# -*- coding: utf-8 -*-
"""会话短期状态存储（session_state）。

存「上一轮结果摘要 last_results、活跃约束、活跃 resume/jd 引用」等跨轮短期状态，
让多轮指代（『第二个』『换上海』）能基于结构化上下文稳定解析，而非把整段对话塞 prompt。

三级回退（按可用性自动降级，保证无 Redis 的开发机照常跑）：
    1. Redis（生产首选，带 TTL）——import / 连接失败则降级；
    2. SQLite 表 session_state（持久化，与项目其它表同库）；
    3. 进程内 dict（最后兜底，重启即失）。

对外只暴露 get_session(sid) / set_session(sid, payload)。payload 为可 JSON 序列化的 dict。
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from resume2job.core import config
from resume2job.storage.paths import SQLITE_PATH

DB_PATH = SQLITE_PATH  # 模块级，便于验收脚本 monkeypatch 到隔离目录

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS session_state (
    session_id  TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

# 进程内兜底
_MEM: dict = {}

# Redis 客户端：None=未初始化，False=不可用（已降级），其余=可用客户端
_redis = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_redis():
    """惰性初始化 Redis；import 缺失或连接失败则永久降级（返回 None）。"""
    global _redis
    if _redis is not None:
        return _redis or None
    try:
        import redis  # 延迟导入，未安装则降级
        client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
        client.ping()
        _redis = client
    except Exception:
        _redis = False  # 标记不可用，后续不再重试
    return _redis or None


def _init_sqlite() -> None:
    import os
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_SQL)
        conn.commit()


def get_session(session_id: str) -> dict:
    """读会话短期状态；不存在返回 {}。"""
    if not session_id:
        return {}
    # 1) Redis
    client = _get_redis()
    if client is not None:
        try:
            raw = client.get(f"r2j:session:{session_id}")
            return json.loads(raw) if raw else {}
        except Exception:
            pass  # 单次失败不致命，落到 SQLite
    # 2) SQLite
    try:
        _init_sqlite()
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT payload_json FROM session_state WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    # 3) 内存
    return dict(_MEM.get(session_id) or {})


def set_session(session_id: str, payload: dict) -> None:
    """写会话短期状态（整体覆盖）。失败静默（短期状态丢失不应中断主流程）。"""
    if not session_id:
        return
    data = json.dumps(payload or {}, ensure_ascii=False)
    client = _get_redis()
    if client is not None:
        try:
            client.set(f"r2j:session:{session_id}", data, ex=config.SESSION_TTL)
            return
        except Exception:
            pass
    try:
        _init_sqlite()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO session_state(session_id, payload_json, updated_at) "
                "VALUES (?, ?, ?)",
                (session_id, data, _utc_now_iso()),
            )
            conn.commit()
        return
    except Exception:
        pass
    _MEM[session_id] = payload or {}
