"""
storage/conversation_store.py

多轮对话记忆——对话记录持久化（工程层面的会话历史）。

与 profile_cache（记住"你是谁"）配合，conversation_store 记住"我们聊过什么"：
    - 每一轮把 user / assistant 两条消息按时间写入 SQLite；
    - 下一轮按 session_id 读回最近若干条历史，作为意图识别的**兜底上下文**参考
      （结构化会话态 last_results/candidate_pool 缺失时才用，且只取前若干字喂 planner）。

与 user_profiles / jobs 共用同一个 data/resume2job.db（见 storage/paths.py）。
"""

import sqlite3
from datetime import datetime, timezone

from resume2job.storage.paths import SQLITE_PATH, ensure_data_dir

DB_PATH = SQLITE_PATH

# 默认会话 id（当前不做多用户/多会话登录，预留字段供后续扩展）
DEFAULT_SESSION_ID = "default_session"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,           -- user / assistant
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""
_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id, id);
"""


def _utc_now_iso() -> str:
    """当前 UTC 时间，ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """建 conversations 表与索引。幂等；失败仅打印警告，不抛异常。"""
    try:
        ensure_data_dir()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.commit()
    except Exception as e:
        print(f"[conversation_store] 警告：初始化 conversations 表失败：{e}")


def append_message(session_id: str, role: str, content: str) -> None:
    """追加一条消息。content 为空则跳过；失败仅打印警告，不抛异常。"""
    content = (content or "").strip()
    if not content:
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id or DEFAULT_SESSION_ID, role, content, _utc_now_iso()),
            )
            conn.commit()
    except Exception as e:
        print(f"[conversation_store] 警告：写入消息失败（session={session_id}）：{e}")


def append_turn(session_id: str, user_query: str, assistant_response: str) -> None:
    """便捷写入一轮对话：先 user 后 assistant。"""
    append_message(session_id, "user", user_query)
    append_message(session_id, "assistant", assistant_response)


def load_history(session_id: str, limit: int = 20) -> list:
    """按时间正序返回该 session 最近 limit 条消息：[{role, content, created_at}]。

    取最近 limit 条后再正序排列，保证返回的是「最近的一段对话、按时间先后」。
    失败或无记录返回空列表。
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT role, content, created_at FROM conversations "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id or DEFAULT_SESSION_ID, int(limit)),
            )
            rows = cur.fetchall()
        return [dict(r) for r in reversed(rows)]
    except Exception as e:
        print(f"[conversation_store] 警告：读取历史失败（session={session_id}）：{e}")
        return []


def list_sessions() -> list:
    """返回所有 session_id（按最近活跃时间倒序）。"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "SELECT session_id, MAX(id) AS last FROM conversations "
                "GROUP BY session_id ORDER BY last DESC"
            )
            return [r[0] for r in cur.fetchall()]
    except Exception as e:
        print(f"[conversation_store] 警告：列出会话失败：{e}")
        return []


def clear_session(session_id: str) -> None:
    """删除某个会话的全部对话记录（调试用）。"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM conversations WHERE session_id = ?",
                         (session_id or DEFAULT_SESSION_ID,))
            conn.commit()
    except Exception as e:
        print(f"[conversation_store] 警告：清空会话失败（session={session_id}）：{e}")


# 模块首次导入时建表（幂等）
init_db()
