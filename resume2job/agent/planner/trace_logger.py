# -*- coding: utf-8 -*-
"""Planner 决策 trace 落库（后训练数据闭环的第一步：只写不训）。

每轮把 (上下文输入 → planner 语义输出 → 执行计划 → 是否澄清) 落成一条样本，
存 SQLite 表 planner_traces，同时追加到 jsonl，便于日后导出做意图分类 / 规划的
SFT / 评测。第一阶段只采集，不消费。任何失败静默（trace 不应影响主流程）。
"""

import os
import json
import sqlite3
from datetime import datetime, timezone

from resume2job.storage.paths import SQLITE_PATH, DATA_DIR

DB_PATH = SQLITE_PATH
JSONL_PATH = os.path.join(DATA_DIR, "planner_traces.jsonl")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS planner_traces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    created_at  TEXT NOT NULL,
    user_query  TEXT,
    planner_output_json TEXT,
    plan_json   TEXT,
    decided_by  TEXT,          -- llm / rule_fallback
    clarified   INTEGER NOT NULL DEFAULT 0
);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(session_id: str, user_query: str, planner_output, plan: dict,
        decided_by: str) -> None:
    """落一条 planner 决策样本。失败静默。"""
    try:
        out_json = (planner_output.model_dump() if hasattr(planner_output, "model_dump")
                    else dict(planner_output or {}))
        record = {
            "session_id": session_id,
            "created_at": _utc_now_iso(),
            "user_query": user_query,
            "planner_output": out_json,
            "plan": dict(plan or {}),
            "decided_by": decided_by,
            "clarified": bool((plan or {}).get("clarify")),
        }
    except Exception:
        return

    # 1) SQLite
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_CREATE_SQL)
            conn.execute(
                "INSERT INTO planner_traces"
                "(session_id, created_at, user_query, planner_output_json, plan_json, decided_by, clarified)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, record["created_at"], user_query,
                 json.dumps(out_json, ensure_ascii=False),
                 json.dumps(record["plan"], ensure_ascii=False),
                 decided_by, int(record["clarified"])),
            )
            conn.commit()
    except Exception:
        pass

    # 2) jsonl（便于导出训练）
    try:
        os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
