"""
storage/profile_cache.py

多轮上下文记忆与用户画像缓存模块（工程层面的会话状态管理 + 用户画像持久化）。

定位说明（面试口径）：
    这里的记忆不是模型层面的长期记忆，而是工程上的会话状态管理与用户画像缓存。
    它用于减少重复上传简历，并支持跨轮对话复用用户偏好。

职责：
    1. 用户上传简历后，把解析出的结构化画像持久化到 SQLite（每个用户最多保留一份，新传覆盖旧的）；
    2. 后续对话未上传简历时，自动从数据库加载已有画像；
    3. 用户仅修改查询条件时，复用已有画像（本节点只负责画像，不动检索参数）；
    4. 用户重新上传简历时，先删除旧画像再写入新画像（始终只有一份）。
"""

# ===== 1. import 区 =====
import os
import json
import sqlite3
from datetime import datetime, timezone

from resume2job.agent.state import AgentState
# 统一存储路径（单一事实来源）：与 jobs 表、对话记录共用同一个 SQLite 文件
from resume2job.storage.paths import SQLITE_PATH


# ===== 2. 常量定义 =====
DB_PATH = SQLITE_PATH

# 当前版本不做多用户认证，user_id 固定为默认值（字段预留供后续扩展）
DEFAULT_USER_ID = "default_user"

# 建表与索引 SQL
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    profile_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    resume_text   TEXT NOT NULL,
    structured_profile TEXT NOT NULL,  -- JSON 字符串
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1  -- 1=active, 0=inactive
);
"""
_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _utc_now_iso() -> str:
    """当前 UTC 时间，ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _get_conn() -> sqlite3.Connection:
    """获取 SQLite 连接；确保 data/ 目录存在，行结果以 sqlite3.Row 返回。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _append_error(state: AgentState, message: str) -> AgentState:
    """把一条错误信息安全追加到 errors（不就地修改原列表），返回浅拷贝 State。"""
    new_state = dict(state)
    errors = list(state.get("errors") or [])
    errors.append(message)
    new_state["errors"] = errors
    return new_state


# ===== 3. init_db =====
def init_db() -> None:
    """执行建表与建索引 SQL。幂等，可重复调用；失败仅打印警告，不抛异常。"""
    try:
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute(_CREATE_TABLE_SQL)
            cur.execute(_CREATE_INDEX_SQL)
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[profile_cache] 警告：初始化数据库失败：{e}")


# ===== 5. delete_profiles（先定义，供 save_profile 调用）=====
def delete_profiles(user_id: str) -> None:
    """删除该用户的所有画像记录（实现「每个用户最多保留一份画像」）。

    save_profile 在写入新画像前调用本函数，确保表中始终只有当前这一份。
    失败仅打印警告，不抛异常。
    """
    try:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[profile_cache] 警告：清除旧画像失败（user_id={user_id}）：{e}")


# ===== 4. save_profile =====
def save_profile(user_id: str, resume_text: str, structured_profile: dict) -> str:
    """保存画像，返回新生成的 profile_id。

    采用「最多保留一份」策略：先删除该用户的所有现有记录，再写入这一份新的，
    因此每个 user_id 在表中始终只有一行。
    created_at / updated_at 同为当前 UTC 时间（ISO 8601）。
    """
    # 1. 删除旧画像（每个用户最多保留一份）
    delete_profiles(user_id)

    # 2. 生成新 profile_id（timestamp 精确到秒）
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d%H%M%S")
    base_profile_id = f"profile_{user_id}_{timestamp}"

    # 3. 序列化结构化画像
    structured_json = json.dumps(structured_profile or {}, ensure_ascii=False)
    now_iso = now.isoformat()

    # 4. 写入数据库；同一秒内重复保存时追加后缀避免主键冲突丢数据
    conn = _get_conn()
    try:
        profile_id = base_profile_id
        suffix = 1
        while True:
            try:
                conn.execute(
                    "INSERT INTO user_profiles "
                    "(profile_id, user_id, resume_text, structured_profile, created_at, updated_at, active) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (profile_id, user_id, resume_text or "", structured_json, now_iso, now_iso),
                )
                conn.commit()
                break
            except sqlite3.IntegrityError:
                suffix += 1
                profile_id = f"{base_profile_id}_{suffix}"
    finally:
        conn.close()

    # 5. 返回新 profile_id
    return profile_id


# ===== 6. load_latest_profile =====
def load_latest_profile(user_id: str):
    """加载该用户的画像（每个用户最多一份，按 updated_at 倒序取首条）。

    找到：返回完整行数据 dict，其中 structured_profile 已反序列化为 dict；
    未找到或异常：返回 None。
    """
    try:
        conn = _get_conn()
        try:
            cur = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        record = dict(row)
        try:
            record["structured_profile"] = json.loads(record["structured_profile"])
        except (json.JSONDecodeError, TypeError):
            record["structured_profile"] = {}
        return record
    except Exception as e:
        print(f"[profile_cache] 警告：加载缓存画像失败（user_id={user_id}）：{e}")
        return None


# ===== 7. profile_cache_node（LangGraph 节点）=====
def profile_cache_node(state: AgentState) -> AgentState:
    """会话画像缓存节点：挂在 intent_router 之后、resume_router 之前。

    情况 A / C：本轮上传了新简历（need_resume_parse=True 且 resume_profile 非空）
        -> 调 save_profile 持久化（内部先 delete_profiles 清掉旧画像，保证最多一份），
           写入 profile_id，profile_source 置 "new_upload"。
    情况 B：本轮未上传（need_resume_parse=False 或 resume_profile 为空）
        -> 加载最近缓存：命中则回填 resume_profile / profile_id，profile_source 置 "cached"，
           并把 plan.need_resume_parse 置 False；未命中则追加错误，交下游处理。
    所有数据库操作异常都被捕获，错误追加到 state["errors"]，流程不中断。
    """
    new_state = dict(state)  # 浅拷贝，不修改原有字段

    plan = state.get("plan") or {}
    need_resume_parse = bool(plan.get("need_resume_parse"))
    resume_profile = state.get("resume_profile") or {}

    # 情况 A / C：本轮上传了新简历（含「重新上传」，处理方式一致）
    if need_resume_parse and resume_profile:
        try:
            # 兼容两种结构：resume_profile 自带 resume_text/structured_profile 包裹，
            # 或 resume_profile 本身即结构化画像。
            resume_text = resume_profile.get("resume_text", "") or ""
            structured_profile = resume_profile.get("structured_profile")
            if not isinstance(structured_profile, dict):
                structured_profile = resume_profile

            profile_id = save_profile(DEFAULT_USER_ID, resume_text, structured_profile)
            new_state["profile_id"] = profile_id
            new_state["profile_source"] = "new_upload"
            print(f"[profile_cache] 新简历已保存，profile_id={profile_id}")
        except Exception as e:
            print(f"[profile_cache] 错误：保存新简历失败：{e}")
            new_state = _append_error(new_state, f"profile_cache_node: 保存新简历失败：{e}")
        return new_state

    # 情况 B：本轮未上传简历，尝试复用最近缓存画像
    try:
        cached_row = load_latest_profile(DEFAULT_USER_ID)
        if cached_row:
            new_state["resume_profile"] = cached_row["structured_profile"]
            new_state["profile_id"] = cached_row["profile_id"]
            new_state["profile_source"] = "cached"
            # 复用画像后跳过重复解析（拷贝 plan，避免修改原 dict）
            new_plan = dict(plan)
            new_plan["need_resume_parse"] = False
            new_state["plan"] = new_plan
            print(f"[profile_cache] 复用缓存画像，profile_id={cached_row['profile_id']}")
        else:
            print("[profile_cache] 警告：未找到缓存画像，请上传简历")
            new_state = _append_error(
                new_state, "profile_cache_node: 未找到缓存画像，请上传简历"
            )
    except Exception as e:
        print(f"[profile_cache] 错误：加载缓存画像失败：{e}")
        new_state = _append_error(new_state, f"profile_cache_node: 加载缓存画像失败：{e}")

    return new_state


# 模块首次导入时建表（幂等）
init_db()
