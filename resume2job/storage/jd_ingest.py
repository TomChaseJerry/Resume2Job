# -*- coding: utf-8 -*-
"""
storage/jd_ingest.py

JD 自动入库节点（LangGraph，挂在 jd_input 之后）。

定位：用户粘贴一条 JD 后、在匹配评分之前，判断该 JD 是否已在岗位库；不在则入库（SQLite 事实源 +
Chroma 索引），已在则复用。**入库的全部逻辑（去重 / 质量校验 / 版本戳 / 写库 / 写向量）已统一下沉到
ingest.lifecycle.ingest_record**——本节点只做「取出 jd_input 已解析的画像 → 调接入层 → 把结果写回
State」的薄封装，与批量建库（retrieval.indexer.index_jobs）共用**同一条接入路径**，避免两路写入漂移。

去重仍是两层（在 lifecycle 内）：精确（公司 + 标题 + 原文哈希）→ 语义（向量近邻 > 阈值）。
"""

from resume2job.agent.state import AgentState


def _append_error(state: AgentState, message: str) -> AgentState:
    """把一条错误信息安全追加到 errors（不就地修改原列表），返回浅拷贝 State。"""
    new_state = dict(state)
    errors = list(state.get("errors") or [])
    errors.append(message)
    new_state["errors"] = errors
    return new_state


def jd_ingest_node(state: AgentState) -> AgentState:
    """JD 自动入库节点：委托 ingest.lifecycle 完成去重 / 校验 / 写库 / 写向量。

    输入：state["jd_text"]（用户粘贴原文）+ state["jd_profiles"][0]（jd_input 已解析的画像）。
    输出：ingested_job_id / jd_is_duplicate；任何异常都被捕获并追加到 errors，流程不中断；
    失败时 ingested_job_id 保持 ""、jd_is_duplicate 保持 False。
    """
    new_state = dict(state)
    new_state.setdefault("ingested_job_id", "")
    new_state.setdefault("jd_is_duplicate", False)

    # 仅在「用户粘贴 JD」链路上工作；推荐链路（无 jd_text）静默跳过
    jd_text = state.get("jd_text") or ""
    if not jd_text.strip():
        return state

    print("[jd_ingest_node] 开始执行...")

    # 复用 jd_input_node 已解析的画像（跨链路一致 + 省 token），缺失则放弃入库
    jd_profiles = state.get("jd_profiles") or []
    jd_profile = jd_profiles[0] if jd_profiles and isinstance(jd_profiles[0], dict) else {}
    if not jd_profile:
        print("[jd_ingest] 警告：缺少结构化 jd_profile，跳过入库")
        return _append_error(new_state, "jd_ingest_node: 缺少结构化 jd_profile，跳过入库")

    # 惰性导入接入层，避免模块加载期触发 Chroma / 循环依赖
    from resume2job.ingest.connectors.user_paste import payload_from_text
    from resume2job.ingest.lifecycle import ingest_record

    payload = payload_from_text(jd_text, company=jd_profile.get("company"),
                                title=jd_profile.get("title"))
    try:
        res = ingest_record(payload, jd_profile=jd_profile)  # 无 job_id → 全量去重模式
    except Exception as e:
        print(f"[jd_ingest] 错误：入库失败：{e}")
        return _append_error(new_state, f"jd_ingest_node: 入库失败：{e}")

    if res.job_id:
        new_state["ingested_job_id"] = res.job_id
    new_state["jd_is_duplicate"] = res.is_duplicate

    if res.action == "failed":
        print(f"[jd_ingest] 入库失败：{res.reason}")
        return _append_error(new_state, f"jd_ingest_node: 入库失败（{res.reason}）")
    if res.reason and res.reason.startswith("chroma_write_failed"):
        # SQLite 已入库；向量缺失可由 scripts/rebuild_index 补建
        new_state = _append_error(
            new_state,
            f"jd_ingest_node: 写入 Chroma 失败（SQLite 已入库，可用 rebuild_index 补建向量）：{res.reason}",
        )

    dup_note = "（重复，复用）" if res.is_duplicate else ""
    print(f"[jd_ingest] {res.action} job_id={res.job_id}{dup_note}，similarity={res.similarity}")
    return new_state
