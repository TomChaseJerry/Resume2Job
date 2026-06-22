# -*- coding: utf-8 -*-
"""编排：把语义（PlannerOutput）+ 会话状态确定性地映射成执行计划 dict（session_action + need_* 开关）。

不调用 LLM。核心是 `session_action`（会话动作）决策（解析建议 item2）：
    - 硬约束变化（城市/学历/岗位类型）       → RETRIEVE   ：重新召回 + 评分 + 报告；
    - 方向切换（本轮显式方向与池活跃方向不相交）→ RETRIEVE  ：旧池没有新方向岗位，只重排捞不出来，必须重召回；
    - 仅软偏好微调（同方向重排序/通勤）且有候选池→ REUSE_RERANK：复用池、仅重算偏好分并重排；
    - request_more 且有候选池                → NEXT_BATCH ：取池中下一批未展示岗位；
    - 用户粘贴 JD                            → USER_JD    ：解析 JD 并适配分析；
    - 指定 job_id 追问适配/详情              → SELECTED   ：读缓存岗位详情，不重检索/不重评分；
    - 指定 job_id 要学习计划/面试题          → ASSIST     ：读缓存岗位，跑 assist_actions。

need_* 开关供 route_job_source / executor / enhancements 读取；report_views 已废弃——
推荐 / JD 适配报告默认含「匹配点 + 技能缺口」，学习计划/面试题仅 ASSIST 驱动。
"""

from resume2job.agent.planner.schema import PlannerOutput


def _base_plan() -> dict:
    return {
        "session_action": "RETRIEVE",
        "reuse_pool": False,
        "next_batch": False,
        "need_resume_parse": True,
        "need_job_search": False,
        "need_jd_input": False,
        "need_jd_parse": False,
        "need_match_score": False,
        "need_recommendation": False,
        "need_skill_gap": False,
        "need_learning_plan": False,
        "need_interview": False,
        "need_commute": False,
        "clarify": False,
        "clarify_question": None,
    }


def _hard_changed(out: PlannerOutput, prev_hard: dict) -> bool:
    """本轮是否显式改了硬约束（城市/学历/岗位类型）——只有显式给出且与会话活跃值不同才算变化。"""
    hc = out.hard_constraints or {}
    prev = prev_hard or {}
    for k in ("city", "education", "job_type"):
        v = hc.get(k)
        if v and str(v).strip() and str(v).strip() != str(prev.get(k) or "").strip():
            return True
    return False


def _commute_enabled(out: PlannerOutput) -> bool:
    """有通勤诉求（时间上限 / 地址 / 显式 enabled）即视为启用。"""
    c = out.commute
    return bool(c and (c.enabled or c.max_minutes or c.home_address))


def _round_direction_tags(out: PlannerOutput) -> list:
    """本轮显式抽到的方向偏好标签（soft_preferences.direction_tags）。"""
    sp = out.soft_preferences or {}
    v = sp.get("direction_tags")
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _tags_overlap(a: list, b: list) -> bool:
    """两组方向标签是否有交集——大小写无关且容子串（『大模型应用』⊂『大模型应用算法』算重叠），
    避免同一方向的不同表面形式被误判为不相交（宁可少判 pivot，也不误判：误判 pivot 只是多召回一次，
    结果仍正确；漏判才会用错池）。"""
    la = [t.lower() for t in a if t]
    lb = [t.lower() for t in b if t]
    for x in la:
        for y in lb:
            if x == y or x in y or y in x:
                return True
    return False


def _direction_pivoted(out: PlannerOutput, sess: dict) -> bool:
    """本轮是否**切换了目标方向**：显式给出方向标签，且与候选池活跃方向标签基本不相交。

    为什么需要：旧候选池是按之前的方向召回的；用户改问全新方向（如「大模型应用→自动驾驶规划」）时
    城市/学历/类型可能都没变，但旧池里根本没有新方向的岗位，只重排（REUSE_RERANK）永远捞不出来，
    必须重召回。本轮未显式给方向 → 非切换（沿用既有复用/换一批逻辑）。
    注：active_direction_tags 跨轮累积（pool 的 Query3 同样累积），故回到之前提过的方向算重叠、
    仍走复用；只有**全新**方向才触发重召回。
    """
    this_round = _round_direction_tags(out)
    if not this_round:
        return False
    pool_tags = [str(t).strip() for t in (sess.get("active_direction_tags") or []) if str(t).strip()]
    if not pool_tags:
        return True   # 池无方向上下文、本轮新增明确方向 → 重召回更稳（旧池未针对该方向召回）
    return not _tags_overlap(this_round, pool_tags)


def build_plan(out: PlannerOutput, ctx, clarify_question: str = None) -> dict:
    """语义 + 会话状态 → 执行计划。clarify_question 非空时短路所有业务开关。"""
    plan = _base_plan()

    # 澄清优先：短路业务链路，图直达 END
    if clarify_question:
        plan["session_action"] = "CLARIFY"
        plan["need_resume_parse"] = False
        plan["clarify"] = True
        plan["clarify_question"] = clarify_question
        return plan

    sess = getattr(ctx, "session_state", None) or {}
    has_pool = bool(sess.get("candidate_pool"))
    prev_hard = sess.get("active_hard_constraints") or {}
    commute_on = _commute_enabled(out)

    # ---- 决定会话动作 ----
    if out.job_source == "USER_JD":
        action = "USER_JD"
    elif out.intent == "ASSIST":
        action = "ASSIST"
    elif out.job_source == "SELECTED":
        action = "SELECTED"
    else:  # RECOMMEND / RETRIEVE（首轮无池时 prev_hard 为空、has_pool 为假，落到末尾 RETRIEVE）
        if _hard_changed(out, prev_hard):
            action = "RETRIEVE"            # 硬约束变化 → 必须重召回
        elif has_pool and _direction_pivoted(out, sess):
            action = "RETRIEVE"            # 方向切换 → 旧池覆盖不到新方向，必须重召回（不能只重排）
        elif out.request_more and has_pool:
            action = "NEXT_BATCH"          # 换一批 → 池中取下一批
        elif has_pool:
            action = "REUSE_RERANK"        # 有池 + 仅软偏好微调/无变化 → 复用池重排
        else:
            action = "RETRIEVE"            # 首次推荐 / 无池 → 重召回

    plan["session_action"] = action

    # ---- 动作 → need_* 开关 ----
    if action == "RETRIEVE":
        plan["need_job_search"] = True
        plan["need_match_score"] = True
        plan["need_recommendation"] = True
        plan["need_skill_gap"] = True
    elif action == "REUSE_RERANK":
        plan["reuse_pool"] = True          # 复用池：不重检索、不重算基础匹配分，仅重算偏好分重排
    elif action == "NEXT_BATCH":
        plan["reuse_pool"] = True
        plan["next_batch"] = True          # 取池中下一批未展示岗位（已评分，直接展示）
    elif action == "USER_JD":
        plan["need_jd_input"] = True
        plan["need_jd_parse"] = True
        plan["need_match_score"] = True
        plan["need_recommendation"] = True
        plan["need_skill_gap"] = True
    elif action == "SELECTED":
        plan["reuse_pool"] = True          # 复用缓存岗位详情，不重检索/不重评分
    elif action == "ASSIST":
        plan["reuse_pool"] = True
        acts = set(out.assist_actions or [])
        plan["need_learning_plan"] = "LEARNING_PLAN" in acts
        plan["need_interview"] = "INTERVIEW_PREP" in acts

    # 通勤：有诉求即启用（RETRIEVE/REUSE_RERANK/SELECTED 下对岗位算通勤）
    if commute_on and action in ("RETRIEVE", "REUSE_RERANK", "NEXT_BATCH", "SELECTED"):
        plan["need_commute"] = True

    return plan
