# -*- coding: utf-8 -*-
"""planner_node：编排 NLU → 纠错 → 澄清 → 编排 → trace，写回 AgentState。

对外唯一入口（graph 注册的节点）。六个子模块各司其职，本文件只做串联与写回。
新增会话动作（session_action）驱动：硬约束变化重召回、软偏好变化复用池重排、
request_more 取下一批、SELECTED/ASSIST 读缓存岗位。
"""

from typing import Optional

from resume2job.tools.commute import DEFAULT_TRANSPORT, VALID_TRANSPORT
from resume2job.agent.planner import (
    context_builder, nlu_extractor, rule_corrector, clarification,
    policy_orchestrator, trace_logger,
)
from resume2job.agent.planner.schema import PlannerOutput

# 通勤约束触发时扩大召回，保证有足够候选供过滤
COMMUTE_TOP_K = 10
# 推荐时多召回并评分一个候选池（>展示数），供「换一批/重排」复用
POOL_TOP_K = 10


def _append_error(state: dict, message: str) -> list:
    errors = list(state.get("errors") or [])
    errors.append(message)
    return errors


def _commute_constraint(out: PlannerOutput) -> dict:
    """PlannerOutput.commute → 通勤约束 dict（喂 commute_intent；统一软：展示+达标加分）。"""
    transport = (out.commute.transport or "").strip().lower()
    if transport not in VALID_TRANSPORT:
        transport = DEFAULT_TRANSPORT
    addr = (out.commute.home_address or "").strip() or None
    enabled = bool(out.commute.enabled or addr or out.commute.max_minutes)
    return {
        "enabled": enabled,
        "home_address": addr,
        "max_minutes": out.commute.max_minutes,
        "transport": transport,
    }


def _legacy_commute_intent(cc: dict) -> dict:
    """兼容现有 commute 工具（compute_and_rank 读 intent 结构）。"""
    addr = cc.get("home_address")
    return {
        "has_commute_constraint": bool(cc.get("enabled") and (addr or cc.get("max_minutes"))),
        "user_address": addr,
        "address_confidence": "district" if addr else "vague",
        "max_commute_minutes": cc.get("max_minutes"),
        "preferred_transport": cc.get("transport") or DEFAULT_TRANSPORT,
        "raw_address_text": addr,
        "error": None if addr else "缺少可解析地址",
    }


def _round_direction_tags(soft_preferences: dict) -> list:
    """本轮抽到的方向偏好标签（direction_tags）。"""
    tags = []
    v = (soft_preferences or {}).get("direction_tags")
    if isinstance(v, list):
        tags.extend(str(x).strip() for x in v if str(x).strip())
    elif isinstance(v, str) and v.strip():
        tags.append(v.strip())
    return tags


def _merge_session_direction_tags(session_state: dict, this_round: list) -> dict:
    """方向偏好仅作用于当前会话：把本轮 direction_tags 并入会话活跃方向偏好（去重保序）。

    方向偏好仅会话级，不做长期持久化（长期偏好存储已废弃）。
    返回 {tag: 1.0} 供 retriever Query3 与 scorer direction_bonus 使用。
    """
    active = list((session_state or {}).get("active_direction_tags") or [])
    seen, merged = set(), []
    for t in active + this_round:
        low = t.lower()
        if t and low not in seen:
            seen.add(low)
            merged.append(t)
    return {t: 1.0 for t in merged}


def planner_node(state: dict) -> dict:
    """Function Calling 规划节点（会话动作驱动版）。"""
    print("[planner_node] 开始执行...")
    new_state = dict(state)

    # 1. 组装结构化上下文（含会话短期状态：活跃硬约束 / 方向偏好 / 候选池）
    ctx = context_builder.build_context(state)
    sess = ctx.session_state or {}

    # 2. NLU 抽语义（LLM；失败走规则兜底）
    decided_by = "llm"
    try:
        out = nlu_extractor.extract(ctx)
    except Exception as exc:
        new_state["errors"] = _append_error(new_state, f"planner: NLU LLM 失败（{exc}），规则兜底")
        out = nlu_extractor.rule_fallback(ctx)
        decided_by = "rule_fallback"

    # 3. 确定性纠错（JD 护栏 / 指代 / 学历护栏 / 正式岗澄清 / 缺槽 / 低置信）
    out = rule_corrector.correct(out, ctx)

    # 4. 澄清决策（任务式必需信息检查）
    clarify_q = clarification.decide(out, ctx)

    # 5. 语义 + 会话状态 → 执行计划（session_action）
    plan = policy_orchestrator.build_plan(out, ctx, clarify_question=clarify_q)

    # 6. trace 落库
    trace_logger.log(state.get("session_id") or "", ctx.current_message, out, plan, decided_by)

    # ---- 写回 State ----
    new_state["plan"] = plan
    new_state["intent"] = out.intent
    new_state["hard_constraints"] = out.hard_constraints or {}
    new_state["soft_preferences"] = out.soft_preferences or {}
    # 方向偏好仅会话级：合并会话活跃方向标签（不做长期持久化）
    new_state["preference_tags"] = _merge_session_direction_tags(sess, _round_direction_tags(out.soft_preferences))
    cc = _commute_constraint(out)
    new_state["commute_intent"] = _legacy_commute_intent(cc)

    # 澄清：短路业务链路，图直达 END
    if plan.get("clarify"):
        new_state["final_response"] = plan.get("clarify_question")
        print(f"[planner_node] 进入澄清：{plan.get('clarify_question')}")
        return new_state

    action = plan.get("session_action")

    # SELECTED / ASSIST：从会话取回指定岗位的完整结果，注入 match_results 供详情/辅助复用
    if action in ("SELECTED", "ASSIST"):
        selected = _pick_last_result(sess, out.selected_item_ref) if out.selected_item_ref else None
        if selected:
            new_state["match_results"] = [selected]
            new_state["selected_item_ref"] = out.selected_item_ref
            print(f"[planner_node] {action} 复用岗位：job_id={selected.get('job_id')}")
        else:
            # 指定的 job_id 不在会话缓存（已过期 / 不属本会话推荐）：明确告知并短路，不静默返回空
            ref = out.selected_item_ref or "（未指明）"
            new_state["final_response"] = (
                f"抱歉，岗位「{ref}」不在当前会话的推荐列表中，可能已过期。"
                f"请提供推荐报告中的 job_id（如 job_1），或先让我为你推荐岗位。"
            )
            new_state["plan"] = {**plan, "clarify": True, "session_action": "CLARIFY",
                                 "reuse_pool": False, "need_commute": False}
            new_state["errors"] = _append_error(new_state, f"planner: job_id={ref} 不在会话缓存，无法 {action}")
            print(f"[planner_node] {action} 失败：job_id={ref} 不在会话缓存，短路告知。")
            return new_state

    # REUSE_RERANK / NEXT_BATCH：注入会话候选池（已评分），executor 据此重排/取下一批
    if plan.get("reuse_pool") and action in ("REUSE_RERANK", "NEXT_BATCH"):
        pool = sess.get("candidate_pool") or []
        new_state["candidate_pool"] = pool
        new_state["shown_job_ids"] = sess.get("shown_job_ids") or []
        print(f"[planner_node] {action} 复用候选池：{len(pool)} 条")

    # 检索配置：仅 RETRIEVE 需要（合并会话活跃硬约束 + 本轮硬约束）
    rc = dict(state.get("retrieval_config") or {})
    prev_hard = sess.get("active_hard_constraints") or {}
    hc = out.hard_constraints or {}

    # 城市 / 岗位类型：会话活跃值打底，本轮显式值覆盖
    for src, rc_key in (("city", "city_filter"), ("job_type", "job_type_filter")):
        v = hc.get(src) or prev_hard.get(src)
        if v and str(v).strip():
            rc[rc_key] = str(v).strip()
    rc.setdefault("job_type_filter", "实习")  # 默认实习

    # 学历硬约束（召回前资格预筛：岗位要求 ≤ 候选人学历）：取自简历画像。
    # Phase 1 取消「只看≥该学历要求」min-required 模式；用户显式提到的学历仅作候选人学历兜底。
    rc["education_filter"] = ctx.resume_degree or prev_hard.get("education") or hc.get("education")
    rc.pop("direction_filter", None)               # 方向不再进 where 过滤

    # display_k=用户想看的岗位数；top_k=实际召回评分的候选池大小（≥展示数，供换一批/重排复用）
    rc.setdefault("display_k", int(rc.get("top_k") or 5))
    if plan.get("need_commute"):
        rc["top_k"] = max(int(rc.get("top_k") or 5), COMMUTE_TOP_K)
    if action == "RETRIEVE":
        rc["top_k"] = max(int(rc.get("top_k") or 5), POOL_TOP_K)

    new_state["retrieval_config"] = rc

    active = {k: v for k, v in rc.items() if v}
    print(f"[planner_node] intent={out.intent} action={action} 检索参数={active}")
    return new_state


def _pick_last_result(session_state: dict, ref: str) -> Optional[dict]:
    """从会话候选池按 job_id 选中目标——**仅限已展示（已叙述）岗位**。

    惰性叙述下，候选池里未展示岗位的 skill_gap / report 为空；SELECTED/ASSIST 必须拿到已叙述岗位
    才能读详情 / 出学习计划 / 面试题。ref 不在 shown_job_ids（已展示集合）时返回 None，
    由调用方走「岗位不在推荐列表」澄清分支，不会把空 skill_gap 喂给下游。
    """
    if not ref:
        return None
    shown = set((session_state or {}).get("shown_job_ids") or [])
    if ref not in shown:
        return None
    for r in (session_state or {}).get("candidate_pool") or []:
        if isinstance(r, dict) and r.get("job_id") == ref:
            return r
    return None
