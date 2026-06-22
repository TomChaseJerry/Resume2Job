"""
agent/nodes/enhancements.py

增强执行节点（位于 match_scorer 之后、END 之前）。

主链路产出 match_results 后，本节点按 **plan 的 need_* 开关**执行对应增强工具：

    - need_commute        compute_commute              通勤计算与重排（高德 API）
    - need_learning_plan  generate_learning_plan       阶段化学习计划（基于首选岗位 skill_gap）
    - need_interview      generate_interview_questions 岗位定制面试练习题（基于首选岗位）

增强「调不调用」由 planner（policy_orchestrator 据 intent=ASSIST + assist_actions 产出 need_learning_plan /
need_interview，据通勤诉求产出 need_commute）决定，本节点按开关确定性执行。工具本体是 Python 函数，逐个写回 AgentState。
通勤参数取自 planner 写入的 state["commute_intent"]（不再由 LLM tool-calling 传参）。
"""

from resume2job.agent.state import AgentState
from resume2job.agent.nodes.executor import append_error, get_plan_flag
from resume2job.generation.learning_plan import build_learning_plan
from resume2job.generation.interview import generate_interview_questions
from resume2job.tools.commute import compute_and_rank
from resume2job.scoring.match_scorer import (
    compute_commute_bonus, calculate_rank_score, judge_match_level,
)


# ---------------------------------------------------------------------------
# 工具本体（确定性 Python，逐个写回 state）
# ---------------------------------------------------------------------------
def _run_commute(state: AgentState) -> AgentState:
    """通勤计算与重排：算各岗位通勤 → 回填 commute_bonus 重算 rank_score → 按 rank_score 重排。"""
    print("[enhancement] 执行工具：compute_commute")

    # 通勤约束来自 planner 写入的 commute_intent（地址/时间上限/交通方式）
    intent = dict(state.get("commute_intent") or {})
    intent.setdefault("preferred_transport", "transit")

    match_results = list(state.get("match_results") or [])
    new_state = dict(state)
    new_state["commute_intent"] = intent

    # 有通勤诉求但抽取不到可用约束（如缺少可解析地址）：在报告中显式说明，而非静默无输出
    if not intent.get("has_commute_constraint") or not intent.get("user_address"):
        reason = intent.get("error") or "缺少可解析的起点或终点地址"
        note = f"通勤评估：当前{reason}，暂未计入通勤分。"
        print(f"[enhancement] 无可用通勤约束，跳过通勤计算：{reason}")
        results = []
        for mr in match_results:
            if isinstance(mr, dict) and isinstance(mr.get("report"), str):
                mr = dict(mr)
                mr["report"] = mr["report"].rstrip() + f"\n\n【通勤】{note}"
            results.append(mr)
        if results:
            new_state["match_results"] = results
        new_state["commute_note"] = note
        return new_state

    if not match_results:
        return append_error(new_state, "enhancement: 无 match_results，无法计算通勤")

    # 从 match_results 提炼通勤计算所需的岗位信息（relevance 用 rank_score 的基础部分）
    jobs = []
    for mr in match_results:
        if not isinstance(mr, dict):
            continue
        jd = mr.get("jd_profile") or {}
        loc = jd.get("location") or {}
        jobs.append({
            "job_id": mr.get("job_id"),
            "company": jd.get("company"),
            "title": jd.get("title"),
            "office_address": loc.get("office_address") or loc.get("district") or loc.get("city"),
            "city": loc.get("city"),
            "final_score": (mr.get("match_score") or {}).get("rank_score"),
        })

    try:
        ranked = compute_and_rank(intent, jobs)
    except Exception as exc:
        return append_error(new_state, f"enhancement: 通勤计算异常：{exc}")

    errors = list(new_state.get("errors") or [])
    if ranked.get("error"):
        errors.append(f"enhancement: {ranked['error']}")

    per_job = ranked.get("per_job") or {}
    mr_by_id = {mr.get("job_id"): mr for mr in match_results if isinstance(mr, dict)}
    max_minutes = intent.get("max_commute_minutes")

    # 1) 回填通勤信息 + 计算 commute_bonus 并重算 rank_score（通勤偏好加分并入排序分）
    for job_id, info in per_job.items():
        mr = mr_by_id.get(job_id)
        if not mr:
            continue
        mr["commute"] = info
        ms = mr.get("match_score") or {}
        bonus_info = compute_commute_bonus(info, max_minutes)
        ms["commute_bonus"] = bonus_info["bonus"]
        ms["commute_bonus_info"] = bonus_info
        ms["rank_score"] = calculate_rank_score(
            ms.get("match_score", 0), ms.get("direction_bonus", 0.0), bonus_info["bonus"],
        )
        ms["match_level"] = judge_match_level(ms["rank_score"])
        mr["match_score"] = ms

        # 追加通勤摘要到报告（含加分与更新后的最终推荐分，保持报告自洽）
        summary = info.get("commute_summary")
        if summary and isinstance(mr.get("report"), str):
            extra = f"\n\n【通勤】{summary}"
            if max_minutes and bonus_info["bonus"]:
                extra += f"（通勤加分 +{int(bonus_info['bonus'])}，最终推荐分 {ms['rank_score']}）"
            mr["report"] = mr["report"].rstrip() + extra

    # 2) 按更新后的 rank_score 降序重排 match_results（通勤加分体现到最终排序）
    reordered = sorted(
        match_results,
        key=lambda r: (r.get("match_score") or {}).get("rank_score", 0) if isinstance(r, dict) else 0,
        reverse=True,
    )
    new_state["match_results"] = reordered

    # 3) 通勤结果汇总写入 state
    new_state["commute_results"] = ranked.get("ranked_jobs", []) + ranked.get("filtered_out", [])
    note = ranked.get("note") or ""
    if note:
        new_state["final_response"] = note
    new_state["errors"] = errors
    print(f"[enhancement] 通勤计算完成：{note}")
    return new_state


def _run_learning_plan(state: AgentState) -> AgentState:
    """阶段化学习计划（单次 LLM 调用，见 generation/learning_plan.py）。"""
    print("[enhancement] 执行工具：generate_learning_plan")
    match_results = state.get("match_results") or []
    new_state = dict(state)
    if not match_results:
        return append_error(new_state, "enhancement: 无 match_results，无法生成学习计划")
    try:
        plan = build_learning_plan(match_results[0], state.get("user_query") or "")
        new_state["learning_plan"] = plan
        if plan.get("error"):
            new_state = append_error(new_state, f"enhancement: {plan['error']}")
    except Exception as exc:
        return append_error(new_state, f"enhancement: 学习计划生成异常：{exc}")
    return new_state


def _run_interview(state: AgentState) -> AgentState:
    """岗位定制面试练习题（见 generation/interview.py）。"""
    print("[enhancement] 执行工具：generate_interview_questions")
    match_results = state.get("match_results") or []
    new_state = dict(state)
    if not match_results:
        return append_error(new_state, "enhancement: 无 match_results，无法生成面试题")
    mr0 = match_results[0] if isinstance(match_results[0], dict) else {}
    try:
        result = generate_interview_questions(
            resume_profile=state.get("resume_profile") or {},
            jd_profile=mr0.get("jd_profile") or {},
            match_result=mr0.get("match_score") or {},
            skill_gap=mr0.get("skill_gap") or {},
        )
        new_state["interview_prep"] = result
        if result.get("error"):
            new_state = append_error(new_state, f"enhancement: {result['error']}")
    except Exception as exc:
        return append_error(new_state, f"enhancement: 面试题生成异常：{exc}")
    return new_state


# ---------------------------------------------------------------------------
# 节点：enhancement_node（按 plan 开关执行）
# ---------------------------------------------------------------------------
def enhancement_node(state: AgentState) -> AgentState:
    """增强执行节点：按 plan 的 need_* 开关（planner 据 intent=ASSIST + assist_actions / 通勤诉求 已决定）执行工具。"""
    # 无评分结果（如纯 skill_gap_only / 检索为空）时增强无意义，静默跳过
    if not state.get("match_results"):
        return state

    print("[enhancement_node] 开始执行...")
    executed = []
    if get_plan_flag(state, "need_commute"):
        state = _run_commute(state)
        executed.append("commute")
    if get_plan_flag(state, "need_learning_plan"):
        state = _run_learning_plan(state)
        executed.append("learning_plan")
    if get_plan_flag(state, "need_interview"):
        state = _run_interview(state)
        executed.append("interview")

    if executed:
        print(f"[enhancement_node] 执行增强：{executed}")
    else:
        print("[enhancement_node] plan 未请求任何增强")
    return state
