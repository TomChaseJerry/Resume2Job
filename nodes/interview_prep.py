"""
nodes/interview_prep.py

面试辅助节点（可选增强，默认关闭）。

仅当 plan["need_interview_prep"] 为 True（planner 依据用户措辞开启）时执行：
    针对当前首选岗位 match_results[0]，调用 interview_preparer 生成模拟面试题，
    写入 state["interview_prep"]。

作答思路框架（generate_answer_framework）是「点击某道题再生成」的第二层能力，
按需在前端/聊天层调用，不在本节点一次性生成，避免输出过长。
"""

from state import AgentState
from nodes.executor import append_error, get_plan_flag

from interview_preparer import generate_interview_questions


def interview_prep_node(state: AgentState) -> AgentState:
    """生成模拟面试题（need_interview_prep 为 False 时早返回）。"""
    print("[interview_prep_node] 开始执行...")

    if not get_plan_flag(state, "need_interview_prep"):
        return state

    match_results = state.get("match_results") or []
    if not match_results:
        return append_error(state, "interview_prep_node: 无 match_results，无法生成面试题")

    mr0 = match_results[0] if isinstance(match_results[0], dict) else {}

    try:
        result = generate_interview_questions(
            resume_profile=state.get("resume_profile") or {},
            jd_profile=mr0.get("jd_profile") or {},
            match_result=mr0.get("match_score") or {},
            skill_gap=mr0.get("skill_gap") or {},
        )
    except Exception as exc:
        return append_error(state, f"interview_prep_node: 面试题生成异常：{exc}")

    new_state = dict(state)
    new_state["interview_prep"] = result
    n = len(result.get("questions") or [])
    print(f"[interview_prep_node] 面试题生成完成，共 {n} 道")
    return new_state
