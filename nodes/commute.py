"""
nodes/commute.py

通勤相关的两个 LangGraph 节点（仅在 plan["need_commute"] 为 True 时实际执行）：

    1. commute_planner_node（早节点，位于 planner 之后、resume_parser 之前）：
       - 用 LLM 抽取通勤意图（住址 / 时间上限 / 交通方式）；
       - 把住址城市回填到 retrieval_config.city_filter（不覆盖用户显式过滤），
         并扩大 top_k，实现「按通勤需求规划检索」。

    2. commute_node（晚节点，位于 match_scorer 之后、END 之前）：
       - 对已评分的 match_results 计算各岗位通勤时长（高德 API，Python 编排）；
       - 过滤 + 排序：达标优先、按 final_score 排，超标排末位、不足则补足；
       - 把通勤摘要并入每条 report，重排 match_results，写 commute_results 与 final_response。

两个节点在 need_commute 为 False 时直接早返回，因此可固定挂在主图上，无需新增条件路由。
"""

from state import AgentState
from commute_calculator import extract_commute_intent, extract_city, compute_and_rank

# 复用 executor 的安全工具，保持错误追加 / plan 读取风格一致
from nodes.executor import append_error, get_plan_flag


# 通勤约束触发时，检索召回的扩大目标（保证有足够候选供过滤）
_COMMUTE_TOP_K = 10


# ===== 早节点：通勤意图抽取 + 规划检索 =====
def commute_planner_node(state: AgentState) -> AgentState:
    """抽取通勤意图并回填检索参数。need_commute 为 False 时早返回。"""
    # 通勤未开启：静默跳过，不打印「开始执行」，避免日志误导（看似执行实则未做事）
    if not get_plan_flag(state, "need_commute"):
        return state
    print("[commute_planner_node] 开始执行...")

    user_query = state.get("user_query") or ""
    intent = extract_commute_intent(user_query)

    new_state = dict(state)
    new_state["commute_intent"] = intent

    # 抽取失败 / 无约束：记录信息但不阻断
    if intent.get("error"):
        new_state = append_error(new_state, f"commute_planner_node: {intent['error']}")
    if not intent.get("has_commute_constraint"):
        print("[commute_planner_node] 未识别到通勤约束，跳过检索规划")
        return new_state

    # 规划检索：城市回填 + 扩大召回
    rc = dict(state.get("retrieval_config") or {})
    city = extract_city(intent.get("user_address"))
    if city and not rc.get("city_filter"):
        rc["city_filter"] = city
    current_top_k = rc.get("top_k") or 5
    rc["top_k"] = max(int(current_top_k), _COMMUTE_TOP_K)
    new_state["retrieval_config"] = rc

    print(f"[commute_planner_node] 通勤约束：住址={intent.get('user_address')}，"
          f"上限={intent.get('max_commute_minutes')}分钟，"
          f"city_filter={rc.get('city_filter')}，top_k={rc['top_k']}")
    return new_state


# ===== 晚节点：通勤计算 + 过滤排序 =====
def _build_commute_jobs(match_results: list) -> list:
    """从 match_results 提炼通勤计算所需的岗位信息。"""
    jobs = []
    for mr in match_results:
        if not isinstance(mr, dict):
            continue
        jd = mr.get("jd_profile") or {}
        loc = jd.get("location") or {}
        office = loc.get("office_address") or loc.get("district") or loc.get("city")
        final_score = (mr.get("match_score") or {}).get("final_score")
        jobs.append({
            "job_id": mr.get("job_id"),
            "company": jd.get("company"),
            "title": jd.get("title"),
            "office_address": office,
            "city": loc.get("city"),
            "final_score": final_score,
        })
    return jobs


def commute_node(state: AgentState) -> AgentState:
    """计算通勤并按通勤约束重排 match_results。need_commute 为 False 时早返回。"""
    # 通勤未开启：静默跳过，不打印「开始执行」，避免日志误导
    if not get_plan_flag(state, "need_commute"):
        return state
    print("[commute_node] 开始执行...")

    intent = state.get("commute_intent") or {}
    match_results = list(state.get("match_results") or [])

    # 通勤已开启但抽取不到可用约束（如缺少可解析地址）：在报告中显式说明，而非静默无输出
    if not intent.get("has_commute_constraint"):
        reason = intent.get("error") or "缺少可解析的起点或终点地址"
        note = f"通勤评估：当前{reason}，暂未计入通勤分。"
        print(f"[commute_node] 无可用通勤约束，跳过通勤计算：{reason}")
        new_state = dict(state)
        # 把说明并入每条报告，保证「节点开启了就有可见输出」
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
        return append_error(state, "commute_node: 无 match_results，无法计算通勤")

    jobs = _build_commute_jobs(match_results)

    try:
        ranked = compute_and_rank(intent, jobs)
    except Exception as exc:
        return append_error(state, f"commute_node: 通勤计算异常：{exc}")

    new_state = dict(state)
    errors = list(state.get("errors") or [])
    if ranked.get("error"):
        errors.append(f"commute_node: {ranked['error']}")

    per_job = ranked.get("per_job") or {}
    mr_by_id = {mr.get("job_id"): mr for mr in match_results if isinstance(mr, dict)}

    # 1) 回填通勤信息到每条 match_result，并把摘要追加进 report
    for job_id, info in per_job.items():
        mr = mr_by_id.get(job_id)
        if not mr:
            continue
        mr["commute"] = info
        summary = info.get("commute_summary")
        if summary and isinstance(mr.get("report"), str):
            mr["report"] = mr["report"].rstrip() + f"\n\n【通勤】{summary}"

    # 2) 按通勤排序结果重排 match_results：ranked_jobs 在前，filtered_out 在后
    ordered_ids = ([v["job_id"] for v in ranked.get("ranked_jobs", [])] +
                   [v["job_id"] for v in ranked.get("filtered_out", [])])
    reordered = [mr_by_id[j] for j in ordered_ids if j in mr_by_id]
    # 容错：补回任何未被排序覆盖的岗位
    for mr in match_results:
        if mr not in reordered:
            reordered.append(mr)

    new_state["match_results"] = reordered

    # 3) 通勤结果汇总写入 state
    new_state["commute_results"] = ranked.get("ranked_jobs", []) + ranked.get("filtered_out", [])
    note = ranked.get("note") or ""
    if note:
        new_state["final_response"] = note
    new_state["errors"] = errors

    print(f"[commute_node] 通勤计算完成：{note}")
    return new_state
