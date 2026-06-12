"""
agent/nodes/enhancements.py

Tool Calling 增强节点（位于 match_scorer 之后、END 之前）。

主链路（解析 → 检索 → 评分 → 报告）产出 match_results 后，本节点把三个
可选增强能力注册为 OpenAI 兼容 tools，由 LLM 根据用户原话**自主决定**
调用哪些（可以一个不调，也可以多个同时调）：

    - compute_commute               通勤计算与重排（高德 API）
    - generate_learning_plan        阶段化学习计划（基于首选岗位 skill_gap）
    - generate_interview_questions  3 道模拟面试题（基于首选岗位）

这是项目中「LLM 通过 Tool Calling 决策 + Python 执行工具」的核心展示点：
    - LLM 只产出 tool_calls（带参数），不执行任何副作用；
    - 工具本体是确定性 Python 函数，逐个执行并把结果写回 AgentState；
    - LLM 不可用 / 调用失败时退回关键词规则兜底，保证增强功能不因此失效。
"""

from typing import Optional

from resume2job.agent.state import AgentState
from resume2job.agent.nodes.executor import append_error
from resume2job.core.config import PLANNER_MODEL
from resume2job.core.llm import get_chat_llm
from resume2job.generation.learning_plan import build_learning_plan
from resume2job.generation.interview import generate_interview_questions
from resume2job.tools.commute import compute_and_rank

# ===== 规则兜底关键词（LLM 不可用时启用）=====
COMMUTE_KEYWORDS = ("通勤", "地铁", "多久到", "小时以内", "路线", "公司地址")
LEARNING_PLAN_KEYWORDS = (
    "学习路径", "学习计划", "学习规划", "学习路线", "进阶路线",
    "怎么学", "如何学", "如何提升", "提升计划", "补齐", "补强",
)
INTERVIEW_KEYWORDS = (
    "面试", "面试题", "模拟面试", "面试准备", "面试辅导", "面试问题",
    "面经", "怎么答", "作答思路", "面试官",
)

# ===== 注册给 LLM 的工具 Schema（OpenAI function calling 格式）=====
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compute_commute",
            "description": "计算候选岗位从用户住址出发的通勤时长，按通勤约束过滤排序并并入报告。"
                           "仅当用户明确提出通勤/距离/路线类诉求时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string",
                                "description": "用户住址（可选，规划阶段已抽取过则留空）"},
                    "max_minutes": {"type": "integer",
                                    "description": "通勤时间上限（分钟，可选）"},
                    "transport": {"type": "string",
                                  "enum": ["transit", "driving", "walking", "cycling"],
                                  "description": "交通方式（可选，默认 transit）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_learning_plan",
            "description": "基于首选岗位的技能差距生成阶段化学习计划。"
                           "仅当用户明确要求学习计划/学习路径/如何提升时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interview_questions",
            "description": "针对首选岗位生成 3 道模拟面试题。"
                           "仅当用户明确要求面试题/模拟面试/面试准备时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_SYSTEM_PROMPT = (
    "你是一个实习求职岗位推荐 Agent 的增强功能调度器。岗位匹配评分已完成，"
    "请根据用户本轮问题判断还需要调用哪些增强工具。\n"
    "原则：用户没有明确提出的诉求，一律不要调用对应工具；"
    "没有任何增强诉求时不调用任何工具，直接回复「无需增强」。"
)


# ---------------------------------------------------------------------------
# 工具本体（确定性 Python，逐个写回 state）
# ---------------------------------------------------------------------------
def _run_commute(state: AgentState, args: Optional[dict] = None) -> AgentState:
    """通勤计算与重排（原 commute_node 逻辑）。"""
    print("[enhancement] 执行工具：compute_commute")
    args = args or {}

    # LLM 工具参数可覆盖/补全 planner 抽取的 commute_intent
    intent = dict(state.get("commute_intent") or {})
    if args.get("address"):
        intent["user_address"] = str(args["address"]).strip()
        intent["raw_address_text"] = intent["user_address"]
        intent["has_commute_constraint"] = True
        intent["error"] = None
    if args.get("max_minutes"):
        intent["max_commute_minutes"] = int(args["max_minutes"])
        intent["has_commute_constraint"] = True
    if args.get("transport"):
        intent["preferred_transport"] = str(args["transport"])
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

    # 从 match_results 提炼通勤计算所需的岗位信息
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
            "final_score": (mr.get("match_score") or {}).get("final_score"),
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

    # 1) 回填通勤信息到每条 match_result，并把摘要追加进 report
    for job_id, info in per_job.items():
        mr = mr_by_id.get(job_id)
        if not mr:
            continue
        mr["commute"] = info
        summary = info.get("commute_summary")
        if summary and isinstance(mr.get("report"), str):
            mr["report"] = mr["report"].rstrip() + f"\n\n【通勤】{summary}"

    # 2) 按通勤排序结果重排 match_results：达标在前，超标在后
    ordered_ids = ([v["job_id"] for v in ranked.get("ranked_jobs", [])] +
                   [v["job_id"] for v in ranked.get("filtered_out", [])])
    reordered = [mr_by_id[j] for j in ordered_ids if j in mr_by_id]
    for mr in match_results:  # 容错：补回任何未被排序覆盖的岗位
        if mr not in reordered:
            reordered.append(mr)
    new_state["match_results"] = reordered

    # 3) 通勤结果汇总写入 state
    new_state["commute_results"] = ranked.get("ranked_jobs", []) + ranked.get("filtered_out", [])
    note = ranked.get("note") or ""
    if note:
        new_state["final_response"] = note
    new_state["errors"] = errors
    print(f"[enhancement] 通勤计算完成：{note}")
    return new_state


def _run_learning_plan(state: AgentState, args: Optional[dict] = None) -> AgentState:
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


def _run_interview(state: AgentState, args: Optional[dict] = None) -> AgentState:
    """3 道模拟面试题（见 generation/interview.py）。"""
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


_TOOL_EXECUTORS = {
    "compute_commute": _run_commute,
    "generate_learning_plan": _run_learning_plan,
    "generate_interview_questions": _run_interview,
}


# ---------------------------------------------------------------------------
# Tool Calling 决策
# ---------------------------------------------------------------------------
def _decide_tools_by_llm(user_query: str) -> list:
    """LLM bind_tools 决策：返回 tool_calls 列表 [{"name", "args"}]。失败抛异常。"""
    llm = get_chat_llm(model=PLANNER_MODEL).bind_tools(_TOOLS)
    response = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"## 用户本轮问题\n{user_query}"},
    ])
    return [
        {"name": tc.get("name"), "args": tc.get("args") or {}}
        for tc in (getattr(response, "tool_calls", None) or [])
        if tc.get("name") in _TOOL_EXECUTORS
    ]


def _decide_tools_by_rules(state: AgentState) -> list:
    """关键词规则兜底：根据用户原话与 planner 的通勤意图决定调用哪些工具。"""
    q = state.get("user_query") or ""
    calls = []
    if state.get("commute_intent") or any(kw in q for kw in COMMUTE_KEYWORDS):
        calls.append({"name": "compute_commute", "args": {}})
    if any(kw in q for kw in LEARNING_PLAN_KEYWORDS):
        calls.append({"name": "generate_learning_plan", "args": {}})
    if any(kw in q for kw in INTERVIEW_KEYWORDS):
        calls.append({"name": "generate_interview_questions", "args": {}})
    return calls


# ---------------------------------------------------------------------------
# 节点：enhancement_node
# ---------------------------------------------------------------------------
def enhancement_node(state: AgentState) -> AgentState:
    """Tool Calling 增强节点：LLM 决定调用哪些增强工具，Python 逐个执行。"""
    # 无评分结果（如纯 skill_gap_only / 检索为空）时增强无意义，静默跳过
    if not state.get("match_results"):
        return state

    print("[enhancement_node] 开始执行...")
    user_query = state.get("user_query") or ""

    try:
        tool_calls = _decide_tools_by_llm(user_query)
        decided_by = "tool_calling"
    except Exception as exc:
        state = append_error(state, f"enhancement_node: Tool Calling 决策失败（{exc}），改用规则兜底")
        tool_calls = _decide_tools_by_rules(state)
        decided_by = "rules"

    # planner 已识别到通勤意图、但 LLM 漏调通勤工具时补上（确定性信号优先；
    # 含「有通勤诉求但缺地址」的场景——通勤工具会在报告中显式说明缺地址）
    if state.get("commute_intent") and not any(
        c["name"] == "compute_commute" for c in tool_calls
    ):
        tool_calls.append({"name": "compute_commute", "args": {}})

    if not tool_calls:
        print("[enhancement_node] LLM 判定无需任何增强工具")
        return state

    print(f"[enhancement_node] 决策方式={decided_by}，"
          f"调用工具：{[c['name'] for c in tool_calls]}")

    for call in tool_calls:
        executor = _TOOL_EXECUTORS[call["name"]]
        state = executor(state, call.get("args"))

    return state
