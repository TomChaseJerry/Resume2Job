"""
nodes/executor.py

把 Stage 1~2 已实现的业务模块封装为 LangGraph 执行节点。

本模块属于「基于 Agentic RAG 的实习岗位智能推荐系统」的执行节点层，
负责串联：简历解析、岗位检索、JD 解析、匹配评分、推荐报告生成。

每个节点遵循统一约定：
    1. 从 AgentState 读取输入；
    2. 依据 state["plan"] 判断是否需要执行（开关为 False 直接返回原 State）；
    3. 调用已有业务模块完成任务（不重写其内部逻辑）；
    4. 把结果写回 AgentState；
    5. 出错时把错误信息追加到 state["errors"]，但不让流程崩溃；
    6. 返回更新后的 AgentState（浅拷贝，不破坏未负责的字段）。
"""

import json
from typing import Optional

from state import AgentState, MatchResult

# 已实现的业务模块，直接复用，不重写内部逻辑
from resume_parser import parse_resume
from jd_parser import parse_jd
from job_retriever import retrieve_jobs
from match_scorer import score_match
from skill_gap_analyzer import analyze_skill_gap
# Stage 4：完整报告优先用 generate_full_report；generate_recommendation 保留作兜底
from recommendation_writer import generate_full_report, generate_recommendation


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def append_error(state: AgentState, message: str) -> AgentState:
    """
    安全地把一条错误/警告信息追加到 errors 列表，返回更新后的浅拷贝 State。

    不假设 errors 字段一定存在，也不就地修改原列表。
    """
    new_state = dict(state)
    errors = list(state.get("errors") or [])
    errors.append(message)
    new_state["errors"] = errors
    return new_state


def get_plan_flag(state: AgentState, flag_name: str, default: bool = False) -> bool:
    """
    安全地读取 plan 中某个开关。

    plan 不存在或字段缺失时返回 default，避免直接索引导致崩溃。
    """
    plan = state.get("plan") or {}
    return bool(plan.get(flag_name, default))


def extract_jd_profile_from_candidate(candidate: dict) -> Optional[dict]:
    """
    从单个候选岗位中提取结构化 JD（jd_profile）。

    优先级：
        1. candidate["jd_profile"] 存在且非空 -> 直接使用；
        2. candidate["metadata"]["jd_profile_json"] -> 尝试 json.loads 还原；
        3. 都拿不到 -> 返回 None。
    不重新加载 JD 原文，也不重复调用 parse_jd（检索阶段已保存结构化结果）。
    """
    if not isinstance(candidate, dict):
        return None

    # 1. 直接携带的结构化 JD
    jd_profile = candidate.get("jd_profile")
    if isinstance(jd_profile, dict) and jd_profile:
        return jd_profile

    # 2. metadata 中的 JSON 字符串
    metadata = candidate.get("metadata") or {}
    jd_profile_json = metadata.get("jd_profile_json")
    if jd_profile_json:
        try:
            parsed = json.loads(jd_profile_json)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            return None

    return None


def get_job_id(jd_profile: dict, index: int) -> str:
    """
    按优先级获取 job_id：
        1. jd_profile["job_id"]
        2. jd_profile["metadata"]["job_id"]
        3. company + title
        4. fallback "job_{index}"
    """
    if isinstance(jd_profile, dict):
        job_id = jd_profile.get("job_id")
        if job_id:
            return str(job_id)

        metadata = jd_profile.get("metadata") or {}
        meta_job_id = metadata.get("job_id")
        if meta_job_id:
            return str(meta_job_id)

        company = jd_profile.get("company") or ""
        title = jd_profile.get("title") or ""
        combined = f"{company}{title}".strip()
        if combined:
            return combined

    return f"job_{index}"


# ---------------------------------------------------------------------------
# 节点 1：resume_parser_node
# ---------------------------------------------------------------------------

def resume_parser_node(state: AgentState) -> AgentState:
    """
    解析用户上传的 PDF 简历，生成 resume_profile。

    - plan["need_resume_parse"] 为 False -> 跳过；
    - resume_profile 已存在 -> 复用，不重复解析；
    - pdf_path 为空 -> 记录错误后返回。
    """
    print("[resume_parser_node] 开始执行...")

    if not get_plan_flag(state, "need_resume_parse"):
        return state

    # 已有结果直接复用，避免重复解析
    if state.get("resume_profile"):
        print("[resume_parser_node] resume_profile 已存在，跳过解析")
        return state

    pdf_path = state.get("pdf_path")
    if not pdf_path:
        return append_error(state, "resume_parser_node: 缺少 pdf_path，无法解析简历")

    new_state = dict(state)
    try:
        resume_profile = parse_resume(pdf_path)
        if not resume_profile:
            return append_error(state, "resume_parser_node: 简历解析结果为空，解析失败")
        new_state["resume_profile"] = resume_profile
        print("[resume_parser_node] 简历解析完成")
    except Exception as exc:
        return append_error(state, f"resume_parser_node: 简历解析异常：{exc}")

    return new_state


# ---------------------------------------------------------------------------
# 节点 2：job_retriever_node
# ---------------------------------------------------------------------------

def job_retriever_node(state: AgentState) -> AgentState:
    """
    根据 resume_profile 从岗位知识库召回候选岗位。

    检索参数全部来自 retrieval_config（缺失则用默认值），
    本节点不手写城市/方向/学历过滤逻辑。
    """
    print("[job_retriever_node] 开始执行...")

    if not get_plan_flag(state, "need_job_search"):
        return state

    resume_profile = state.get("resume_profile")
    if not resume_profile:
        return append_error(state, "job_retriever_node: 缺少 resume_profile，无法检索岗位")

    # 优先读取 retrieval_config，缺失时使用默认值
    config = state.get("retrieval_config") or {}
    top_k = config.get("top_k", 5)
    city_filter = config.get("city_filter")
    direction_filter = config.get("direction_filter")
    education_filter = config.get("education_filter")

    new_state = dict(state)
    try:
        candidate_jobs = retrieve_jobs(
            resume_profile=resume_profile,
            top_k=top_k,
            city_filter=city_filter,
            direction_filter=direction_filter,
            education_filter=education_filter,
        )
        candidate_jobs = candidate_jobs or []
        new_state["candidate_jobs"] = candidate_jobs
        print(f"[job_retriever_node] 召回候选岗位数量：{len(candidate_jobs)}")
    except Exception as exc:
        return append_error(state, f"job_retriever_node: 岗位检索异常：{exc}")

    return new_state


# ---------------------------------------------------------------------------
# 节点 3：jd_input_node
# ---------------------------------------------------------------------------

def jd_input_node(state: AgentState) -> AgentState:
    """
    处理用户直接粘贴 JD 的场景：解析 jd_text 并写入 jd_profiles。

    仅处理用户输入的 JD，不处理岗位检索结果。
    """
    print("[jd_input_node] 开始执行...")

    if not get_plan_flag(state, "need_jd_input"):
        return state

    jd_text = state.get("jd_text")
    if not jd_text:
        return append_error(
            state, "jd_input_node: plan 要求处理 JD 输入，但 jd_text 为空"
        )

    # 仅当 need_jd_parse 为 True 时才解析
    if not get_plan_flag(state, "need_jd_parse"):
        return state

    new_state = dict(state)
    try:
        jd_profile = parse_jd(jd_text)
        if not jd_profile:
            return append_error(state, "jd_input_node: JD 解析结果为空，解析失败")
        new_state["jd_profiles"] = [jd_profile]
        print("[jd_input_node] JD 解析完成，写入 jd_profiles")
    except Exception as exc:
        return append_error(state, f"jd_input_node: JD 解析异常：{exc}")

    return new_state


# ---------------------------------------------------------------------------
# 节点 4：jd_analyzer_node
# ---------------------------------------------------------------------------

def jd_analyzer_node(state: AgentState) -> AgentState:
    """
    整理岗位结构化信息，确保 match_scorer_node 能读取到 jd_profiles。

    两类来源：
        1. 岗位检索结果 candidate_jobs（内含已保存的结构化 jd_profile）；
        2. 用户粘贴 JD 后已得到的 jd_profiles。
    不重新加载 JD 原文，也不对检索结果重复调用 parse_jd。
    """
    print("[jd_analyzer_node] 开始执行...")

    candidate_jobs = state.get("candidate_jobs") or []
    existing_profiles = state.get("jd_profiles") or []

    # 规则 1：不需要解析 JD 且没有候选岗位，无事可做
    if not get_plan_flag(state, "need_jd_parse") and not candidate_jobs:
        return state

    # 规则 2：已有 jd_profiles 且没有候选岗位，直接复用
    if existing_profiles and not candidate_jobs:
        print("[jd_analyzer_node] 已存在 jd_profiles 且无候选岗位，直接复用")
        return state

    new_state = dict(state)
    errors = list(state.get("errors") or [])

    if candidate_jobs:
        # 从候选岗位中提取结构化 JD
        jd_profiles = []
        for idx, candidate in enumerate(candidate_jobs):
            jd_profile = extract_jd_profile_from_candidate(candidate)
            if jd_profile:
                jd_profiles.append(jd_profile)
            else:
                # 单个岗位无法提取 -> 记录 warning 并跳过，不中断
                errors.append(
                    f"jd_analyzer_node: 第 {idx} 个候选岗位缺少可用 jd_profile，已跳过"
                )

        if not jd_profiles:
            errors.append("jd_analyzer_node: 候选岗位中没有任何可用 jd_profile")

        new_state["jd_profiles"] = jd_profiles
        new_state["errors"] = errors
        print(f"[jd_analyzer_node] 整理结构化 JD 数量：{len(jd_profiles)}")
        return new_state

    # 没有候选岗位，沿用已有 jd_profiles（可能为空）
    if not existing_profiles:
        errors.append("jd_analyzer_node: 没有任何可用 jd_profile")
        new_state["errors"] = errors
    return new_state


# ---------------------------------------------------------------------------
# 节点 5：match_scorer_node
# ---------------------------------------------------------------------------

def _safe_skill_gap(resume_profile: dict, jd_profile: dict, match_score: dict,
                    job_id: str, errors: list) -> dict:
    """调用 analyze_skill_gap；失败时返回带 error 的空 skill_gap，不中断岗位。"""
    try:
        skill_gap = analyze_skill_gap(
            resume_profile=resume_profile,
            jd_profile=jd_profile,
            match_result=match_score,
        )
        if isinstance(skill_gap, dict):
            return skill_gap
        errors.append(f"match_scorer_node: 岗位 {job_id} 的 skill_gap 返回非法类型，已置空")
    except Exception as exc:
        errors.append(f"match_scorer_node: 岗位 {job_id} 技能差距分析失败：{exc}")
    # 兜底：结构与 analyze_skill_gap 一致的空 skill_gap
    return {
        "items": [],
        "matched_skills": [],
        "weak_skills": [],
        "missing_skills": [],
        "overall_risk": "low",
        "overall_risk_reason": "技能差距分析未能产出结果。",
        "top_suggestion": "建议正常投递，并在简历中突出相关项目证据。",
        "error": f"skill_gap 生成失败（岗位 {job_id}）",
    }


def _safe_full_report(jd_profile: dict, match_score: dict, skill_gap: dict,
                      job_id: str, errors: list) -> str:
    """生成完整报告；generate_full_report 失败时回退到 generate_recommendation。"""
    try:
        return generate_full_report(jd_profile, match_score, skill_gap)
    except Exception as exc:
        errors.append(
            f"match_scorer_node: 岗位 {job_id} generate_full_report 失败：{exc}，回退旧版报告"
        )
    try:
        return generate_recommendation(jd_profile, match_score, skill_gap)
    except Exception as exc2:
        errors.append(f"match_scorer_node: 岗位 {job_id} generate_recommendation 也失败：{exc2}")
        return ""


def match_scorer_node(state: AgentState) -> AgentState:
    """
    对每个岗位执行：匹配评分 -> 技能差距分析 -> 生成完整报告（Stage 4 固定内部链路）。

    - score_match 失败：跳过该岗位并记录错误；
    - analyze_skill_gap 失败：不跳过，生成带 error 的空 skill_gap 继续；
    - generate_full_report 失败：回退到 generate_recommendation；
    - 单个岗位失败不会中断其它岗位；
    - 最终结果按 match_score["final_score"] 降序排序。
    """
    print("[match_scorer_node] 开始执行...")

    need_match = get_plan_flag(state, "need_match_score")
    need_reco = get_plan_flag(state, "need_recommendation")

    # 两个开关都为 False -> 跳过
    if not need_match and not need_reco:
        return state

    resume_profile = state.get("resume_profile")
    if not resume_profile:
        return append_error(state, "match_scorer_node: 缺少 resume_profile，无法评分")

    jd_profiles = state.get("jd_profiles") or []
    if not jd_profiles:
        return append_error(state, "match_scorer_node: 缺少 jd_profiles，无法评分")

    new_state = dict(state)
    errors = list(state.get("errors") or [])
    match_results: list[MatchResult] = []
    skill_gaps_all: list[dict] = []  # 同步写回 state["skill_gaps"]

    for idx, jd_profile in enumerate(jd_profiles):
        job_id = get_job_id(jd_profile, idx)
        try:
            # 1. 匹配评分（失败则跳过该岗位）
            match_score = score_match(resume_profile, jd_profile)
        except Exception as exc:
            errors.append(f"match_scorer_node: 岗位 {job_id} 评分失败：{exc}")
            continue

        # 2. 技能差距分析（失败不跳过，返回带 error 的空结构）
        skill_gap = _safe_skill_gap(resume_profile, jd_profile, match_score, job_id, errors)

        # 3. 完整报告（full_report 失败回退旧版）
        report = _safe_full_report(jd_profile, match_score, skill_gap, job_id, errors)

        match_results.append(
            MatchResult(
                job_id=job_id,
                jd_profile=jd_profile,
                match_score=match_score,
                skill_gap=skill_gap,
                report=report,
            )
        )
        # 附带 job_id，便于 state["skill_gaps"] 溯源
        skill_gaps_all.append({"job_id": job_id, **skill_gap})

    # 按 final_score 降序排序（缺失时按 0 处理）
    match_results.sort(
        key=lambda r: (r.get("match_score") or {}).get("final_score", 0),
        reverse=True,
    )

    new_state["match_results"] = match_results
    new_state["skill_gaps"] = skill_gaps_all
    new_state["errors"] = errors
    print(f"[match_scorer_node] 完成评分岗位数量：{len(match_results)}，"
          f"技能差距分析数量：{len(skill_gaps_all)}")
    return new_state
