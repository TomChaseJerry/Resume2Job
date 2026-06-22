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

import re
import json
import concurrent.futures
from typing import Optional

from resume2job.agent.state import AgentState, MatchResult

# 已实现的业务模块，直接复用，不重写内部逻辑
from resume2job.parsing.resume_parser import parse_resume
from resume2job.parsing.jd_parser import parse_jd
from resume2job.retrieval.retriever import retrieve_jobs
from resume2job.retrieval.indexer import lookup_ingested_jd_profile
from resume2job.scoring.match_scorer import (
    score_match, compute_direction_bonus, calculate_rank_score, judge_match_level,
)
# 合并调用（1 次 LLM 产出 skill_gap + 报告）+ 纯 Python 重组 + 规则兜底完整报告 + 规则兜底 skill_gap
# （skill_gap 视图 2026-06-21 已并入 recommendation 报告层，不再有独立 scoring/skill_gap 模块）
from resume2job.generation.recommendation import (
    generate_report_and_gap, recompose_report, generate_full_report, rule_based_skill_gap,
)


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
    - pdf_path 为空 -> 静默跳过（多轮 / ASSIST / 复用池等无 PDF 轮次由 profile_cache 加载缓存画像；
      若确实无任何画像，planner 的澄清机制已在更早拦截，不在此处报错制造噪音）。
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
        # 本轮没传 PDF：交给下游 profile_cache 复用缓存画像（无缓存的「无简历推荐」已被澄清拦截）
        print("[resume_parser_node] 本轮无 PDF，跳过解析（由 profile_cache 复用缓存画像）")
        return state

    new_state = dict(state)
    try:
        resume_profile = parse_resume(pdf_path)
        # 解析失败可区分（PDF 无文本 / 模型异常 / 输出非 JSON）：给出精确反馈，不与「成功但字段缺失」混淆
        if not resume_profile or (isinstance(resume_profile, dict) and resume_profile.get("error")):
            reason = resume_profile.get("error") if isinstance(resume_profile, dict) and resume_profile.get("error") \
                else "简历解析结果为空"
            stage = resume_profile.get("error_stage") if isinstance(resume_profile, dict) else None
            return append_error(state, f"resume_parser_node: 简历解析失败[{stage or 'unknown'}]：{reason}")
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
    job_type_filter = config.get("job_type_filter") or "实习"  # 岗位类型硬约束，默认实习
    # 候选人学历 → rank（本1硕2博3）供召回前学历资格预筛：优先 retrieval_config.education_filter
    # （planner 显式约束），否则取简历 highest_degree；无法识别 → None（学历不卡）。
    from resume2job.parsing.jd_parser import DEGREE_RANK
    edu_str = str(config.get("education_filter") or resume_profile.get("highest_degree") or "").strip()
    user_degree_rank = DEGREE_RANK.get(edu_str)

    new_state = dict(state)
    try:
        candidate_jobs = retrieve_jobs(
            resume_profile=resume_profile,
            top_k=top_k,
            city_filter=city_filter,
            user_degree_rank=user_degree_rank,
            job_type_filter=job_type_filter,
            preferences=state.get("preference_tags") or None,  # 方向偏好仅助召回（Query3），排序走评分层
        )
        candidate_jobs = candidate_jobs or []

        # 城市 / 岗位类型是用户显式硬约束：该约束下召回为零（检索 post-filter 已保证绝不拿
        # 别的城市/类型凑数）时提前终止——清空候选、写 final_response 告知并询问是否放宽，
        # 下游评分节点自然跳过，不浪费任何 LLM 评分调用。用户下一轮回复「不限城市」等即可恢复。
        from resume2job.parsing.jd_parser import normalize_city
        want_city = normalize_city(city_filter)
        want_jt = (str(job_type_filter).strip() if job_type_filter else "")
        if (want_city or want_jt) and not candidate_jobs:
            cond = "、".join(x for x in (want_city, (f"{want_jt}岗位" if want_jt else "")) if x) or "当前条件"
            new_state["candidate_jobs"] = []
            new_state["final_response"] = (
                f"知识库暂无满足「{cond}」的岗位，本轮未做推荐。"
                f"要不要放宽城市或岗位类型？回复「不限城市」或换一种岗位类型即可。"
            )
            print(f"[job_retriever_node] 「{cond}」无岗位，提前终止推荐链路。")
            return new_state

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

    # 跨链路一致性：若粘贴的 JD 与知识库中某条已入库 JD 文本完全一致，直接复用入库画像，
    # 使「单 JD 评估」与「岗位推荐」看到同一份结构化 JD，评分不再因重复解析而漂移；同时省一次 LLM。
    reused = lookup_ingested_jd_profile(jd_text)
    if reused:
        new_state["jd_profiles"] = [reused]
        print("[jd_input_node] 命中已入库 JD（按文本哈希），复用入库画像，跳过解析")
        return new_state

    try:
        jd_profile = parse_jd(jd_text)
        # 解析失败可区分（JD 空 / 模型异常 / 输出非 JSON）：给出精确反馈
        if not jd_profile or (isinstance(jd_profile, dict) and jd_profile.get("error")):
            reason = jd_profile.get("error") if isinstance(jd_profile, dict) and jd_profile.get("error") \
                else "JD 解析结果为空"
            stage = jd_profile.get("error_stage") if isinstance(jd_profile, dict) else None
            return append_error(state, f"jd_input_node: JD 解析失败[{stage or 'unknown'}]：{reason}")
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

def _score_one_job(resume_profile: dict, jd_profile: dict, job_id: str,
                   errors: list, user_direction_tags=None):
    """只评分（score_match：project 一次 LLM + 规则）→ match_score，供候选池排序。
    **不**生成 skill_gap / 报告（那是展示岗位才做的惰性叙述）。失败返回 None。"""
    try:
        return score_match(resume_profile, jd_profile, user_direction_tags)
    except Exception as exc:
        errors.append(f"match_scorer_node: 岗位 {job_id} 评分失败：{exc}")
        return None


def _ensure_narrated(mr: dict, resume_profile: dict, errors: list) -> dict:
    """惰性生成「展示岗位」的 skill_gap + 报告（合并为一次 LLM）：

    - 未叙述过（无 _writer 缓存）→ generate_report_and_gap（1 次 LLM 产出 items+reason+suggestion），
      缓存 skill_gap / _writer / report；
    - 已叙述（有 _writer）→ 用缓存 reason/suggestion + 当前 match_score 纯 Python 重组报告
      （换一批/重排后 rank_score 已变，重组刷新分数行，**不再调 LLM**）。
    """
    if not isinstance(mr, dict):
        return mr
    jd = mr.get("jd_profile") or {}
    ms = mr.get("match_score") or {}
    job_id = mr.get("job_id")
    if mr.get("_writer"):
        old_report = mr.get("report") or ""
        new_report = recompose_report(jd, ms, mr.get("skill_gap"), mr["_writer"])
        # 保留上一轮 _run_commute 追加的【通勤】尾段（从 narrative 重组会丢掉它），刷新分数行的同时不丢通勤信息
        m = re.search(r"\n*【通勤】.*$", old_report, re.DOTALL)
        if m and "【通勤】" not in new_report:
            new_report = new_report.rstrip() + "\n\n" + m.group(0).strip()
        mr["report"] = new_report
        return mr
    try:
        report, skill_gap, writer_out = generate_report_and_gap(resume_profile, jd, ms)
        mr["skill_gap"] = skill_gap
        mr["_writer"] = writer_out
        mr["report"] = report
    except Exception as exc:
        errors.append(f"match_scorer_node: 岗位 {job_id} 报告生成失败：{exc}，规则兜底")
        try:
            skill_gap = rule_based_skill_gap(jd, ms)
            mr["skill_gap"] = skill_gap
            mr["_writer"] = {"reason": "", "suggestion": ""}
            mr["report"] = generate_full_report(jd, ms, skill_gap)
        except Exception as exc2:  # 兜底也异常（极端畸形 JD）：给最小可用结果，绝不让单岗中断整批
            errors.append(f"match_scorer_node: 岗位 {job_id} 规则兜底也失败：{exc2}")
            mr["skill_gap"] = {}
            mr["_writer"] = {"reason": "", "suggestion": ""}
            mr["report"] = (f"【推荐岗位】{jd.get('company') or '未知公司'} - {jd.get('title') or '未知岗位'}\n"
                            f"（本岗位报告生成失败）")
    return mr


def _narrate_batch(batch: list, resume_profile: dict, errors: list) -> None:
    """对一批展示岗位并发惰性叙述（已缓存的只重组报告，新岗位才调 LLM）。

    单岗叙述异常被隔离（_ensure_narrated 内已兜底；这里再兜一层），不让任一岗位失败中断整批。
    """
    if not batch:
        return
    workers = min(8, len(batch))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_ensure_narrated, mr, resume_profile, errors) for mr in batch]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception as exc:
                errors.append(f"_narrate_batch: 岗位叙述异常（已隔离）：{exc}")


def _display_k(state: AgentState) -> int:
    """本轮展示岗位数（用户请求数）；候选池可比它大，供换一批/重排复用。"""
    cfg = state.get("retrieval_config") or {}
    return max(1, int(cfg.get("display_k") or cfg.get("top_k") or 5))


def _serve_next_batch(state: AgentState) -> AgentState:
    """NEXT_BATCH：从候选池取下一批未展示岗位（已评分，不重新评分）。"""
    print("[match_scorer_node] 复用候选池：取下一批未展示岗位")
    pool = state.get("candidate_pool") or []
    shown = set(state.get("shown_job_ids") or [])
    display_k = _display_k(state)
    unshown = [r for r in pool if isinstance(r, dict) and r.get("job_id") not in shown]
    new_state = dict(state)
    if not unshown:
        new_state["match_results"] = []
        new_state["final_response"] = (
            "当前候选岗位已全部展示完。可以修改城市、岗位类型或方向偏好后再搜索，我再为你召回一批。"
        )
        print("[match_scorer_node] 候选池已无更多未展示岗位。")
        return new_state
    batch = unshown[:display_k]
    errors = list(state.get("errors") or [])
    _narrate_batch(batch, state.get("resume_profile") or {}, errors)  # 惰性叙述新展示岗位
    new_state["candidate_pool"] = pool  # 池项被就地写入叙述缓存，整池回存
    new_state["match_results"] = batch
    new_state["shown_job_ids"] = list(shown | {r.get("job_id") for r in batch})
    new_state["skill_gaps"] = [{"job_id": r.get("job_id"),
                                "company": (r.get("jd_profile") or {}).get("company"),
                                "title": (r.get("jd_profile") or {}).get("title"),
                                **(r.get("skill_gap") or {})} for r in batch]
    new_state["errors"] = errors
    print(f"[match_scorer_node] 下一批 {len(batch)} 个岗位（池剩余未展示 {len(unshown) - len(batch)}）。")
    return new_state


def _rerank_pool(state: AgentState) -> AgentState:
    """REUSE_RERANK：复用候选池，仅按新方向偏好重算 direction_bonus + rank_score 并重排（不重评分）。

    重排后只对展示 Top-N 惰性叙述：已叙述的纯 Python 重组报告刷新分数行，新进 Top-N 的才调一次合并 LLM。
    """
    print("[match_scorer_node] 复用候选池：重算方向偏好分并重排")
    pool = list(state.get("candidate_pool") or [])
    display_k = _display_k(state)
    user_direction_tags = list((state.get("preference_tags") or {}).keys())
    for r in pool:
        if not isinstance(r, dict):
            continue
        ms = dict(r.get("match_score") or {})
        info = compute_direction_bonus(r.get("jd_profile") or {}, user_direction_tags)
        ms["direction_bonus"] = info["bonus"]
        ms["direction_bonus_info"] = info
        ms["rank_score"] = calculate_rank_score(
            ms.get("match_score", 0), info["bonus"], ms.get("commute_bonus", 0.0),
        )
        ms["match_level"] = judge_match_level(ms["rank_score"])
        r["match_score"] = ms
    pool.sort(key=lambda r: (r.get("match_score") or {}).get("rank_score", 0), reverse=True)
    batch = pool[:display_k]
    errors = list(state.get("errors") or [])
    _narrate_batch(batch, state.get("resume_profile") or {}, errors)  # 缓存→重组报告刷新分数，新岗位→LLM
    new_state = dict(state)
    new_state["candidate_pool"] = pool
    new_state["match_results"] = batch
    # 已展示岗位累加（不复位）：避免后续换一批重复展示已见岗位
    new_state["shown_job_ids"] = list(set(state.get("shown_job_ids") or []) | {r.get("job_id") for r in batch})
    new_state["skill_gaps"] = [{"job_id": r.get("job_id"),
                                "company": (r.get("jd_profile") or {}).get("company"),
                                "title": (r.get("jd_profile") or {}).get("title"),
                                **(r.get("skill_gap") or {})} for r in batch]
    new_state["errors"] = errors
    print(f"[match_scorer_node] 复用池重排完成，展示 Top-{len(batch)}。")
    return new_state


def match_scorer_node(state: AgentState) -> AgentState:
    """评分 + 惰性叙述（解析建议：技能差距与报告合并为一次 LLM、且只对展示岗位生成）。

    会话动作分支（item8）：
      - next_batch：从候选池取下一批未展示岗位，惰性叙述（不重评分）；
      - reuse_pool（REUSE_RERANK）：复用池仅重算方向偏好分重排，惰性叙述 Top-N（不重评分）；
      - 否则（RETRIEVE/USER_JD）：① 整池只评分（project 一次 LLM/岗）→ 按 rank_score 排序；
        ② **只对展示 Top-N 惰性叙述**（一次合并 LLM 产出 skill_gap + 报告），未展示岗位换一批时再叙述。

    单岗评分/叙述失败不中断其它；结果按 rank_score 降序。
    """
    print("[match_scorer_node] 开始执行...")

    # —— 候选池复用分支（不重评分）——
    if get_plan_flag(state, "next_batch"):
        return _serve_next_batch(state)
    if get_plan_flag(state, "reuse_pool") and (state.get("candidate_pool")):
        return _rerank_pool(state)

    need_match = get_plan_flag(state, "need_match_score")
    need_reco = get_plan_flag(state, "need_recommendation")
    need_gap = get_plan_flag(state, "need_skill_gap")

    # 三个开关都为 False -> 跳过（如 SELECTED/ASSIST：match_results 已由 planner 注入）
    if not need_match and not need_reco and not need_gap:
        return state

    resume_profile = state.get("resume_profile")
    if not resume_profile:
        return append_error(state, "match_scorer_node: 缺少 resume_profile，无法评分")

    jd_profiles = state.get("jd_profiles") or []
    if not jd_profiles:
        # 上游已提前终止并给出用户可读说明（如城市硬约束无召回）：静默跳过，不再追加噪音错误
        if state.get("final_response"):
            return state
        return append_error(state, "match_scorer_node: 缺少 jd_profiles，无法评分")

    new_state = dict(state)
    errors = list(state.get("errors") or [])
    user_direction_tags = list((state.get("preference_tags") or {}).keys())

    # 1) 整池只评分（project 一次 LLM/岗 + 规则）→ match_score，并发跑；不生成 skill_gap/报告
    pool: list[MatchResult] = []
    workers = min(8, len(jd_profiles))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for idx, jd_profile in enumerate(jd_profiles):
            job_id = get_job_id(jd_profile, idx)
            fut = ex.submit(_score_one_job, resume_profile, jd_profile, job_id, errors, user_direction_tags)
            futs[fut] = (job_id, jd_profile)
        for fut in concurrent.futures.as_completed(futs):
            job_id, jd_profile = futs[fut]
            match_score = fut.result()
            if match_score is None:  # 评分失败 -> 跳过该岗位
                continue
            pool.append(MatchResult(job_id=job_id, jd_profile=jd_profile,
                                    match_score=match_score, skill_gap={}, report=""))

    # 2) 按 rank_score 降序排序
    pool.sort(key=lambda r: (r.get("match_score") or {}).get("rank_score", 0), reverse=True)

    # 3) 只对展示 Top-N 惰性叙述（一次合并 LLM/岗，产出 skill_gap + 报告）
    display_k = _display_k(state)
    shown = pool[:display_k]
    _narrate_batch(shown, resume_profile, errors)

    new_state["candidate_pool"] = pool                  # 全量评分候选池（含未叙述）供换一批/重排
    new_state["match_results"] = shown
    new_state["shown_job_ids"] = [r.get("job_id") for r in shown]
    new_state["skill_gaps"] = [{"job_id": r.get("job_id"),
                                "company": (r.get("jd_profile") or {}).get("company"),
                                "title": (r.get("jd_profile") or {}).get("title"),
                                **(r.get("skill_gap") or {})} for r in shown]
    new_state["errors"] = errors
    print(f"[match_scorer_node] 评分 {len(pool)} 个（候选池），叙述展示 Top-{len(shown)}（合并调用）。")
    return new_state
