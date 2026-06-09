"""
nodes/planner.py

LangGraph 工作流的「意图识别」与「任务规划」节点。

本模块属于「基于 Agentic RAG 的实习岗位智能推荐系统」的工作流调度部分，
负责决定后续是否执行：简历解析、岗位检索、JD 解析、匹配评分、
推荐报告生成、Skill Gap 分析、通勤计算等节点。

包含两个 LangGraph 节点：
    1. intent_router_node：判断任务类型 -> 写入 state["task_type"]；
    2. planner_node：依据 task_type 生成执行计划 -> 写入 state["plan"]。

两者的分工：
    - intent_router_node 使用 LLM 做「语义级」意图分类（用户到底想干什么），
      LLM 不可靠时退回到 Python 关键词规则兜底；
    - planner_node 不使用 LLM，而是用确定性的 Python 规则把 task_type
      展开成布尔计划。规划是固定的业务流程编排，必须可预测、可复现、零成本，
      用 LLM 反而会引入不确定性，因此这里刻意用规则实现。
"""

import os
from typing import Optional

from state import AgentState, PlanConfig


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 三个合法的任务类型
VALID_TASK_TYPES = ("job_recommendation", "jd_evaluation", "skill_gap_only")

# 意图分类用更快的轻量模型：输出仅三选一标签，不需要 qwen-max 的重推理，
# qwen-plus 足够且明显更快。可用环境变量覆盖（回退 qwen-max）。
INTENT_MODEL_NAME = os.environ.get("RESUME2JOB_INTENT_MODEL", "qwen-plus")

# 通勤相关关键词：命中任意一个则认为用户关心通勤，需开启 need_commute
COMMUTE_KEYWORDS = (
    "通勤", "地铁", "距离", "多久到", "小时以内",
    "1小时", "一个小时", "路线", "公司地址",
)

# 规则兜底用关键词
JD_EVAL_KEYWORDS = ("适合", "分析", "能不能投", "匹配", "评价")
SKILL_GAP_KEYWORDS = ("差距", "缺哪些", "补什么", "skill gap", "能力差")
JOB_REC_KEYWORDS = ("推荐", "找岗位", "找实习", "岗位推荐", "适合我的岗位")

# 可选增强模块关键词：命中才开启对应工具（默认关闭）
LEARNING_PLAN_KEYWORDS = (
    "学习路径", "学习计划", "学习规划", "学习路线", "进阶路线",
    "怎么学", "如何学", "如何提升", "提升计划", "补齐", "补强",
)
INTERVIEW_KEYWORDS = (
    "面试", "面试题", "模拟面试", "面试准备", "面试辅导", "面试问题",
    "面经", "怎么答", "作答思路", "面试官",
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _append_error(state: AgentState, message: str) -> list:
    """
    安全地把一条错误/警告信息追加到 errors 列表并返回新列表。

    不假设 state 中一定存在 errors 字段，也不就地修改原列表（浅拷贝语义）。
    """
    errors = list(state.get("errors") or [])
    errors.append(message)
    return errors


def _is_valid_task_type(value: Optional[str]) -> bool:
    """校验是否为三个合法 task_type 之一。"""
    return value in VALID_TASK_TYPES


def _clean_llm_output(text: str) -> str:
    """
    清洗 LLM 输出，尽量提取出合法 task_type：
        - 去除首尾空白、反引号、星号等 Markdown 符号；
        - 转为小写；
        - 在文本中扫描三个合法值，命中即返回；
        - 未命中返回空字符串。
    """
    if not text:
        return ""
    cleaned = text.strip().strip("`*# \n\t").lower()
    # 直接命中
    if cleaned in VALID_TASK_TYPES:
        return cleaned
    # LLM 可能夹带多余文字，做包含式匹配
    for t in VALID_TASK_TYPES:
        if t in cleaned:
            return t
    return ""


def _rule_based_intent(
    user_query: str,
    jd_text: Optional[str],
) -> Optional[str]:
    """
    规则兜底意图识别。

    优先级与任务说明保持一致：
        1. 有 JD + 评价类词 -> jd_evaluation；
        2. 能力差距类词 -> skill_gap_only；
        3. 推荐类词 -> job_recommendation；
        4. 都未命中 -> 返回 None（交由调用方决定默认值）。
    """
    query = (user_query or "")

    # 规则 1：提供了 JD 且问题里有评价/匹配类词
    if jd_text and any(kw in query for kw in JD_EVAL_KEYWORDS):
        return "jd_evaluation"

    # 规则 2：能力差距类
    if any(kw.lower() in query.lower() for kw in SKILL_GAP_KEYWORDS):
        return "skill_gap_only"

    # 规则 3：推荐岗位类
    if any(kw in query for kw in JOB_REC_KEYWORDS):
        return "job_recommendation"

    return None


def _has_commute_intent(user_query: str) -> bool:
    """检测用户问题中是否包含通勤相关关键词。"""
    query = user_query or ""
    return any(kw in query for kw in COMMUTE_KEYWORDS)


def _format_history(messages: Optional[list], max_chars: int = 800) -> str:
    """把 state['messages'] 渲染为简短对话文本，供意图识别参考上下文。超长保留最近部分。"""
    if not messages:
        return ""
    label = {"user": "用户", "assistant": "助手"}
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = str(m.get("content") or "").strip()
        if content:
            lines.append(f"{label.get(m.get('role'), m.get('role'))}：{content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "…（更早对话省略）\n" + text[-max_chars:]
    return text


def _llm_classify_intent(
    user_query: str,
    has_jd: bool,
    has_pdf: bool,
    history_text: str = "",
) -> str:
    """
    调用 LLM 做意图分类，返回清洗后的 task_type（可能为空字符串）。

    多轮场景下传入 history_text（最近对话），帮助理解依赖上下文的追问
    （如「换成上海的」「那第二个岗位呢」）。
    若环境变量缺失或调用失败，抛出异常交由上层捕获并走规则兜底。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        # 主动抛错，让上层走兜底分支并记录原因
        raise RuntimeError("DASHSCOPE_API_KEY 未配置")

    # 延迟导入，避免无 LLM 依赖的环境在 import 阶段就失败
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=INTENT_MODEL_NAME,
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )

    system_prompt = (
        "你是一个求职助手的意图分类器。你的任务是结合最近对话历史、用户问题和是否提供 JD，"
        "判断当前任务类型。\n"
        "你只能输出以下三个值之一：\n"
        "job_recommendation\n"
        "jd_evaluation\n"
        "skill_gap_only\n"
        "不要输出任何解释、标点、Markdown 或额外文字。"
    )

    history_block = f"## 最近对话\n{history_text}\n\n" if history_text else ""
    user_prompt = (
        f"{history_block}"
        f"## 本轮用户问题\n{user_query}\n"
        f"是否提供了 JD 原文：{'有' if has_jd else '无'}\n"
        f"是否提供了简历 PDF：{'有' if has_pdf else '无'}\n\n"
        "请结合上下文判断任务类型，只输出：\n"
        "job_recommendation / jd_evaluation / skill_gap_only"
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return _clean_llm_output(getattr(response, "content", "") or "")


# ---------------------------------------------------------------------------
# 节点 1：intent_router_node
# ---------------------------------------------------------------------------

def intent_router_node(state: AgentState) -> AgentState:
    """
    意图识别节点。

    作用：根据用户输入判断任务类型，写入 state["task_type"]。

    LLM 与规则兜底的关系：
        - 先用 LLM 做语义分类（更能理解自然语言表达）；
        - LLM 不可用 / 调用失败 / 输出非法时，退回到 Python 关键词规则；
        - 规则也无法判定时，最终默认 job_recommendation，并记录一条 warning。
    本节点只更新 task_type（必要时追加 errors），不触碰其它字段。
    """
    new_state = dict(state)

    user_query = state.get("user_query") or ""
    jd_text = state.get("jd_text")
    pdf_path = state.get("pdf_path")

    has_jd = bool(jd_text)
    has_pdf = bool(pdf_path)

    history_text = _format_history(state.get("messages"))

    task_type = ""
    llm_failed_reason = None

    # 1. 优先尝试 LLM 分类（带最近对话上下文）
    try:
        task_type = _llm_classify_intent(user_query, has_jd, has_pdf, history_text)
    except Exception as exc:  # 环境变量缺失 / 网络 / SDK 错误等
        llm_failed_reason = str(exc)
        task_type = ""

    if llm_failed_reason:
        new_state["errors"] = _append_error(
            new_state,
            f"intent_router_node: LLM 意图识别失败（{llm_failed_reason}），改用规则兜底",
        )

    # 2. LLM 输出非法 -> 规则兜底
    if not _is_valid_task_type(task_type):
        rule_result = _rule_based_intent(user_query, jd_text)
        if rule_result:
            task_type = rule_result
        else:
            # 3. 规则也无法判定 -> 默认 job_recommendation 并告警
            task_type = "job_recommendation"
            new_state["errors"] = _append_error(
                new_state,
                "intent_router_node: LLM 输出非法 task_type，已回退为 job_recommendation",
            )

    new_state["task_type"] = task_type
    return new_state


# ---------------------------------------------------------------------------
# 节点 2：planner_node
# ---------------------------------------------------------------------------

def _build_plan(
    task_type: str,
    user_query: str,
    jd_text: Optional[str],
) -> PlanConfig:
    """
    依据 task_type 构建 PlanConfig（确定性规则，不调用 LLM）。

    PlanConfig 字段含义：
        need_resume_parse   是否解析简历 PDF
        need_job_search     是否检索岗位知识库
        need_jd_input       是否使用用户提供的 JD 原文
        need_jd_parse       是否解析 JD
        need_match_score    是否进行匹配评分
        need_recommendation 是否生成推荐报告
        need_skill_gap      是否做技能差距分析
        need_commute        是否做通勤计算
    """
    if task_type == "jd_evaluation":
        plan: PlanConfig = {
            "need_resume_parse": True,
            "need_job_search": False,
            "need_jd_input": True,
            "need_jd_parse": True,
            "need_match_score": True,
            "need_recommendation": True,
            "need_skill_gap": True,
            "need_commute": False,
        }
    elif task_type == "skill_gap_only":
        # 有 JD 则基于 JD 做 gap；无 JD 则后续基于目标方向做 gap。
        # Skill Gap 场景通常不需要推荐报告。
        plan = {
            "need_resume_parse": True,
            "need_job_search": False,
            "need_jd_input": bool(jd_text),
            "need_jd_parse": bool(jd_text),
            "need_match_score": False,
            "need_recommendation": False,
            "need_skill_gap": True,
            "need_commute": False,
        }
    else:
        # 默认 / job_recommendation
        plan = {
            "need_resume_parse": True,
            "need_job_search": True,
            "need_jd_input": False,
            "need_jd_parse": False,
            "need_match_score": True,
            "need_recommendation": True,
            "need_skill_gap": True,
            "need_commute": False,
        }

    # 通勤关键词检测：命中则开启 need_commute（对所有任务类型生效）
    if _has_commute_intent(user_query):
        plan["need_commute"] = True

    # 可选增强模块：默认关闭，仅当用户问题中出现相应说法时才开启（对所有任务类型生效）
    q = user_query or ""
    plan["need_learning_plan"] = any(kw in q for kw in LEARNING_PLAN_KEYWORDS)
    plan["need_interview_prep"] = any(kw in q for kw in INTERVIEW_KEYWORDS)

    return plan


def planner_node(state: AgentState) -> AgentState:
    """
    任务规划节点。

    作用：根据 task_type、用户输入生成执行计划，写入 state["plan"]。

    为什么不用 LLM 而用 Python 规则：
        规划是固定的业务流程编排（哪些节点该跑、哪些该跳过），
        要求确定、可复现、可调试、零调用成本；用 LLM 会带来不确定性
        与额外延迟/费用，得不偿失。因此这里用纯规则映射。

    本节点只更新 plan（必要时追加 errors），不触碰其它字段。
    """
    new_state = dict(state)

    task_type = state.get("task_type")
    user_query = state.get("user_query") or ""
    jd_text = state.get("jd_text")
    pdf_path = state.get("pdf_path")

    errors: list = list(state.get("errors") or [])

    # 1. task_type 缺失或非法 -> 回退 job_recommendation
    if not _is_valid_task_type(task_type):
        errors.append(
            "planner_node: task_type 缺失或非法，已回退为 job_recommendation"
        )
        task_type = "job_recommendation"

    # 2. 构建计划
    plan = _build_plan(task_type, user_query, jd_text)

    # 3. 前置依赖校验（仅告警，不阻断流程）
    if task_type == "job_recommendation":
        if not pdf_path:
            errors.append(
                "planner_node: job_recommendation 任务需要简历 PDF，但当前 pdf_path 为空"
            )
    elif task_type == "jd_evaluation":
        if not jd_text:
            errors.append(
                "planner_node: jd_evaluation 任务需要 JD 原文，但当前 jd_text 为空"
            )
        if not pdf_path:
            errors.append(
                "planner_node: jd_evaluation 任务需要简历 PDF，但当前 pdf_path 为空"
            )
    elif task_type == "skill_gap_only":
        if not pdf_path:
            errors.append(
                "planner_node: skill_gap_only 任务需要简历 PDF，但当前 pdf_path 为空"
            )

    new_state["plan"] = plan
    new_state["errors"] = errors
    return new_state
