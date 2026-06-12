"""
agent/nodes/planner.py

Function Calling 规划节点（旧 intent_router + planner + commute_planner 三合一）。

单次 LLM 结构化输出（OpenAI 兼容 function calling，langchain
with_structured_output）一并完成：
    1. 任务类型分类 task_type（job_recommendation / jd_evaluation / skill_gap_only）；
    2. 检索过滤条件抽取（city / direction / education）——取代旧 chat.py 里
       脆弱的「常见城市词」关键词匹配；
    3. 通勤约束抽取（住址 / 时间上限 / 交通方式）——取代旧 commute_planner 的
       独立 LLM 调用；住址城市回填 retrieval_config 并扩大召回。

LLM 与规则的分工：
    - LLM 负责语义理解（结构化输出由 function calling 保证 schema 合法）；
    - task_type → PlanConfig 的展开是确定性的 Python 规则（可预测、零成本）；
    - LLM 不可用 / 失败时退回关键词规则兜底，最终默认 job_recommendation。

可选增强（通勤计算 / 学习计划 / 面试题）不在本节点决策——由
enhancement_node 的 Tool Calling 在评分完成后自主决定。
"""

from typing import Optional, Literal

from pydantic import BaseModel, Field

from resume2job.agent.state import AgentState, PlanConfig
from resume2job.core.config import PLANNER_MODEL
from resume2job.core.llm import get_chat_llm
from resume2job.tools.commute import extract_city, DEFAULT_TRANSPORT, VALID_TRANSPORT

# 通勤约束触发时，检索召回的扩大目标（保证有足够候选供过滤）
COMMUTE_TOP_K = 10

# 三个合法的任务类型
VALID_TASK_TYPES = ("job_recommendation", "jd_evaluation", "skill_gap_only")

# ===== 规则兜底用关键词 =====
JD_EVAL_KEYWORDS = ("适合", "分析", "能不能投", "匹配", "评价")
SKILL_GAP_KEYWORDS = ("差距", "缺哪些", "补什么", "skill gap", "能力差")
JOB_REC_KEYWORDS = ("推荐", "找岗位", "找实习", "岗位推荐", "适合我的岗位")
COMMUTE_KEYWORDS = (
    "通勤", "地铁", "距离", "多久到", "小时以内",
    "1小时", "一个小时", "路线", "公司地址",
)


# ---------------------------------------------------------------------------
# Function Calling 输出 Schema
# ---------------------------------------------------------------------------
class PlannerOutput(BaseModel):
    """规划器的结构化输出（经 function calling 强制 schema）。"""

    task_type: Literal["job_recommendation", "jd_evaluation", "skill_gap_only"] = Field(
        description="任务类型：job_recommendation=从岗位库推荐岗位；"
                    "jd_evaluation=评估用户提供的某条 JD 是否适合；"
                    "skill_gap_only=只分析能力差距，不出推荐报告"
    )
    city_filter: Optional[str] = Field(
        default=None, description="用户希望限定的工作城市（如「北京」），未提及则为 null"
    )
    direction_filter: Optional[str] = Field(
        default=None, description="用户希望限定的岗位方向（如「大模型算法」），未提及则为 null"
    )
    education_filter: Optional[str] = Field(
        default=None, description="用户提到的学历限定（本科/硕士/博士），未提及则为 null"
    )
    has_commute_constraint: bool = Field(
        default=False, description="用户是否提出了通勤约束（给了住址或通勤时间要求）"
    )
    commute_address: Optional[str] = Field(
        default=None,
        description="用户当前居住地址原文（如「北京市海淀区五道口」），未提及则为 null，禁止编造",
    )
    max_commute_minutes: Optional[int] = Field(
        default=None, description="通勤时间上限，统一换算为分钟（「1小时以内」→60），未提及则为 null"
    )
    preferred_transport: Optional[str] = Field(
        default=None,
        description="交通方式：transit（地铁/公交）/ driving / walking / cycling，未提及则为 null",
    )


_PLANNER_SYSTEM = (
    "你是一个实习岗位推荐 Agent 的规划器。根据最近对话和用户本轮问题，"
    "判断任务类型并抽取检索过滤条件与通勤约束。\n"
    "规则：\n"
    "1. 用户提供了 JD 原文时，任务只可能是 jd_evaluation 或 skill_gap_only；\n"
    "2. 过滤条件与通勤信息只能来自用户原话（含对话上下文，如「换成上海的」），禁止凭空补全；\n"
    "3. 多轮追问要结合上下文理解（如「那第二个岗位呢」延续上一轮任务类型）。"
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _append_error(state: AgentState, message: str) -> list:
    """安全地把一条错误/警告信息追加到 errors 列表并返回新列表。"""
    errors = list(state.get("errors") or [])
    errors.append(message)
    return errors


def _format_history(messages: Optional[list], max_chars: int = 800) -> str:
    """把 state['messages'] 渲染为简短对话文本，供规划参考上下文。超长保留最近部分。"""
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


def _rule_based_intent(user_query: str, jd_text: Optional[str]) -> Optional[str]:
    """规则兜底意图识别（LLM 不可用时）。"""
    query = user_query or ""
    if jd_text and any(kw in query for kw in JD_EVAL_KEYWORDS):
        return "jd_evaluation"
    if any(kw.lower() in query.lower() for kw in SKILL_GAP_KEYWORDS):
        return "skill_gap_only"
    if any(kw in query for kw in JOB_REC_KEYWORDS):
        return "job_recommendation"
    return None


def _llm_plan(user_query: str, has_jd: bool, has_pdf: bool,
              history_text: str) -> PlannerOutput:
    """调用 LLM 做 function calling 结构化规划。失败抛异常交由上层兜底。"""
    llm = get_chat_llm(model=PLANNER_MODEL).with_structured_output(
        PlannerOutput, method="function_calling"
    )
    history_block = f"## 最近对话\n{history_text}\n\n" if history_text else ""
    user_prompt = (
        f"{history_block}"
        f"## 本轮用户问题\n{user_query}\n"
        f"是否提供了 JD 原文：{'有' if has_jd else '无'}\n"
        f"是否提供了简历 PDF：{'有' if has_pdf else '无'}"
    )
    return llm.invoke([
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": user_prompt},
    ])


def _build_plan(task_type: str, jd_text: Optional[str]) -> PlanConfig:
    """task_type → PlanConfig 的确定性展开（不调用 LLM）。

    规划是固定的业务流程编排，要求可预测、可复现、零调用成本，
    因此刻意用纯规则映射而非 LLM。
    """
    if task_type == "jd_evaluation":
        return {
            "need_resume_parse": True,
            "need_job_search": False,
            "need_jd_input": True,
            "need_jd_parse": True,
            "need_match_score": True,
            "need_recommendation": True,
            "need_skill_gap": True,
        }
    if task_type == "skill_gap_only":
        # 有 JD 则基于 JD 做 gap；无 JD 则后续基于目标方向做 gap。
        return {
            "need_resume_parse": True,
            "need_job_search": False,
            "need_jd_input": bool(jd_text),
            "need_jd_parse": bool(jd_text),
            "need_match_score": False,
            "need_recommendation": False,
            "need_skill_gap": True,
        }
    # 默认 / job_recommendation
    return {
        "need_resume_parse": True,
        "need_job_search": True,
        "need_jd_input": False,
        "need_jd_parse": False,
        "need_match_score": True,
        "need_recommendation": True,
        "need_skill_gap": True,
    }


def _build_commute_intent(out: PlannerOutput) -> dict:
    """把规划输出整理成 commute 工具使用的 intent 结构。"""
    transport = (out.preferred_transport or "").strip().lower()
    if transport not in VALID_TRANSPORT:
        transport = DEFAULT_TRANSPORT
    address = (out.commute_address or "").strip() or None
    return {
        "has_commute_constraint": bool(
            out.has_commute_constraint and (address or out.max_commute_minutes)
        ),
        "user_address": address,
        "address_confidence": "district" if address else "vague",
        "max_commute_minutes": out.max_commute_minutes,
        "preferred_transport": transport,
        "raw_address_text": address,
        # 用户表达了通勤诉求但没给地址：记录原因，commute 工具会在报告中显式说明
        "error": None if address else "缺少可解析地址",
    }


# ---------------------------------------------------------------------------
# 节点：planner_node
# ---------------------------------------------------------------------------
def planner_node(state: AgentState) -> AgentState:
    """Function Calling 规划节点。

    写入：task_type / plan / commute_intent（有通勤诉求时）/
          retrieval_config（LLM 抽取的过滤条件回填，不覆盖用户显式传参）。
    """
    print("[planner_node] 开始执行...")
    new_state = dict(state)

    user_query = state.get("user_query") or ""
    jd_text = state.get("jd_text")
    pdf_path = state.get("pdf_path")
    has_jd, has_pdf = bool(jd_text), bool(pdf_path)

    history_text = _format_history(state.get("messages"))

    # 1. LLM Function Calling 规划；失败走关键词规则兜底
    out: Optional[PlannerOutput] = None
    try:
        out = _llm_plan(user_query, has_jd, has_pdf, history_text)
    except Exception as exc:
        new_state["errors"] = _append_error(
            new_state, f"planner_node: LLM 规划失败（{exc}），改用规则兜底"
        )

    if out is not None:
        task_type = out.task_type
    else:
        task_type = _rule_based_intent(user_query, jd_text) or "job_recommendation"
        # 兜底路径下构造空规划输出，统一后续处理
        out = PlannerOutput(
            task_type=task_type,
            has_commute_constraint=any(kw in user_query for kw in COMMUTE_KEYWORDS),
        )

    # 2. 结构信号硬约束：显式提供 JD 原文时绝不该是 job_recommendation
    #    （否则会忽略用户给的 JD 转而去向量库另找岗位）。LLM 偶发误判，规则强制纠正。
    if has_jd and task_type == "job_recommendation":
        corrected = ("skill_gap_only"
                     if any(kw.lower() in user_query.lower() for kw in SKILL_GAP_KEYWORDS)
                     else "jd_evaluation")
        new_state["errors"] = _append_error(
            new_state,
            f"planner_node: 已提供 JD，task_type 由 job_recommendation 纠正为 {corrected}",
        )
        task_type = corrected

    new_state["task_type"] = task_type
    new_state["plan"] = _build_plan(task_type, jd_text)

    # 3. 检索过滤条件回填（不覆盖用户显式传入的 CLI 参数）
    rc = dict(state.get("retrieval_config") or {})
    for key, value in (("city_filter", out.city_filter),
                       ("direction_filter", out.direction_filter),
                       ("education_filter", out.education_filter)):
        if value and str(value).strip() and not rc.get(key):
            rc[key] = str(value).strip()

    # 4. 通勤诉求：写入 commute_intent，住址城市回填 + 扩大召回
    if out.has_commute_constraint or any(kw in user_query for kw in COMMUTE_KEYWORDS):
        intent = _build_commute_intent(out)
        new_state["commute_intent"] = intent
        city = extract_city(intent.get("user_address"))
        if city and not rc.get("city_filter"):
            rc["city_filter"] = city
        rc["top_k"] = max(int(rc.get("top_k") or 5), COMMUTE_TOP_K)
        print(f"[planner_node] 通勤约束：住址={intent.get('user_address')}，"
              f"上限={intent.get('max_commute_minutes')}分钟，top_k={rc['top_k']}")

    new_state["retrieval_config"] = rc

    # 5. 前置依赖校验（仅告警，不阻断流程；缓存画像可在 profile_cache 兜底）
    if not has_pdf:
        new_state["errors"] = _append_error(
            new_state, f"planner_node: {task_type} 任务需要简历，当前未上传（将尝试缓存画像）"
        )
    if task_type == "jd_evaluation" and not has_jd:
        new_state["errors"] = _append_error(
            new_state, "planner_node: jd_evaluation 任务需要 JD 原文，但当前 jd_text 为空"
        )

    active_filters = {k: v for k, v in rc.items() if v}
    print(f"[planner_node] task_type={task_type}，检索参数={active_filters}")
    return new_state
