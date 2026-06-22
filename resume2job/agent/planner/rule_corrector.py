# -*- coding: utf-8 -*-
"""确定性纠错：在 LLM 语义输出之上加不依赖 LLM 的硬护栏。

「跑错链路比多问一句更糟」——这里用规则纠正 LLM 偶发误判，并补标缺失槽位，
为后续 clarification / policy_orchestrator 提供可靠输入。
"""

from resume2job.core.config import PLANNER_CONFIDENCE_MIN
from resume2job.agent.planner.schema import PlannerOutput
from resume2job.agent.planner.context_builder import PlannerContext

# 「正式岗」含义不明（校招/社招？）→ 触发澄清，不臆测
_AMBIGUOUS_FULLTIME_KW = ("正式岗", "正式岗位", "正式工作")
# 仅这些显式诉求才允许把岗位学历要求写进 hard_constraints.education（否则视为误抽，学历取自简历）
_EDU_FILTER_CUES = ("及以上", "学历要求", "只看", "限硕", "限博", "要求硕", "要求博")


def correct(out: PlannerOutput, ctx: PlannerContext) -> PlannerOutput:
    """就地纠正并返回 out（同一对象）。规则：

    1. 有 JD 原文却判成检索/推荐 → 强制 USER_JD + EVALUATE（绝不忽略用户给的 JD）；
    2. SELECTED/ASSIST 指代：仅当 selected_item_ref 是 last_results 里的明确 job_id 才认可，
       否则标 selected_item 缺槽，交澄清追问；
    3. ASSIST 必须有有效 job_id 且至少一个 assist_action（默认补全）；
    4. 学历护栏：无显式『只看…要求/及以上』诉求时不接受 hard_constraints.education（学历取自简历）；
    5. 『正式岗』含义不明 → 标 job_type_ambiguous 缺槽交澄清；
    6. 缺简历/缺 JD 等前置 → 补 missing_slots；低置信度 → 标 intent 让 clarification 接管。
    """
    q = ctx.current_message or ""

    # 1) JD 硬护栏（结构信号纠正）
    if ctx.has_jd and (out.job_source != "USER_JD" or out.intent not in ("EVALUATE", "ASSIST")):
        out.job_source = "USER_JD"
        out.intent = "EVALUATE"

    # 2) 指代解析：只接受**已展示岗位**的明确 job_id（last_results 或会话累计 shown_job_ids）；
    #    解析不到 → 追问用户指明 job_id（保证 _pick_last_result 能取到已叙述岗位）
    if out.job_source == "SELECTED" or out.intent == "ASSIST":
        shown_ids = (ctx.session_state or {}).get("shown_job_ids")
        resolved = _resolve_selected(out.selected_item_ref, ctx.last_results, shown_ids)
        if resolved:
            out.selected_item_ref = resolved
        else:
            out.selected_item_ref = None
            _add_missing(out, "selected_item")

    # 3) ASSIST 默认补全 assist_actions（既未指明则两者都给）
    if out.intent == "ASSIST" and not out.assist_actions:
        out.assist_actions = ["LEARNING_PLAN", "INTERVIEW_PREP"]

    # 4) 学历护栏：仅显式诉求才保留 hard_constraints.education，否则剔除（默认取简历画像学历）
    if isinstance(out.hard_constraints, dict) and out.hard_constraints.get("education"):
        if not any(c in q for c in _EDU_FILTER_CUES):
            out.hard_constraints.pop("education", None)

    # 5) 『正式岗』含义不明 → 澄清（不自动映射为校招/社招）
    if any(k in q for k in _AMBIGUOUS_FULLTIME_KW) and not (
            isinstance(out.hard_constraints, dict) and out.hard_constraints.get("job_type")):
        _add_missing(out, "job_type_ambiguous")

    # 6) 前置依赖缺失：推荐 / JD 评估 / 辅助都需简历画像
    need_resume = (out.intent in ("RECOMMEND", "ASSIST")
                   or (out.intent == "EVALUATE" and out.job_source == "USER_JD"))
    if need_resume and not ctx.has_pdf and not ctx.active_resume_summary:
        _add_missing(out, "resume")          # 无简历且缓存无画像
    if out.job_source == "USER_JD" and not ctx.has_jd:
        _add_missing(out, "jd")              # 判为评估 JD 却没给 JD

    # 低置信度：交澄清，不默认跑推荐
    if out.confidence < PLANNER_CONFIDENCE_MIN:
        _add_missing(out, "intent")

    return out


def _resolve_selected(ref, last_results: list, shown_ids=None):
    """仅当 ref 是**已展示岗位**的明确 job_id 才返回它；否则 None（含模糊指代/无上轮结果）。

    已展示集合 = 会话累计 shown_job_ids ∪ 上一轮 last_results（跨批次的已展示岗位都可指代）。
    不支持「第二个」这类位置指代——用户须从报告里指明 job_id。
    """
    if not ref:
        return None
    ref = str(ref).strip()
    if ref in set(shown_ids or []):
        return ref
    for r in (last_results or []):
        if r.get("job_id") == ref:
            return ref
    return None


def _add_missing(out: PlannerOutput, slot: str) -> None:
    if slot not in out.missing_slots:
        out.missing_slots.append(slot)
