# -*- coding: utf-8 -*-
"""NLU 抽取：LLM Function Calling，只把用户诉求抽成 PlannerOutput（语义层）。

只负责「理解用户想要什么」，不做任何编排决策（那是 policy_orchestrator 的事）。
LLM 不可用 / 失败时由 rule_fallback 用关键词规则给一个保守的 PlannerOutput，
保证整条链路不因 LLM 故障中断。
"""

from typing import Optional

from resume2job.core.config import PLANNER_MODEL
from resume2job.core.llm import get_chat_llm
from resume2job.agent.planner.schema import PlannerOutput
from resume2job.agent.planner.context_builder import PlannerContext

# ===== 规则兜底关键词 =====
_REC_KW = ("推荐", "找岗位", "找实习", "岗位推荐", "适合我的岗位", "有什么岗位")
_MORE_KW = ("换一批", "再来一批", "多给几个", "还有别的", "还有其他", "再来几个", "下一批", "更多岗位")
_LEARN_KW = ("学习路径", "学习计划", "学习规划", "怎么学", "如何提升", "补齐", "补强", "进阶")
_INTV_KW = ("面试", "面试题", "模拟面试", "面经", "练习题", "怎么答", "作答")
_COMMUTE_KW = ("通勤", "地铁", "公交", "多久到", "小时以内", "路线", "公司地址", "住")
_FOLLOWUP_KW = ("第一个", "第二个", "第三个", "那个", "上面", "刚才", "它", "这个岗位")
_REFINE_KW = ("换成", "换", "改成", "只看", "不要")


_SYSTEM = (
    "你是实习求职岗位推荐 Agent 的 NLU 模块。只负责理解用户诉求，按 schema 输出结构化语义，"
    "不要替用户决定执行步骤。规则：\n"
    "1. 提供了 JD 原文 → job_source=USER_JD，intent=EVALUATE；\n"
    "2. 针对上一轮某个岗位**要学习计划 / 面试练习题** → intent=ASSIST，job_source=SELECTED，"
    "assist_actions 勾选 LEARNING_PLAN / INTERVIEW_PREP（可多选）；\n"
    "3. 针对上一轮某个岗位**追问适配 / 详情**（非辅助）→ intent=EVALUATE，job_source=SELECTED；\n"
    "   2/3 都需用户**明确给出 job_id**（如 job_1）→ 填 selected_item_ref；只说『第二个』『那个』等模糊指代、"
    "无 job_id → selected_item_ref 留 null 并在 missing_slots 加 'selected_item'；\n"
    "4. 在上一轮基础上改条件（『换上海』『只看校招』『更想看后训练』）→ intent=RECOMMEND，job_source=RETRIEVE，"
    "硬约束变化写 hard_constraints、方向偏好写 soft_preferences.direction_tags、通勤写 commute；\n"
    "5. 『换一批 / 多给几个 / 还有别的吗』→ request_more=true（其余不变）；\n"
    "6. 硬约束只含 city / education / job_type（实习/校招/社招）：\n"
    "   - 学历**默认取自简历画像**，不要从普通对话推断用户学历；仅当用户明说『只看硕士及以上要求的岗位』才写 hard_constraints.education；\n"
    "   - job_type：明说『校招』『社招』填对应值；未明说不填（默认实习）；『正式岗』含义不明→**不要臆测**，留空（会澄清）；\n"
    "7. 方向偏好（搜广推/后训练/大模型应用…）入 soft_preferences.direction_tags；"
    "通勤诉求（『通勤1小时内』『给出通勤时间』『地铁多久』）填 commute，均不进 hard_constraints；\n"
    "8. 一切槽位只能来自用户原话与上下文，禁止编造；语义清晰时给高 confidence，"
    "语义模糊 / 缺关键对象 / 条件冲突时才调低 confidence 并在 missing_slots 标注。"
)


def _user_prompt(ctx: PlannerContext) -> str:
    blocks = []
    lr = ctx.last_results_block()
    if lr:
        blocks.append(f"## 上一轮结果（指代解析用）\n{lr}")
    if ctx.active_resume_summary:
        blocks.append(f"## 已有简历画像\n{ctx.active_resume_summary}")
    if ctx.active_jd_summary:
        blocks.append(f"## 当前 JD\n{ctx.active_jd_summary}")
    if ctx.explicit_ui_filters:
        blocks.append(f"## 已显式指定的过滤\n{ctx.explicit_ui_filters}")
    if ctx.recent_dialogue and not lr:
        blocks.append(f"## 最近对话（兜底参考）\n{ctx.recent_dialogue}")
    head = "\n\n".join(blocks)
    return (
        f"{head}\n\n## 本轮用户问题\n{ctx.current_message}\n"
        f"是否提供 JD 原文：{'有' if ctx.has_jd else '无'}\n"
        f"是否提供简历 PDF：{'有' if ctx.has_pdf else '无'}"
    )


def extract(ctx: PlannerContext) -> PlannerOutput:
    """LLM FC 抽语义；失败抛异常交由 node 走 rule_fallback。"""
    llm = get_chat_llm(model=PLANNER_MODEL).with_structured_output(
        PlannerOutput, method="function_calling"
    )
    return llm.invoke([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _user_prompt(ctx)},
    ])


def rule_fallback(ctx: PlannerContext) -> PlannerOutput:
    """LLM 不可用时的关键词兜底。确定性强的输入给较高置信度，仅语义模糊才低置信交澄清。"""
    q = ctx.current_message or ""

    # 指代：只认明确写出的 job_id（在上一轮结果里出现过的）；模糊指代不解析、交澄清
    selected_ref = None
    is_followup = any(k in q for k in _FOLLOWUP_KW)
    for r in (ctx.last_results or []):
        jid = r.get("job_id")
        if jid and jid in q:
            selected_ref = jid
            is_followup = True
            break

    want_learn = any(k in q for k in _LEARN_KW)
    want_intv = any(k in q for k in _INTV_KW)
    request_more = any(k in q for k in _MORE_KW)
    assist_actions = []

    # —— 意图判定 + 置信度分级（确定性强 → 高置信）——
    confidence = 0.85
    if ctx.has_jd:                                         # 本轮给了 JD
        intent, source = "EVALUATE", "USER_JD"
    elif (want_learn or want_intv) and (selected_ref or is_followup):  # 指定岗位要辅助
        intent, source = "ASSIST", "SELECTED"
        if want_learn:
            assist_actions.append("LEARNING_PLAN")
        if want_intv:
            assist_actions.append("INTERVIEW_PREP")
        if not selected_ref:                              # 模糊指代无 job_id → 低置信澄清
            confidence = 0.4
    elif selected_ref or is_followup:                     # 指定岗位追问适配/详情
        intent, source = "EVALUATE", "SELECTED"
        if not selected_ref:
            confidence = 0.4
    elif request_more:                                    # 换一批
        intent, source = "RECOMMEND", "RETRIEVE"
    elif any(k in q for k in _REFINE_KW) or any(k in q for k in _REC_KW):
        intent, source = "RECOMMEND", "RETRIEVE"
    else:
        intent, source = "RECOMMEND", "RETRIEVE"
        confidence = 0.6                                  # 无明确意图词 → 略低，默认推荐

    commute = {"enabled": any(k in q for k in _COMMUTE_KW),
               "transport": "transit", "home_address": None, "max_minutes": None}

    return PlannerOutput(
        intent=intent, job_source=source, request_more=request_more,
        assist_actions=assist_actions,
        commute=commute, selected_item_ref=selected_ref,
        confidence=confidence,
    )
