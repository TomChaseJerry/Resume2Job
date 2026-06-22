# -*- coding: utf-8 -*-
"""澄清策略：按**当前任务**检查必需信息 → 一句面向用户的澄清问题。

「跑错链路比多问一句更糟」。不再用固定 resume→jd→selected→intent 顺序，
而是据 intent / job_source / commute 判断本任务真正缺什么。无需澄清返回 None。
"""

from typing import Optional

from resume2job.agent.planner.schema import PlannerOutput
from resume2job.agent.planner.context_builder import PlannerContext

_Q_RESUME = "我还没有你的简历画像，请先上传简历 PDF，或简单说说你的技能与方向，我才能帮你推荐/评估。"
_Q_JD = "你想评估哪个岗位？请把岗位 JD 原文发给我。"
_Q_SELECTED = "请提供推荐报告中的 job_id（如 job_1）再追问该岗位。"
_Q_ORIGIN = "要按通勤筛选/计算，请告诉我出发地（如『北京海淀区中关村』）。"
_Q_FULLTIME = "你说的『正式岗』是指校招（应届/校园招聘）还是社招（社会招聘）？请明确一下。"
_Q_INTENT = "我不太确定你的诉求——你是想推荐岗位、评估某个岗位，还是针对某岗位出学习计划/面试练习题？"


def _commute_needs_origin(out: PlannerOutput) -> bool:
    """通勤需实际计算（设了时间上限）却没出发地 → 需追问出发地。"""
    c = out.commute
    return bool(c and c.enabled and c.max_minutes and not c.home_address)


def decide(out: PlannerOutput, ctx: PlannerContext) -> Optional[str]:
    """据当前任务检查必需信息，返回澄清问题；无需澄清返回 None。"""
    missing = set(out.missing_slots or [])

    # 1) 高优先：含义不明 / 低置信意图（先消解再谈任务必需信息）
    if "job_type_ambiguous" in missing:
        return _Q_FULLTIME
    if "intent" in missing:
        return _Q_INTENT

    # 2) 任务式必需信息
    if out.intent == "RECOMMEND":
        if "resume" in missing:
            return _Q_RESUME
    elif out.intent == "EVALUATE" and out.job_source == "USER_JD":
        if "jd" in missing:
            return _Q_JD
        if "resume" in missing:
            return _Q_RESUME
    elif out.intent == "ASSIST" or out.job_source == "SELECTED":
        # 指定岗位分析 / 辅助：需有效 job_id（+ 简历画像）
        if "selected_item" in missing:
            return _Q_SELECTED
        if "resume" in missing:
            return _Q_RESUME

    # 3) 通勤计算需出发地
    if _commute_needs_origin(out):
        return _Q_ORIGIN

    return None
