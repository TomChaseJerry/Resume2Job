# -*- coding: utf-8 -*-
"""Planner 的结构化契约。

对象：
    - PlannerOutput ：nlu_extractor 的 LLM Function Calling 产物（只描述「用户想要什么」，不做编排）；
    - CommuteSlot   ：PlannerOutput.commute 子对象（通勤软偏好：报告附时间/路线，给上限则达标加 commute_bonus）。

执行计划（怎么执行）由 policy_orchestrator 依语义 + 会话状态生成为**裸 dict**（session_action +
need_* 开关 + clarify 字段），供 executor / route_job_source / enhancements 读取，不再用 TypedDict 契约。
设计原则：LLM 只产 PlannerOutput（语义），执行计划由确定性规则 + 会话状态生成（编排）。
推荐报告 / JD 适配报告默认含「匹配点 + 技能缺口」，不再由 report_views 单独控制；
学习计划 / 面试练习题仅由 intent=ASSIST + assist_actions 显式驱动（需 SELECTED + 有效 job_id）。
"""

from typing import Optional, List, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# nlu_extractor 的 Function Calling 产物（只抽语义，不做编排）
# ---------------------------------------------------------------------------
class CommuteSlot(BaseModel):
    """通勤槽位（FC 子对象）。"""
    enabled: bool = Field(default=False, description="用户是否提出通勤诉求")
    home_address: Optional[str] = Field(default=None, description="居住地址原文，未提及为 null，禁止编造")
    max_minutes: Optional[int] = Field(default=None, description="通勤时间上限（分钟，「1小时内」→60）")
    transport: Optional[str] = Field(default=None, description="transit/driving/walking/cycling，未提及 null")


class PlannerOutput(BaseModel):
    """规划器语义输出（经 function calling 强制 schema）。只描述「用户想要什么」，不做编排。"""

    intent: Literal["RECOMMEND", "EVALUATE", "ASSIST"] = Field(
        description="RECOMMEND=从岗位库找推荐；EVALUATE=评估某条具体岗位/JD是否适合；"
                    "ASSIST=对已推荐岗位生成学习计划/面试练习题（需 SELECTED + 有效 job_id）"
    )
    job_source: Literal["RETRIEVE", "USER_JD", "SELECTED"] = Field(
        default="RETRIEVE",
        description="岗位来自哪里：RETRIEVE=检索岗位库；USER_JD=用户粘贴的 JD；SELECTED=指代上一轮结果中的某个岗位",
    )
    request_more: bool = Field(
        default=False,
        description="是否请求下一批岗位（『换一批』『多给几个』『还有别的吗』）；从已召回未展示候选中取下一批",
    )
    assist_actions: List[Literal["LEARNING_PLAN", "INTERVIEW_PREP"]] = Field(
        default_factory=list,
        description="仅 intent=ASSIST 时填：要的辅助产出，学习计划 LEARNING_PLAN / 面试练习题 INTERVIEW_PREP（可多选）",
    )
    hard_constraints: dict = Field(
        default_factory=dict,
        description="硬过滤条件，键可含 city / education / job_type（实习/校招/社招，默认实习）；仅来自用户原话。"
                    "学历默认取自简历画像，不要从普通对话推断；用户明说『只看硕士及以上要求的岗位』才写 education。"
                    "『正式岗』含义不明（校招/社招？）→不要臆测，留空交澄清",
    )
    soft_preferences: dict = Field(
        default_factory=dict, description="软偏好（加权非过滤，不进 where），键 direction_tags(list)",
    )
    commute: CommuteSlot = Field(default_factory=CommuteSlot, description="通勤约束（软偏好）")
    selected_item_ref: Optional[str] = Field(
        default=None,
        description="SELECTED/ASSIST 时用户**明确指明的 job_id**（如 job_1，来自上一轮报告）；"
                    "只说『第二个』『那个』等模糊指代而无 job_id 时留 null（会触发追问）",
    )
    missing_slots: List[str] = Field(
        default_factory=list, description="完成当前意图还缺的关键信息（如 resume/jd/selected_item/address）",
    )
    confidence: float = Field(
        default=1.0, description="对本次意图判断的置信度 0~1；不确定时给低分，会触发澄清",
    )
