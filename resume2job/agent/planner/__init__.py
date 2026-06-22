"""Planner 包：对外只暴露 planner_node（graph.py 导入入口保持稳定）。

内部按职责拆分为：
    schema           结构化契约（PlannerOutput / CommuteSlot）
    context_builder  组装 PlannerContext（当前输入 + 会话短期状态 + 结构化摘要）
    nlu_extractor    LLM Function Calling，只抽语义
    rule_corrector   确定性纠错
    clarification    缺槽 / 低置信度的澄清策略
    policy_orchestrator  语义 → 执行计划 dict（need_* 开关）
    trace_logger     落 planner 决策样本（后训练数据闭环，第一阶段只写不训）
    node             planner_node：编排以上六步
"""

from resume2job.agent.planner.node import planner_node

__all__ = ["planner_node"]
