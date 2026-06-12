"""
state.py

LangGraph 工作流的全局 State 定义。

本模块属于「基于 Agentic RAG 的实习岗位智能推荐系统」的工作流基础设施，
负责串联简历解析、岗位检索、JD 解析、匹配评分、报告生成等节点。

设计约定：
    - 所有 LangGraph 节点都从 `AgentState` 读取输入；
    - 每个节点把自己的输出写回 `AgentState` 对应字段；
    - 节点之间不直接互相调用，只通过共享的 `AgentState` 传递数据。

本文件只定义 State 类型与初始化函数，不包含任何节点逻辑或具体业务实现。
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal


# ---------------------------------------------------------------------------
# 1. PlanConfig
# ---------------------------------------------------------------------------
class PlanConfig(TypedDict):
    """
    Planner 节点生成的任务规划（主链路开关）。

    每个布尔字段表示某个子任务是否需要执行，
    后续节点据此决定自身是否被激活（条件分支 / 跳过）。

    注意：通勤 / 学习计划 / 面试题等可选增强不在此处——它们由
    enhancement_node 的 LLM Tool Calling 在评分完成后自主决定是否调用。
    """
    need_resume_parse: bool   # 是否需要解析简历 PDF
    need_job_search: bool     # 是否需要从岗位知识库检索岗位
    need_jd_input: bool       # 用户是否直接提供 JD 原文
    need_jd_parse: bool       # 是否需要解析 JD
    need_match_score: bool    # 是否需要进行岗位匹配评分
    need_recommendation: bool # 是否需要生成推荐报告
    need_skill_gap: bool      # 是否需要技能差距分析


# ---------------------------------------------------------------------------
# 2. RetrievalConfig
# ---------------------------------------------------------------------------
class RetrievalConfig(TypedDict):
    """
    岗位检索参数，由用户输入初始化，供 Job Search 节点使用。
    """
    top_k: int                          # 召回岗位数量
    city_filter: Optional[str]          # 城市过滤条件，例如「北京」
    direction_filter: Optional[str]     # 岗位方向过滤条件，例如「大模型算法」
    education_filter: Optional[str]     # 学历过滤条件，例如「硕士」


# ---------------------------------------------------------------------------
# 3. MatchResult
# ---------------------------------------------------------------------------
class MatchResult(TypedDict):
    """
    单个岗位的匹配结果与推荐报告。

    由 Match Scorer 与 Recommendation Writer 节点共同填充，
    是「岗位 -> 评分 -> 报告」的聚合产物。
    """
    job_id: str                     # 岗位 ID
    jd_profile: Dict[str, Any]      # 结构化 JD（JD Parser 输出）
    match_score: Dict[str, Any]     # Match Scorer 输出
    skill_gap: Dict[str, Any]       # Skill Gap Analyzer 输出（Stage 4 内部分析）
    report: str                     # Recommendation Writer 输出的推荐报告


# ---------------------------------------------------------------------------
# 4. AgentState
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    全局工作流状态，被所有 LangGraph 节点共享读写。

    字段按来源分为三类：
        - 用户输入：工作流启动时由 get_initial_state 填充；
        - 中间结果：由各业务节点在执行过程中写入；
        - 最终输出：汇总节点写入，返回给用户。
    """

    # ---------- 用户输入（启动时填充，节点只读） ----------
    user_query: str                 # 用户的自然语言请求
    jd_text: Optional[str]          # 用户直接粘贴的 JD 原文（场景 2）
    pdf_path: Optional[str]         # 用户上传的简历 PDF 路径（场景 1）

    # ---------- 多轮对话记忆（Conversation Store 注入 / 回写） ----------
    session_id: str                 # 会话标识，用于读写对话历史，默认 ""
    messages: List[Dict[str, Any]]  # 本轮注入的历史消息 [{role, content, ...}]

    # ---------- 任务规划（Planner 节点写入） ----------
    task_type: Optional[Literal["job_recommendation", "jd_evaluation", "skill_gap_only"]]
    plan: Optional[PlanConfig]      # 各子任务是否执行的规划

    # ---------- 通勤意图（Commute Planner 节点写入） ----------
    commute_intent: Optional[Dict[str, Any]]  # 住址 / 时间上限 / 交通方式等通勤约束

    # ---------- 检索配置（启动时填充，Job Search 节点读取） ----------
    retrieval_config: RetrievalConfig

    # ---------- 简历画像（Resume Parser 节点写入） ----------
    resume_profile: Optional[Dict[str, Any]]  # 结构化简历

    # ---------- 用户画像缓存（Profile Cache 节点写入） ----------
    profile_id: str           # 当前使用的画像 profile_id，默认 ""
    profile_source: str       # "new_upload" | "cached" | ""（未加载）

    # ---------- 岗位检索结果（Job Search 节点写入） ----------
    search_queries: Dict[str, str]            # 检索使用的查询语句
    candidate_jobs: List[Dict[str, Any]]      # 召回的候选岗位列表

    # ---------- JD 解析结果（JD Parser 节点写入） ----------
    jd_profiles: List[Dict[str, Any]]         # 结构化 JD 列表

    # ---------- 匹配评分与推荐报告（Match Scorer + Recommendation Writer 写入） ----------
    match_results: List[MatchResult]          # 每个岗位的评分 + 报告聚合

    # ---------- Skill Gap 分析结果（Stage 4 / Skill Gap 节点写入） ----------
    skill_gaps: List[Dict[str, Any]]

    # ---------- 学习路径规划（Learning Plan 节点写入，按需启用） ----------
    learning_plan: Dict[str, Any]             # 基于 match_results[0].skill_gap 生成的阶段化学习计划

    # ---------- 面试辅助（Interview Prep 节点写入，按需启用） ----------
    interview_prep: Dict[str, Any]            # 模拟面试题（generate_interview_questions 输出）

    # ---------- JD 自动入库结果（JD Ingest 节点写入） ----------
    ingested_job_id: str      # 本次入库或复用的 job_id，默认 ""
    jd_is_duplicate: bool     # 本次 JD 是否为重复，默认 False

    # ---------- 通勤计算结果（enhancement_node 的通勤工具写入） ----------
    commute_results: List[Dict[str, Any]]
    commute_note: Optional[str]               # 通勤约束不可用时的说明文案

    # ---------- 最终输出（汇总节点写入，返回给用户） ----------
    final_response: Optional[str]             # 汇总后的最终回复

    # ---------- 错误信息（任意节点可追加） ----------
    # 统一为字符串，格式约定 "<node>: <message>"
    errors: List[str]


# ---------------------------------------------------------------------------
# 初始化函数
# ---------------------------------------------------------------------------
def get_initial_state(
    user_query: str,
    pdf_path: Optional[str] = None,
    jd_text: Optional[str] = None,
    top_k: int = 5,
    city_filter: Optional[str] = None,
    direction_filter: Optional[str] = None,
    education_filter: Optional[str] = None,
    session_id: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
) -> AgentState:
    """
    构造工作流的初始 `AgentState`。

    仅填充用户输入与检索配置；其余字段为初始空值，
    等待后续 LangGraph 节点逐步写入。

    Args:
        user_query:        用户的自然语言请求（必填）。
        pdf_path:          简历 PDF 路径，默认为 None。
        jd_text:           用户粘贴的 JD 原文，默认为 None。
        top_k:             召回岗位数量，默认为 5。
        city_filter:       城市过滤条件。
        direction_filter:  岗位方向过滤条件。
        education_filter:  学历过滤条件。

    Returns:
        完整初始化的 AgentState。
    """
    return AgentState(
        # 用户输入
        user_query=user_query,
        jd_text=jd_text,
        pdf_path=pdf_path,

        # 多轮对话记忆
        session_id=session_id,
        messages=messages or [],

        # 任务规划：由 Planner 节点后续填充
        task_type=None,
        plan=None,

        # 通勤意图：由 Commute Planner 节点后续填充
        commute_intent=None,

        # 检索配置
        retrieval_config=RetrievalConfig(
            top_k=top_k,
            city_filter=city_filter,
            direction_filter=direction_filter,
            education_filter=education_filter,
        ),

        # 简历画像：尚未生成
        resume_profile=None,

        # 用户画像缓存：尚未加载
        profile_id="",
        profile_source="",

        # 岗位检索结果：列表 / 字典初始化为空
        search_queries={},
        candidate_jobs=[],

        # JD 解析结果
        jd_profiles=[],

        # 匹配评分与推荐报告
        match_results=[],

        # Skill Gap 分析结果
        skill_gaps=[],

        # 学习路径规划：尚未生成
        learning_plan={},

        # 面试辅助：尚未生成
        interview_prep={},

        # JD 自动入库结果：尚未入库
        ingested_job_id="",
        jd_is_duplicate=False,

        # 通勤计算结果
        commute_results=[],
        commute_note=None,

        # 最终输出：尚未生成
        final_response=None,

        # 错误信息
        errors=[],
    )
