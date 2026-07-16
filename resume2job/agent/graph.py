"""
agent/graph.py

把所有 LangGraph 节点组装为「基于 Agentic RAG 的实习岗位智能推荐系统」的工作流主图。

本文件只负责：
    1. 用 StateGraph 组装节点；
    2. 添加固定边与条件路由；
    3. 编译并返回可执行 app；
    4. 提供本地调试用的 main 入口与多轮对话封装 run_turn。

不在此实现任何业务逻辑（简历解析、JD 解析、岗位检索、评分、报告生成均在节点中完成）。

主流程：
    START
      ↓
    planner（Function Calling：intent + job_source + assist_actions + 硬约束 + 通勤，规则兜底）
      ↓
    resume_parser -> profile_cache（复用缓存画像 / 保存新画像）
      ↓ (条件路由 route_job_source)
      ├── job_retriever（混合检索：BM25+向量+RRF+rerank）→ jd_analyzer
      ├── jd_input → jd_ingest（粘贴 JD 自动去重入库）→ jd_analyzer
      ├── jd_analyzer
      └── END
      ↓
    match_scorer（评分 + 技能差距 + 推荐报告）
      ↓
    enhancements（Tool Calling：LLM 自主决定调用 通勤 / 学习计划 / 面试题 工具）
      ↓
    END
"""

import argparse
import time
import uuid

from langgraph.graph import StateGraph, START, END

from resume2job.agent.state import AgentState, get_initial_state
from resume2job.agent.trace import TurnTrace, summarize_state, changed_keys
from resume2job.observability import events
from resume2job.agent.planner import planner_node
from resume2job.agent.nodes.executor import (
    resume_parser_node,
    job_retriever_node,
    jd_input_node,
    jd_analyzer_node,
    match_scorer_node,
)
from resume2job.agent.nodes.enhancements import enhancement_node
from resume2job.storage.profile_cache import profile_cache_node
from resume2job.storage.jd_ingest import jd_ingest_node
from resume2job.storage import conversation_store, session_store


# ---------------------------------------------------------------------------
# 条件路由函数
# ---------------------------------------------------------------------------

def route_job_source(state: AgentState) -> str:
    """
    岗位来源路由函数（不是节点，只读 state、只返回路由标签，不修改 state）。

    根据 plan 决定 profile_cache 之后走向哪条分支：
        - 候选池复用/换一批/SELECTED/ASSIST -> match_scorer（直达，不重检索/不重评分）
        - 需要检索岗位                      -> job_retriever
        - 用户直接粘贴 JD                   -> jd_input
        - 需要解析/评分/推荐                -> jd_analyzer
        - 澄清/其余                         -> end

    返回值只能是：match_scorer / job_retriever / jd_input / jd_analyzer / end
    """
    plan = state.get("plan")
    if not plan:
        return "end"
    # 澄清短路：planner 判定需追问时不跑任何业务节点，直达 END（final_response 已设澄清问题）
    if plan.get("clarify"):
        return "end"
    # 候选池复用（换一批 / 重排）+ SELECTED/ASSIST（match_results 已由 planner 注入）：
    # 直达 match_scorer（不重检索/不重评分），再走 enhancements
    if plan.get("reuse_pool") or plan.get("next_batch"):
        return "match_scorer"
    if plan.get("need_job_search"):
        return "job_retriever"
    if plan.get("need_jd_input"):
        return "jd_input"
    if plan.get("need_jd_parse"):
        return "jd_analyzer"
    if plan.get("need_match_score") or plan.get("need_recommendation"):
        return "jd_analyzer"
    return "end"


# ---------------------------------------------------------------------------
# 构建工作流图
# ---------------------------------------------------------------------------

def _observed(name: str, fn):
    """包住节点：经 events.run_node 记录每节点时延 + 设「当前节点」（供 LLM token 归属）。

    无活跃 request trace 时 run_node 直接透传（零开销旁路），故 run_turn_traced / pipeline / 直接
    invoke 均不受影响——只有 run_turn 开了 request_scope 的链路才会被观测。
    """
    def _wrapped(state):
        return events.run_node(name, fn, state)
    _wrapped.__name__ = getattr(fn, "__name__", name)
    return _wrapped


def build_graph():
    """组装并编译 LangGraph 工作流，返回可执行 app。"""
    graph = StateGraph(AgentState)

    # 1. 注册节点（route_job_source 是路由函数，不在此注册）；每个节点包一层观测旁路
    graph.add_node("planner", _observed("planner", planner_node))              # Function Calling 规划
    graph.add_node("resume_parser", _observed("resume_parser", resume_parser_node))
    graph.add_node("profile_cache", _observed("profile_cache", profile_cache_node))  # 画像缓存：复用 / 保存用户画像
    graph.add_node("job_retriever", _observed("job_retriever", job_retriever_node))  # 混合检索召回
    graph.add_node("jd_input", _observed("jd_input", jd_input_node))
    graph.add_node("jd_ingest", _observed("jd_ingest", jd_ingest_node))          # 粘贴 JD 自动去重入库
    graph.add_node("jd_analyzer", _observed("jd_analyzer", jd_analyzer_node))
    graph.add_node("match_scorer", _observed("match_scorer", match_scorer_node))
    graph.add_node("enhancements", _observed("enhancements", enhancement_node))     # Tool Calling 可选增强

    # 2. 固定边：入口 -> 规划 -> 简历解析 -> 画像缓存
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "resume_parser")
    graph.add_edge("resume_parser", "profile_cache")

    # 3. 条件路由：profile_cache 之后按 plan 决定岗位来源分支
    graph.add_conditional_edges(
        "profile_cache",
        route_job_source,
        {
            "job_retriever": "job_retriever",
            "jd_input": "jd_input",
            "jd_analyzer": "jd_analyzer",
            "match_scorer": "match_scorer",   # 候选池复用 / SELECTED / ASSIST 直达评分节点
            "end": END,
        },
    )

    # 4. 固定边：各岗位来源分支汇聚到 jd_analyzer -> match_scorer -> enhancements -> END
    graph.add_edge("job_retriever", "jd_analyzer")
    graph.add_edge("jd_input", "jd_ingest")
    graph.add_edge("jd_ingest", "jd_analyzer")
    graph.add_edge("jd_analyzer", "match_scorer")
    graph.add_edge("match_scorer", "enhancements")
    graph.add_edge("enhancements", END)

    # 5. 编译并返回
    return graph.compile()


# ---------------------------------------------------------------------------
# 多轮对话：单轮执行封装（加载历史 -> 执行图 -> 记录对话）
# ---------------------------------------------------------------------------

def _summarize_response(final_state: AgentState) -> str:
    """从最终 State 提炼一段面向用户/可入库的助手回复摘要。"""
    # 1. 通勤等汇总文案优先
    final_response = (final_state.get("final_response") or "").strip()
    parts = []
    if final_response:
        parts.append(final_response)

    # 2. 岗位结果摘要（取前 3 条）
    match_results = final_state.get("match_results") or []
    if match_results:
        lines = []
        for mr in match_results[:3]:
            jd = mr.get("jd_profile") or {}
            score = (mr.get("match_score") or {}).get("rank_score")
            company = jd.get("company") or "未知公司"
            title = jd.get("title") or "未知岗位"
            lines.append(f"[{mr.get('job_id')}] {company} - {title}（推荐分 {score}）")
        parts.append("为你匹配到以下岗位（追问某岗请指明方括号内的岗位 ID）：\n" + "\n".join(lines))

    # 3. 纯技能差距分析摘要（skill_gap_only：无 match_results 时）
    skill_gaps = final_state.get("skill_gaps") or []
    if not match_results and skill_gaps:
        sg = skill_gaps[0]
        head = " - ".join(x for x in (sg.get("company"), sg.get("title")) if x) or "目标岗位"
        top = (sg.get("top_suggestion") or "").strip()
        summary = f"已完成对照「{head}」的能力差距分析。"
        if top:
            summary += f"重点建议：{top}"
        parts.append(summary)

    # 4. ASSIST：学习计划 / 面试练习题一句话（若有）
    plan = final_state.get("learning_plan") or {}
    if plan.get("overall_suggestion"):
        parts.append("学习建议：" + plan["overall_suggestion"])
    interview = final_state.get("interview_prep") or {}
    if interview.get("questions"):
        parts.append(f"已生成 {len(interview['questions'])} 道岗位定制面试练习题。")

    if not parts:
        return "本轮未生成岗位结果。"
    return "\n\n".join(parts)


def _persist_session(session_id: str, final_state: AgentState) -> None:
    """把本轮结果写入会话短期状态，供下一轮指代解析、换一批/重排与硬约束变化判定。

    存：
      - last_results：本轮**展示**的岗位轻量列表 [{rank, job_id, title, city}]，喂下一轮 planner 指代；
      - candidate_pool：完整已评分候选池（含未展示），供 SELECTED/ASSIST 读详情、换一批、重排；
      - shown_job_ids：已展示 job_id（换一批从未展示中取）；
      - active_hard_constraints / active_direction_tags：会话活跃约束/偏好，供 policy 判定「硬约束是否变化」。
    SELECTED/ASSIST 轮（selected_item_ref 非空）不覆盖列表/池，避免把多岗位压成单条。
    """
    if not session_id or final_state.get("selected_item_ref"):
        return
    mrs = final_state.get("match_results") or []
    if not mrs:
        return
    last_results = []
    for i, mr in enumerate(mrs, start=1):
        jd = mr.get("jd_profile") or {}
        loc = jd.get("location") or {}
        last_results.append({"rank": i, "job_id": mr.get("job_id"),
                             "title": jd.get("title"), "city": loc.get("city")})

    pool = final_state.get("candidate_pool") or mrs
    rc = final_state.get("retrieval_config") or {}
    sess = session_store.get_session(session_id)
    sess["last_results"] = last_results
    sess["candidate_pool"] = pool              # 整池（含未展示）：SELECTED/ASSIST 读详情、换一批、重排复用
    sess["shown_job_ids"] = final_state.get("shown_job_ids") or [r.get("job_id") for r in mrs]
    sess["active_hard_constraints"] = {
        "city": rc.get("city_filter"),
        "education": rc.get("education_filter"),
        "job_type": rc.get("job_type_filter"),
    }
    sess["active_direction_tags"] = list((final_state.get("preference_tags") or {}).keys())
    session_store.set_session(session_id, sess)


def run_turn(
    app,
    user_query: str,
    session_id: str = conversation_store.DEFAULT_SESSION_ID,
    pdf_path=None,
    jd_text=None,
    top_k: int = 5,
    city_filter=None,
    education_filter=None,
) -> AgentState:
    """执行一轮多轮对话：

        1. 按 session_id 读取历史对话，注入 state["messages"]（供规划节点参考上下文）；
        2. 执行工作流；
        3. 把本轮 user 提问与 assistant 回复摘要写入对话记录（持久化）。

    画像复用由 profile_cache_node 负责（首轮上传后续可不传简历），
    对话上下文由本函数 + conversation_store 负责，二者共同构成「多轮记忆」。
    """
    history = conversation_store.load_history(session_id)

    initial_state = get_initial_state(
        user_query=user_query,
        pdf_path=pdf_path,
        jd_text=jd_text,
        top_k=top_k,
        city_filter=city_filter,
        education_filter=education_filter,
        session_id=session_id,
        messages=history,
    )

    # 请求级链路追踪（Stage 2）：分配 request_id，开 request_scope（退出时落盘 trace）。
    # 各节点 / core.llm / retriever 沿途经 contextvar 记录器写快照；失败不影响主链路。
    turn_no = len(history) // 2 + 1
    request_id = f"req-{turn_no}-{uuid.uuid4().hex[:10]}"
    with events.request_scope(request_id, session_id, user_query,
                              model_versions=events.snapshot_model_versions(),
                              index_version=events.snapshot_index_version()):
        final_state = app.invoke(initial_state)
        events.record_state_errors((final_state or {}).get("errors") or [])

    # 记录本轮对话（user + assistant）+ 写会话短期状态（供下一轮指代）
    response = _summarize_response(final_state)
    conversation_store.append_turn(session_id, user_query, response)
    _persist_session(session_id, final_state)
    # 把 request_id 附到返回（replay / 调用方关联用；额外键，不影响 State 通道）
    final_state = dict(final_state)
    final_state["request_id"] = request_id
    return final_state


def run_turn_traced(
    app,
    user_query: str,
    session_id: str = conversation_store.DEFAULT_SESSION_ID,
    pdf_path=None,
    jd_text=None,
    top_k: int = 5,
    city_filter=None,
    education_filter=None,
):
    """与 run_turn 等价的执行，但额外返回逐节点链路 trace（回归测试 / 调试用）。

    实现上用 app.stream(stream_mode="updates") 替代 app.invoke：LangGraph 在每个节点
    执行后吐出 {node_name: 该节点返回的产物}。本函数据此：
        1. 累加产物还原完整 final_state（节点均返回整份 State，累加无副作用）；
        2. 对每个节点 diff 出改动字段，连同关键信号快照、耗时、新增错误记入 TurnTrace。
    节点本身零改动。返回 (final_state, trace)。

    trace_id 取 f"{session_id}#{轮次}"，轮次由已有对话历史推算（每轮 = user+assistant 两条）。
    """
    history = conversation_store.load_history(session_id)

    initial_state = get_initial_state(
        user_query=user_query,
        pdf_path=pdf_path,
        jd_text=jd_text,
        top_k=top_k,
        city_filter=city_filter,
        education_filter=education_filter,
        session_id=session_id,
        messages=history,
    )

    turn_no = len(history) // 2 + 1
    trace = TurnTrace(f"{session_id}#{turn_no}", session_id, user_query)

    final_state = dict(initial_state)
    last_t = time.time()
    for chunk in app.stream(initial_state, stream_mode="updates"):
        now = time.time()
        # 线性图通常每个 chunk 只含一个节点；并行 super-step 会含多个，统一遍历
        for node_name, node_out in chunk.items():
            prev = final_state
            merged = {**prev, **(node_out or {})}
            prev_errs = prev.get("errors") or []
            cur_errs = merged.get("errors") or []
            trace.record(
                node=node_name,
                elapsed_s=now - last_t,
                changed=changed_keys(prev, merged),
                summary=summarize_state(merged),
                errors_added=cur_errs[len(prev_errs):],
            )
            final_state = merged
        last_t = now

    response = _summarize_response(final_state)
    conversation_store.append_turn(session_id, user_query, response)
    _persist_session(session_id, final_state)
    return final_state, trace


# ---------------------------------------------------------------------------
# main 本地调试入口
# ---------------------------------------------------------------------------

def main():
    """本地调试入口：只做参数读取、状态初始化、图执行和结果打印，不含业务逻辑。"""
    parser = argparse.ArgumentParser(description="Agentic RAG 实习岗位推荐工作流")
    parser.add_argument("--query", required=True, help="用户的自然语言请求")
    parser.add_argument("--pdf_path", default=None, help="简历 PDF 路径")
    parser.add_argument("--jd_file", default=None, help="JD 原文文件路径（UTF-8）")
    parser.add_argument("--top_k", type=int, default=5, help="召回岗位数量")
    parser.add_argument("--city", default=None, help="城市硬约束")
    parser.add_argument("--education", default=None, help="学历硬约束")
    parser.add_argument("--session_id", default=conversation_store.DEFAULT_SESSION_ID,
                        help="会话 ID（同一 ID 跨次运行共享对话历史，实现多轮记忆）")
    args = parser.parse_args()

    # 如提供 jd_file，则 UTF-8 读取其内容作为 jd_text
    jd_text = None
    if args.jd_file:
        with open(args.jd_file, "r", encoding="utf-8") as f:
            jd_text = f.read()

    # 构建工作流，并以「单轮对话」方式执行（自动加载历史 + 记录本轮）
    app = build_graph()
    final_state = run_turn(
        app,
        user_query=args.query,
        session_id=args.session_id,
        pdf_path=args.pdf_path,
        jd_text=jd_text,
        top_k=args.top_k,
        city_filter=args.city,
        education_filter=args.education,
    )

    # 打印结果
    match_results = final_state.get("match_results") or []
    errors = final_state.get("errors") or []

    print("\n" + "=" * 60)
    print("工作流执行完成")
    print("=" * 60)

    # 通勤汇总（若有）
    final_response = final_state.get("final_response")
    if final_response:
        print(f"\n[通勤汇总] {final_response}")

    if match_results:
        for idx, result in enumerate(match_results, start=1):
            print(f"\n--- 岗位 {idx}（job_id={result.get('job_id')}）---")
            report = result.get("report") or "（本岗位无推荐报告）"
            print(report)
    elif errors:
        print("\n未生成匹配结果，错误信息如下：")
        for err in errors:
            print(f"- {err}")
    else:
        print("\n本次执行没有生成任何匹配结果，也没有错误信息。")


if __name__ == "__main__":
    main()
