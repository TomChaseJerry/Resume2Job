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
    planner（Function Calling：task_type + 检索过滤 + 通勤意图，规则兜底）
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

from langgraph.graph import StateGraph, START, END

from resume2job.agent.state import AgentState, get_initial_state
from resume2job.agent.nodes.planner import planner_node
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
from resume2job.storage import conversation_store


# ---------------------------------------------------------------------------
# 条件路由函数
# ---------------------------------------------------------------------------

def route_job_source(state: AgentState) -> str:
    """
    岗位来源路由函数（不是节点，只读 state、只返回路由标签，不修改 state）。

    根据 plan 决定 profile_cache 之后走向哪条分支：
        - 需要检索岗位        -> job_retriever
        - 用户直接粘贴 JD     -> jd_input
        - 需要解析/评分/推荐  -> jd_analyzer
        - 其余                -> end

    返回值只能是：job_retriever / jd_input / jd_analyzer / end
    """
    plan = state.get("plan")
    if not plan:
        return "end"
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

def build_graph():
    """组装并编译 LangGraph 工作流，返回可执行 app。"""
    graph = StateGraph(AgentState)

    # 1. 注册节点（route_job_source 是路由函数，不在此注册）
    graph.add_node("planner", planner_node)              # Function Calling 规划
    graph.add_node("resume_parser", resume_parser_node)
    graph.add_node("profile_cache", profile_cache_node)  # 画像缓存：复用 / 保存用户画像
    graph.add_node("job_retriever", job_retriever_node)  # 混合检索召回
    graph.add_node("jd_input", jd_input_node)
    graph.add_node("jd_ingest", jd_ingest_node)          # 粘贴 JD 自动去重入库
    graph.add_node("jd_analyzer", jd_analyzer_node)
    graph.add_node("match_scorer", match_scorer_node)
    graph.add_node("enhancements", enhancement_node)     # Tool Calling 可选增强

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
            score = (mr.get("match_score") or {}).get("final_score")
            company = jd.get("company") or "未知公司"
            title = jd.get("title") or "未知岗位"
            lines.append(f"{company} - {title}（匹配度 {score}）")
        parts.append("为你匹配到以下岗位：\n" + "\n".join(lines))

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

    # 4. 学习计划一句话（若有）
    plan = final_state.get("learning_plan") or {}
    if plan.get("overall_suggestion"):
        parts.append("学习建议：" + plan["overall_suggestion"])

    if not parts:
        return "本轮未生成岗位结果。"
    return "\n\n".join(parts)


def run_turn(
    app,
    user_query: str,
    session_id: str = conversation_store.DEFAULT_SESSION_ID,
    pdf_path=None,
    jd_text=None,
    top_k: int = 5,
    city_filter=None,
    direction_filter=None,
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
        direction_filter=direction_filter,
        education_filter=education_filter,
        session_id=session_id,
        messages=history,
    )

    final_state = app.invoke(initial_state)

    # 记录本轮对话（user + assistant）
    response = _summarize_response(final_state)
    conversation_store.append_turn(session_id, user_query, response)
    return final_state


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
    parser.add_argument("--city", default=None, help="城市过滤条件")
    parser.add_argument("--direction", default=None, help="岗位方向过滤条件")
    parser.add_argument("--education", default=None, help="学历过滤条件")
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
        direction_filter=args.direction,
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
