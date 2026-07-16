# -*- coding: utf-8 -*-
"""
scripts/verify_observability.py —— Stage 2（resume2job/observability）请求级链路追踪的可复盘验收脚本。

默认运行：全程**不调 API、不改业务库**（用临时落盘路径做 events 自检 + 离线单测，再只读真实 request_traces 存储）。
加 --with-api：额外真实跑一轮 run_turn（解析 / 检索 / 评分，会调 API），展示采集到的完整 trace；
              若库里已有 ≥2 条 trace，再演示 diff_traces 前后对比（不额外调 API）。

用法：
    python scripts/verify_observability.py                # 离线体检（推荐先跑）
    python scripts/verify_observability.py --with-api     # 额外真实采一条 + 展示
"""

import os
import sys
import json
import argparse
import tempfile
import contextvars
import concurrent.futures

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def hr(t): print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)
def ok(m): print("  [OK] " + m)
def info(m): print("  - " + m)


def section_imports():
    hr("1. 模块导入 & 计算图自检（无 API）—— 证明 observability 接线无导入环")
    from resume2job.agent.graph import build_graph
    build_graph()
    from resume2job.observability import events, redaction, replay  # noqa: F401
    ok("observability(events/redaction/replay) 可导入；LangGraph 可编译（graph/planner/retriever/executor 接入 events 无环）")


def _use_temp_store(events):
    """把 events 落盘路径临时切到临时目录，返回 (原路径, 临时目录) 供恢复。"""
    orig = (events.JSONL_PATH, events.DB_PATH)
    tmp = tempfile.mkdtemp(prefix="obs_verify_")
    events.JSONL_PATH = os.path.join(tmp, "rt.jsonl")
    events.DB_PATH = os.path.join(tmp, "rt.db")
    return orig, tmp


def _restore_store(events, orig, tmp):
    import shutil
    events.JSONL_PATH, events.DB_PATH = orig
    shutil.rmtree(tmp, ignore_errors=True)


def section_events_offline():
    hr("2. events 记录器离线自检（临时落盘，不污染真实库）")
    from resume2job.observability import events
    orig, tmp = _use_temp_store(events)
    try:
        _section_events_offline_body(events)
    finally:
        _restore_store(events, orig, tmp)


def _section_events_offline_body(events):
    with events.request_scope("verify-1", "sess", "北京 大模型 实习",
                              model_versions={"chat_model": "x"}, index_version={"embedding_version": "y"}):
        events.record_query_plan({"intent": "RECOMMEND", "session_action": "RETRIEVE"})
        events.record_constraint_filter({"city": "北京", "job_type": "实习"}, allowed_count=24)
        events.record_retrieval_queries({"query_1": "大模型 应用", "query_2": "RAG", "query_3": "大模型应用"})
        events.record_channel_hits("dense", [{"job_id": "a", "vector_score": 0.8}, {"job_id": "b", "vector_score": 0.7}])
        events.record_channel_hits("bm25", [{"job_id": "b", "bm25_score": 0.9}])
        events.record_rrf([{"job_id": "b", "retrieval_score": 0.03}, {"job_id": "a", "retrieval_score": 0.02}])
        events.record_rerank([{"job_id": "b", "rerank_score": 0.95}])
        events.record_final_ranked([{"job_id": "b", "rank": 1, "rank_score": 82}])
        events.record_tool_call("commute", 1200.0, summary={"n": 1})

        def _node(_):
            events.record_llm_call("chat", "m", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, 800.0)
            return _
        events.run_node("match_scorer", _node, {})

    t = events.get_trace("verify-1")
    assert t and t["token_usage"]["by_node"].get("match_scorer") == 150 and t["latency_ms"].get("match_scorer") is not None
    assert t["retrieval"]["allowed_count"] == 24 and t["final_ranked_jobs"][0]["job_id"] == "b"
    ok(f"全程记录 + 落盘回读：token by_node={t['token_usage']['by_node']}，时延节点={list(t['latency_ms'])}")
    assert events.record_feedback("verify-1", "saved")
    ok("user_feedback 回填成功（SQLite 摘要表）")
    # no-op 安全
    assert events.current() is None
    events.record_query_plan({"x": 1}); events.record_llm_call("chat", "m", None, 1.0)
    ok("无活跃 trace 时所有 record_* 均 no-op（pipeline / eval / 直接调用不受影响）")


def section_thread_propagation():
    hr("3. 线程上下文传播自检（copy_context；无 API）—— 评分 narration 在线程池里也能被采到")
    from resume2job.observability import events
    orig, tmp = _use_temp_store(events)
    try:
        _section_thread_propagation_body(events)
    finally:
        _restore_store(events, orig, tmp)


def _section_thread_propagation_body(events):
    with events.request_scope("thr-1", "s", "q") as tr:
        def _node(_):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futs = [ex.submit(contextvars.copy_context().run,
                                  lambda: events.record_llm_call("chat", "m", {"total_tokens": 100}, 1.0))
                        for _ in range(3)]
                for f in futs:
                    f.result()
            return _
        events.run_node("match_scorer", _node, {})
        total = tr._token_usage()["total_tokens"]
    info(f"线程池内 3 次调用（各 100 token）采集合计：{total}")
    assert total == 300, f"线程上下文未传播，仅采到 {total}"
    ok("copy_context 让线程池子任务的 LLM token 被正确归集（修复 _narrate_batch 的线程盲区）")


def section_redaction():
    hr("4. 脱敏 redaction 自检（无 API）—— 保留技能/学历/城市，移除身份信息")
    from resume2job.observability import redaction
    prof = {"name": "张三", "contact": {"email": "a@b.com", "phone": "13800138000", "github": "github.com/z"},
            "current_location": "北京", "skills": ["PyTorch", "RAG"], "highest_degree": "硕士"}
    red = redaction.redact_resume_profile(prof)
    info(f"contact 脱敏后：{red['contact']}；name={red['name']}；skills 保留={red['skills']}；学历={red['highest_degree']}")
    assert red["name"] == "<NAME>" and red["contact"]["email"] == "<EMAIL>" and red["contact"]["phone"] == "<PHONE>"
    assert red["skills"] == ["PyTorch", "RAG"] and red["current_location"] == "北京"
    assert redaction.redact_text("身份证 11010119900307861X 手机 13800138000 邮箱 x@y.com") == "身份证 <ID> 手机 <PHONE> 邮箱 <EMAIL>"
    ok("身份证/手机/邮箱/URL 命中替换；技能/学历/城市原样保留")


def section_replay_diff():
    hr("5. replay 前后对比自检（合成两条 trace；无 API）")
    from resume2job.observability import replay
    old = {"request_id": "r1", "final_ranked_jobs": [{"job_id": "a", "rank": 1, "rank_score": 80}, {"job_id": "b", "rank": 2, "rank_score": 70}],
           "token_usage": {"total_tokens": 1000}, "latency_ms": {"total": 5000}, "model_versions": {"m": "v1"}, "errors": []}
    new = {"request_id": "r1", "final_ranked_jobs": [{"job_id": "b", "rank": 1, "rank_score": 85}, {"job_id": "c", "rank": 2, "rank_score": 75}],
           "token_usage": {"total_tokens": 1200}, "latency_ms": {"total": 4000}, "model_versions": {"m": "v2"}, "errors": []}
    d = replay.diff_traces(old, new, k=2)
    info("diff 摘要：" + d["summary"])
    assert d["added"] == ["c"] and d["removed"] == ["a"] and d["token_delta"] == 200 and d["model_versions_changed"]
    ok("diff_traces 正确识别 Top-K 变化 / 名次分变化 / token·时延 delta / 版本变更")


def section_real_store():
    hr("6. 真实 request_traces 存储速览（只读）")
    from resume2job.observability import events
    import sqlite3
    traces = events.load_traces()
    info(f"data/request_traces.jsonl 共 {len(traces)} 条 trace")
    if traces:
        last = traces[-1]
        tu = last.get("token_usage") or {}
        info(f"最近一条：request_id={last.get('request_id')}  total_token={tu.get('total_tokens')}  "
             f"总时延={ (last.get('latency_ms') or {}).get('total') }ms  最终岗位数={len(last.get('final_ranked_jobs') or [])}")
        info(f"  token by_node={tu.get('by_node')}")
    try:
        rows = sqlite3.connect(events.DB_PATH).execute(
            "SELECT request_id, intent, n_final, total_tokens, total_latency_ms, user_feedback "
            "FROM request_traces ORDER BY created_at DESC LIMIT 5").fetchall()
        if rows:
            info("SQLite 摘要表最近 5 条（request_id / intent / n_final / tokens / latency_ms / feedback）：")
            for r in rows:
                print("     ", r)
    except Exception as e:
        info(f"（request_traces 摘要表暂为空或不可读：{e}）")
    if not traces:
        info("（暂无 trace —— 跑一次 `python scripts/verify_observability.py --with-api` 或任意 run_turn 即可产生）")


def section_e2e():
    hr("7. 真实 run_turn 全链路采集（--with-api：解析/检索/评分会调 API）")
    pdf = os.path.join(_PROJECT_ROOT, "Resumes", "resume_test_1.pdf")
    if not os.path.isfile(pdf):
        info("未找到 Resumes/resume_test_1.pdf，跳过")
        return
    from resume2job.agent.graph import build_graph, run_turn
    from resume2job.observability import events, replay
    app = build_graph()
    final = run_turn(app, "帮我找北京的大模型应用实习", session_id="obs_verify_e2e", pdf_path=pdf, top_k=3)
    rid = final.get("request_id")
    t = events.get_trace(rid)
    if not t:
        info("未采到 trace（异常）"); return
    tu = t["token_usage"]
    info(f"request_id={rid}")
    info(f"各节点时延(ms)={t['latency_ms']}")
    info(f"LLM 调用={tu['n_calls']}  总token={tu['total_tokens']}  by_node={tu['by_node']}  by_kind={tu['by_kind']}")
    info(f"检索候选 dense/bm25/rrf/rerank={len(t['retrieval']['dense_candidates'])}/"
         f"{len(t['retrieval']['bm25_candidates'])}/{len(t['retrieval']['rrf_candidates'])}/{len(t['retrieval']['rerank'])}")
    info(f"final_ranked={[(j['job_id'], j.get('rank_score')) for j in t['final_ranked_jobs']]}")
    ok("一次 run_turn 的全链路（规划/召回各通道/融合/精排/评分/token/时延）已落盘")
    # 若已有 ≥2 条 trace，演示 diff（不额外调 API）
    traces = events.load_traces()
    if len(traces) >= 2:
        d = replay.diff_traces(traces[-2], traces[-1], k=3)
        info("diff_traces（最近两条 trace 对比）：" + d.get("summary", ""))
        ok("前后对比（diff_traces）可用；线上回归用 replay.replay_request(request_id) 重跑历史请求再 diff")


def main():
    ap = argparse.ArgumentParser(description="Stage 2 observability 验收（默认离线、不改业务库）")
    ap.add_argument("--with-api", action="store_true", help="额外真实跑一轮 run_turn 采集（需联网 + DASHSCOPE_API_KEY）")
    args = ap.parse_args()
    section_imports()
    section_events_offline()
    section_thread_propagation()
    section_redaction()
    section_replay_diff()
    section_real_store()
    if args.with_api:
        section_e2e()
    else:
        hr("7. 真实 run_turn 采集（已跳过）")
        info("加 --with-api 真实跑一轮，看完整 trace（含 token by_node / 各阶段候选）")
    hr("验收完成")
    print("  结果在哪看：data/request_traces.jsonl（完整 trace）+ SQLite request_traces 表（可查询摘要 + feedback）")


if __name__ == "__main__":
    main()
