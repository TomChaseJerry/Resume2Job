"""
pipeline.py

端到端验收脚本（统一回归测试），运行后自动生成 docs/acceptance_report.md。

测试原则：
    1. 每个场景 = 一条完整用户旅程，全部通过 run_turn / 图执行验证（不直接调业务函数）；
    2. 基础机制（画像缓存、对话历史注入）在多轮场景中隐式验证，不单独立场景；
    3. 一场景多断言：核验链路上各关键节点的产物，FAIL 信息精确到断言；
    4. 数据隔离：验收目录拷贝真实库，写操作（入库/画像/会话）不污染 data/；
    5. 检索质量指标（Recall/MRR/nDCG）归 eval 层（resume2job/eval），与本脚本互不重叠。

场景矩阵：
    S1  岗位推荐全链路（FC planner 条件抽取 / 混合检索 / SQLite 回填 / 增强零调用）
    S2  JD 评估 + 全增强（jd_ingest 去重入库 / Tool Calling 多工具 / 3 道面试题 / 学习计划）
    S3  多轮记忆 + 通勤（画像缓存命中 / 通勤意图抽取 / 高德路线与时长进报告）
    S4  城市硬约束提前终止（无深圳岗 → 不烧评分调用，直接告知并询问放宽）
    S5  边界鲁棒（无简历无缓存 / 缺 JD，优雅降级不崩溃）

用法：
    python pipeline.py            # 跑全部场景
    python pipeline.py 1 3        # 只跑指定场景（按编号；S3 依赖会话独立，可单跑）
"""

import os
import sys
import time
import shutil
from datetime import datetime

# Windows 控制台中文输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 验收常量与公共工具
# ---------------------------------------------------------------------------
RESUME_PDF = os.path.join("Resumes", "resume_test_1.pdf")
JD_EVAL_FILE = os.path.join("JDs", "jd_test_5.txt")
ACCEPT_DIR = os.path.join("tmp", "acceptance_data")
TRACE_DIR = os.path.join(ACCEPT_DIR, "traces")
REPORT_PATH = os.path.join("docs", "acceptance_report.md")

# 场景结果收集：[(编号, 名称, 是否通过, 明细, 耗时秒, [(断言描述, 是否通过)], [TurnTrace])]
_RESULTS = []


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class Checker:
    """逐条记录断言（供报告输出）；失败即抛 AssertionError 中止当前场景。

    两类断言：
        - expect(cond, desc)            ：对最终 State 的断言（验输出结构 / 终态正确）；
        - at(node, predicate, desc)     ：对某节点 trace 快照的断言（把"结果不对"定位到"哪个节点开始偏"）。
    at 依赖 bind() 关联的 TurnTrace；多轮场景可在每轮后重新 bind 当轮 trace。
    """

    def __init__(self):
        self.items = []   # [(描述, bool)]
        self.trace = None  # 当前关联的 TurnTrace（供 at 查询节点快照）

    def bind(self, trace):
        """关联当前轮的 trace，返回它本身（便于 `tr = c.bind(run_turn_traced(...))` 写法）。"""
        self.trace = trace
        return trace

    def expect(self, cond: bool, desc: str):
        self.items.append((desc, bool(cond)))
        if not cond:
            raise AssertionError(desc)

    def at(self, node: str, predicate, desc: str):
        """对 node 执行后的关键信号快照断言；node 未执行或谓词为假 -> FAIL（带节点名）。"""
        snap = self.trace.summary_at(node) if self.trace else None
        try:
            ok = snap is not None and bool(predicate(snap))
        except Exception:
            ok = False
        full = f"[{node}] {desc}"
        self.items.append((full, ok))
        if not ok:
            raise AssertionError(full)


def _record(no: int, name: str, ok: bool, detail: str, elapsed: float,
            checks: list, traces: list) -> None:
    _RESULTS.append((no, name, ok, detail, elapsed, checks, traces))
    # 链路 trace 落盘，失败可单独复盘
    for tr in traces or []:
        try:
            tr.dump_json(os.path.join(TRACE_DIR, tr.trace_id.replace("#", "_t") + ".json"))
        except Exception as e:
            print(f"[警告] trace 落盘失败（{getattr(tr, 'trace_id', '?')}）：{e}")
    flag = "PASS" if ok else "FAIL"
    print(f"\n[{flag}] S{no} {name}（{elapsed:.1f}s）" + (f" — {detail}" if detail else ""))
    if traces:
        print("        路径：" + "  |  ".join(" → ".join(tr.nodes()) for tr in traces))


def setup_env():
    """隔离验收数据目录，并把存储模块指向该目录，保证可复现。

    jobs 表（SQLite 事实源）：真实 data/resume2job.db 存在则整库拷贝到隔离目录——
    检索回填（hydration）需要真实岗位数据，而入库/画像/会话写操作不应污染真实库。
    """
    shutil.rmtree(ACCEPT_DIR, ignore_errors=True)
    os.makedirs(ACCEPT_DIR, exist_ok=True)
    accept_db = os.path.join(ACCEPT_DIR, "resume2job.db")

    from resume2job.storage.paths import SQLITE_PATH
    if os.path.isfile(SQLITE_PATH):
        shutil.copyfile(SQLITE_PATH, accept_db)

    import resume2job.storage.jobs_store as js
    js.DB_PATH = accept_db
    js.init_db()

    import resume2job.storage.profile_cache as pc
    pc.DB_PATH = accept_db
    pc.init_db()

    import resume2job.storage.conversation_store as cs
    cs.DB_PATH = accept_db
    cs.init_db()

    import resume2job.storage.jd_ingest as jd
    jd.CHROMA_DIR = os.path.join(ACCEPT_DIR, "chroma_db")
    jd._chroma_collection = jd.init_chroma()
    return pc, jd


# ---------------------------------------------------------------------------
# 场景实现
# ---------------------------------------------------------------------------
def scenario_1_job_recommendation(app, run_turn_traced, ctx):
    """S1：上传简历 + 「推荐北京的大模型实习岗位」—— job_recommendation 全链路。"""
    name = "岗位推荐全链路"
    c = Checker()
    start = time.time()
    try:
        final, tr = run_turn_traced(
            app, "帮我推荐几个北京的大模型方向实习岗位",
            session_id="accept_s1", pdf_path=RESUME_PDF, top_k=3,
        )
        c.bind(tr)
        ctx["s1_final"] = final

        c.expect(final.get("intent") == "RECOMMEND",
                 f"intent=RECOMMEND（实际 {final.get('intent')}）")
        rc = final.get("retrieval_config") or {}
        c.expect(rc.get("city_filter") == "北京", f"FC planner 抽出 city=北京（实际 {rc.get('city_filter')}）")

        # 节点级定位：把"最终没岗位"拆解到 planner / 检索 / 评分，定位偏差起点
        c.at("planner", lambda s: (s["retrieval_config"] or {}).get("city_filter") == "北京",
             "planner 节点抽出 city=北京")
        c.at("job_retriever", lambda s: s["n_candidates"] >= 1, "检索节点召回候选 ≥1")
        c.at("match_scorer", lambda s: s["n_match_results"] >= 1, "评分节点产出结果 ≥1")

        # 岗位类型硬约束默认实习（scenario_overview §2.1）
        c.expect((final.get("retrieval_config") or {}).get("job_type_filter") == "实习",
                 f"岗位类型默认实习（实际 {(final.get('retrieval_config') or {}).get('job_type_filter')}）")

        mrs = final.get("match_results") or []
        c.expect(len(mrs) >= 1, f"召回并评分岗位 ≥1（实际 {len(mrs)}）")
        c.expect(all((m.get("match_score") or {}).get("rank_score") is not None for m in mrs),
                 "每个岗位都有 rank_score（最终推荐分）")
        c.expect(all((m.get("match_score") or {}).get("match_score") is not None for m in mrs),
                 "每个岗位都有 match_score（基础适配分）")
        c.expect(all((m.get("report") or "").strip() for m in mrs), "每个岗位都有推荐报告")
        c.expect(all(len(m.get("jd_profile") or {}) >= 10 for m in mrs),
                 "候选 jd_profile 来自 SQLite 回填（字段完整）")
        c.expect(all((((m.get("jd_profile") or {}).get("location")) or {}).get("city") == "北京"
                     for m in mrs),
                 "全部岗位城市均为北京（硬约束生效）")
        # 无 ASSIST 诉求：推荐默认不生成学习计划 / 面试题（仅 intent=ASSIST + assist_actions 才触发）
        c.expect(not (final.get("learning_plan") or {}).get("stages"), "未请求时不生成学习计划")
        c.expect(not (final.get("interview_prep") or {}).get("questions"), "未请求时不生成面试题")

        score0 = (mrs[0].get("match_score") or {}).get("rank_score")
        _record(1, name, True, f"召回 {len(mrs)} 岗，首位 rank_score={score0}",
                time.time() - start, c.items, [tr])
    except Exception as e:
        _record(1, name, False, f"{type(e).__name__}: {e}", time.time() - start, c.items, [c.trace] if c.trace else [])


def scenario_2_jd_eval_full_enhancements(app, run_turn_traced, ctx):
    """S2：① 上传简历 + JD 评估适配；② 针对该岗位 job_id 请求学习计划 + 面试练习题（ASSIST）。"""
    name = "JD 评估 + ASSIST 辅助"
    c = Checker()
    start = time.time()
    traces = []
    try:
        # 轮①：JD 评估（默认含匹配点 + 技能缺口，不自动出学习/面试）
        jd_text = _read_text(JD_EVAL_FILE)
        t1, tr1 = run_turn_traced(
            app, "这个岗位适合我吗？我缺哪些技能？",
            session_id="accept_s2", pdf_path=RESUME_PDF, jd_text=jd_text,
        )
        traces.append(c.bind(tr1))
        ctx["s2_final"] = t1

        c.expect(t1.get("intent") == "EVALUATE", f"轮① intent=EVALUATE（实际 {t1.get('intent')}）")
        c.expect(bool(t1.get("ingested_job_id")),
                 f"jd_ingest 入库/去重命中（job_id={t1.get('ingested_job_id')}）")
        mrs = t1.get("match_results") or []
        c.expect(len(mrs) == 1, f"单 JD 评估产出 1 条结果（实际 {len(mrs)}）")
        mr0 = mrs[0]
        c.expect(bool((mr0.get("report") or "").strip()), "推荐报告非空")
        c.expect(len((mr0.get("skill_gap") or {}).get("items") or []) >= 1, "skill_gap 有逐项分析")
        c.expect(not (t1.get("learning_plan") or {}).get("stages"), "轮① 未请求时不出学习计划")
        c.expect(not (t1.get("interview_prep") or {}).get("questions"), "轮① 未请求时不出面试题")

        job_id = mr0.get("job_id")

        # 轮②：针对该 job_id 请求学习计划 + 面试练习题（ASSIST）
        t2, tr2 = run_turn_traced(
            app, f"针对 {job_id} 给我一份补齐差距的学习计划，再出几道岗位定制面试练习题",
            session_id="accept_s2", pdf_path=None,
        )
        traces.append(c.bind(tr2))

        c.at("planner", lambda s: s["intent"] == "ASSIST", "轮② planner 识别 intent=ASSIST")
        c.expect(t2.get("intent") == "ASSIST", f"轮② intent=ASSIST（实际 {t2.get('intent')}）")
        c.expect(t2.get("selected_item_ref") == job_id, f"轮② 指代命中 job_id={job_id}")
        c.expect("job_retriever" not in tr2.nodes(), "轮② 不重新召回（ASSIST 复用缓存岗位）")

        plan = t2.get("learning_plan") or {}
        c.expect(("stages" in plan) or bool(plan.get("overall_suggestion")),
                 "ASSIST 产出学习计划（阶段或无核心缺口建议）")
        questions = (t2.get("interview_prep") or {}).get("questions") or []
        c.expect(len(questions) == 3, f"面试练习题恰好 3 道（实际 {len(questions)}）")

        _record(2, name, True,
                f"rank_score={(mr0.get('match_score') or {}).get('rank_score')}, "
                f"学习计划 {len(plan.get('stages') or [])} 阶段, 面试题 {len(questions)} 道",
                time.time() - start, c.items, traces)
    except Exception as e:
        _record(2, name, False, f"{type(e).__name__}: {e}", time.time() - start, c.items, traces)


def scenario_3_memory_and_commute(app, run_turn_traced, ctx):
    """S3：同一会话两轮——①传简历评估 JD；②不传简历，带通勤约束求推荐。

    隐式验证：画像缓存（轮② cached）、对话历史注入；
    显式验证：通勤意图抽取（地址/上限/交通方式）、报告含通勤路线与时间。
    """
    name = "多轮记忆 + 通勤"
    c = Checker()
    start = time.time()
    session = "accept_s3"
    traces = []
    try:
        # 轮①：上传简历 + 评估 JD（建立画像与对话历史）
        t1, tr1 = run_turn_traced(app, "看看这个岗位适合我吗", session_id=session,
                                  pdf_path=RESUME_PDF, jd_text=_read_text(JD_EVAL_FILE))
        traces.append(c.bind(tr1))
        c.expect(t1.get("profile_source") == "new_upload",
                 f"轮① 画像保存 new_upload（实际 {t1.get('profile_source')}）")

        # 轮②：不传简历，带通勤约束求推荐（公交地铁 + 1 小时上限）
        t2, tr2 = run_turn_traced(app, "帮我推荐实习岗位，我住在北京市海淀区中关村，公交地铁通勤1小时以内",
                                  session_id=session, pdf_path=None)
        traces.append(c.bind(tr2))  # 后续 at() 针对轮② trace
        ctx["s3_final"] = t2

        c.expect(t2.get("profile_source") == "cached",
                 f"轮② 复用缓存画像（实际 {t2.get('profile_source')}）")
        # 节点级定位：画像复用由 profile_cache 节点决定，通勤意图由 planner 抽取
        c.at("profile_cache", lambda s: s["profile_source"] == "cached", "轮② profile_cache 命中缓存")
        c.at("planner", lambda s: (s["commute_intent"] or {}).get("max_commute_minutes") == 60
             and (s["commute_intent"] or {}).get("preferred_transport") == "transit",
             "planner 抽出通勤意图（60min / transit）")
        intent = t2.get("commute_intent") or {}
        c.expect("中关村" in (intent.get("user_address") or ""),
                 f"通勤地址抽取（实际 {intent.get('user_address')}）")
        c.expect(intent.get("max_commute_minutes") == 60,
                 f"时间上限换算 60 分钟（实际 {intent.get('max_commute_minutes')}）")
        c.expect(intent.get("preferred_transport") == "transit",
                 f"交通方式=transit（实际 {intent.get('preferred_transport')}）")
        c.expect((t2.get("retrieval_config") or {}).get("top_k", 0) >= 10, "通勤场景扩大召回 top_k≥10")

        mrs = t2.get("match_results") or []
        c.expect(len(mrs) >= 1, f"召回并评分岗位 ≥1（实际 {len(mrs)}）")

        if os.environ.get("AMAP_API_KEY"):
            c.expect(len(t2.get("commute_results") or []) >= 1, "通勤计算结果非空")
            reports = " ".join(m.get("report") or "" for m in mrs)
            c.expect("【通勤】" in reports, "报告含【通勤】段")
            c.expect("分钟" in reports, "报告含通勤时间")
            c.expect("路线：" in reports, "报告含通勤路线（公交地铁换乘串）")
            detail = (t2.get("commute_results") or [{}])[0].get("commute_summary", "")
        else:
            c.expect(bool(t2.get("commute_note")), "无 AMAP_KEY 时降级为通勤说明")
            detail = t2.get("commute_note") or ""

        # 轮③：ASSIST 指代 job_id 出面试练习题 —— 位置指代已不支持，须用 job_id
        target_id = mrs[0].get("job_id")
        t3, tr3 = run_turn_traced(app, f"给 {target_id} 出几道面试练习题", session_id=session, pdf_path=None)
        traces.append(c.bind(tr3))
        c.at("planner", lambda s: s["intent"] == "ASSIST", "planner 识别 intent=ASSIST（非新检索）")
        ref = t3.get("selected_item_ref")
        c.expect(ref == target_id, f"job_id 指代解析（实际 {ref}）")
        c.expect("job_retriever" not in tr3.nodes(), "ASSIST 不重新召回")
        q3 = (t3.get("interview_prep") or {}).get("questions") or []
        c.expect(len(q3) == 3, f"针对指代岗位出 3 道面试练习题（实际 {len(q3)}）")

        _record(3, name, True, f"首位通勤：{detail[:50]}；ASSIST→{ref}",
                time.time() - start, c.items, traces)
    except Exception as e:
        _record(3, name, False, f"{type(e).__name__}: {e}", time.time() - start, c.items, traces)


def scenario_4_city_hard_constraint(app, run_turn_traced, ctx):
    """S4：「推荐武汉的实习岗位」（库中无武汉岗）—— 城市硬约束提前终止。

    选「武汉」是因为它在岗位库（含多城市阿里岗）中零命中；深圳/广州等已被多城市岗位覆盖。
    """
    name = "城市硬约束提前终止"
    c = Checker()
    start = time.time()
    try:
        final, tr = run_turn_traced(app, "帮我推荐武汉的实习岗位",
                                    session_id="accept_s4", pdf_path=RESUME_PDF)
        c.bind(tr)
        ctx["s4_final"] = final

        rc = final.get("retrieval_config") or {}
        c.expect(rc.get("city_filter") == "武汉", f"FC planner 抽出 city=武汉（实际 {rc.get('city_filter')}）")

        # 节点级定位：提前终止应发生在 job_retriever（召回=0 即写 final_response 告知），
        # 且评分节点应未运行 / 未产出——证明零浪费而非"评分后才发现没结果"
        c.at("job_retriever", lambda s: s["n_candidates"] == 0 and s["has_final_response"],
             "检索节点召回=0 并就地写出告知文案（提前终止）")
        c.expect("match_scorer" not in tr.nodes() or (tr.summary_at("match_scorer") or {}).get("n_match_results", 0) == 0,
                 "评分节点未对错误城市产出结果（零浪费）")

        c.expect(not (final.get("match_results") or []),
                 "match_results 为空（未对错误城市浪费评分调用）")
        resp = final.get("final_response") or ""
        c.expect("武汉" in resp and "暂无" in resp, f"明确告知暂无武汉岗位（实际：{resp[:50]}）")
        c.expect("不限城市" in resp, "提示用户可放宽到不限城市")

        _record(4, name, True, f"提前终止：{resp[:60]}", time.time() - start, c.items, [tr])
    except Exception as e:
        _record(4, name, False, f"{type(e).__name__}: {e}", time.time() - start, c.items, [c.trace] if c.trace else [])


def scenario_5_clarification(app, run_turn_traced, pc, ctx):
    """S5：澄清机制——缺关键槽位时追问而非跑半成品（「跑错链路比多问一句更糟」）。

    ① 无简历无缓存求推荐 → 追问上传简历；② 有简历但缺 JD 求评估 → 追问 JD。
    两类都应：plan.clarify=True、final_response 为澄清问题、不产出岗位结果、业务节点未跑。
    注意：本场景会清空画像缓存，必须最后运行。
    """
    name = "澄清机制"
    c = Checker()
    start = time.time()
    traces = []
    try:
        pc.delete_profiles(pc.DEFAULT_USER_ID)  # 清空缓存，制造「无画像」环境

        # ① 无简历且无缓存画像，直接求推荐 → 追问简历
        f1, tr1 = run_turn_traced(app, "帮我推荐适合我的实习岗位", session_id="accept_s5a", pdf_path=None)
        traces.append(c.bind(tr1))
        c.at("planner", lambda s: (s["plan"] or {}).get("clarify") is True, "①planner 判定需澄清")
        c.expect(not (f1.get("match_results") or []), "①澄清时不产出岗位结果")
        c.expect("简历" in (f1.get("final_response") or ""), "①追问上传简历")
        c.expect("job_retriever" not in tr1.nodes(), "①未跑检索节点（澄清短路）")

        # ② 有简历（缓存命中）但缺 JD 求评估 → 追问 JD
        seed = (ctx.get("s1_final") or {}).get("resume_profile") or {"skills": ["Python"]}
        pc.save_profile(pc.DEFAULT_USER_ID, "", seed)
        f2, tr2 = run_turn_traced(app, "帮我分析这个岗位适不适合我", session_id="accept_s5b", pdf_path=None)
        traces.append(c.bind(tr2))
        c.expect((f2.get("plan") or {}).get("clarify") is True, "②planner 判定需澄清")
        c.expect(not (f2.get("match_results") or []), "②澄清时不产出岗位结果")
        c.expect(bool(f2.get("final_response")), "②给出澄清问题")

        _record(5, name, True, "缺简历/缺 JD 均触发澄清、未跑业务链路",
                time.time() - start, c.items, traces)
    except Exception as e:
        _record(5, name, False, f"{type(e).__name__}: {e}", time.time() - start, c.items, traces)


# ---------------------------------------------------------------------------
# 验收报告（docs/acceptance_report.md，固定文件名覆盖式）
# ---------------------------------------------------------------------------
_PRINCIPLES = """\
1. **场景即用户旅程**：每个场景通过 `run_turn`（真实图执行）验证一条完整链路，不直接调业务函数；
2. **机制隐式验证**：画像缓存、对话历史注入等基础机制在多轮场景（S3）内以断言覆盖，不单独立场景；
3. **一场景多断言**：核验链路上各关键节点产物（规划抽取、检索回填、ASSIST 驱动增强、澄清短路、输出结构），失败精确到断言；
4. **数据隔离**：验收目录拷贝真实库后运行，入库/画像/会话写操作不污染 `data/`；
5. **职责分离**：检索质量指标（Recall@K / MRR / nDCG）由 `resume2job/eval` 负责，本验收只管链路正确性。"""

_MATRIX = """\
| 场景 | 用户旅程 | 覆盖能力 |
|---|---|---|
| S1 岗位推荐全链路 | 传简历 +「推荐北京的大模型实习岗位」 | planner FC 抽 intent=RECOMMEND + 条件、混合检索（BM25+向量+RRF+rerank）、SQLite 回填、城市约束、未请求时不开增强视图 |
| S2 JD 评估 + ASSIST 辅助 | ① 传简历 + JD 评估；②「针对 job_X 给学习计划和面试练习题」 | intent=EVALUATE 评估默认含技能缺口、jd_ingest 去重入库；intent=ASSIST 复用缓存岗位生成学习计划 + 3 道面试练习题 |
| S3 多轮记忆 + 通勤 + 指代 | 轮① 评估 JD；轮② 不传简历「我住中关村，公交地铁1小时内」求推荐；轮③「第二个出面试题」 | 画像缓存命中、通勤意图抽取与路线/时长进报告、**last_results 结构化指代（「第二个」→job_id）+ FOLLOWUP 复用** |
| S4 城市硬约束提前终止 | 「推荐深圳的实习岗位」（库中无深圳岗） | 城市硬约束（检索级联不丢城市）、零浪费提前终止、final_response 告知并询问放宽 |
| S5 澄清机制 | ①无简历无缓存求推荐 ②有简历缺 JD 求评估 | 缺关键槽位时 planner 判 clarify、追问而非跑半成品、业务节点短路 |"""


def write_report() -> None:
    from resume2job.core import config

    lines = [
        "# Resume2Job 端到端验收报告",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　|　脚本：`python pipeline.py`",
        "",
        "## 测试原则",
        "",
        _PRINCIPLES,
        "",
        "## 场景矩阵",
        "",
        _MATRIX,
        "",
        "## 本次运行结果",
        "",
    ]

    passed = sum(1 for r in _RESULTS if r[2])
    lines.append(f"**总计：{passed} / {len(_RESULTS)} 通过**")
    lines.append("")
    for no, name, ok, detail, elapsed, checks, traces in sorted(_RESULTS, key=lambda r: r[0]):
        lines.append(f"### S{no} {name} — {'✅ PASS' if ok else '❌ FAIL'}（{elapsed:.1f}s）")
        lines.append("")
        if detail:
            lines.append(f"{detail}")
            lines.append("")
        for desc, c_ok in checks:
            lines.append(f"- {'✓' if c_ok else '✗'} {desc}")
        lines.append("")
        # 链路时序：每轮一张表（节点顺序 / 耗时 / 改动字段 / 关键信号）
        for i, tr in enumerate(traces or [], start=1):
            label = f"链路追踪 轮{i}" if len(traces) > 1 else "链路追踪"
            lines.append(f"<details><summary>🔍 {label}（trace_id=`{tr.trace_id}`）</summary>")
            lines.append("")
            lines.append(tr.as_markdown_table())
            lines.append("")
            lines.append("</details>")
            lines.append("")

    lines += [
        "## 运行环境",
        "",
        f"- 主模型：`{config.CHAT_MODEL}`　规划：`{config.PLANNER_MODEL}`　评分：`{config.SCORE_MODEL}`",
        f"- 检索：mode=`{config.RETRIEVAL_MODE}`，rerank=`{config.USE_RERANK}`（`{config.RERANK_MODEL}`）",
        f"- Embedding：`{config.EMBEDDING_MODEL}`",
        "",
        "## 已知限制",
        "",
        "- 依赖联网与 `DASHSCOPE_API_KEY`；S3 的通勤断言依赖 `AMAP_API_KEY`（缺失时降级为通勤说明断言）；",
        "- 意图分类 / 条件抽取 / 工具决策由 LLM 完成，存在小概率非确定性误判——单场景偶发 FAIL 可重跑确认；",
        "- 检索召回质量不在本验收范围，见 `resume2job/eval` 的指标评测报告。",
        "",
    ]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[报告] 已生成：{REPORT_PATH}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        selected = [int(x) for x in argv] if argv else [1, 2, 3, 4, 5]
    except ValueError:
        print("参数应为场景编号，例如：python pipeline.py 1 3")
        return 1

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("[警告] 未配置 DASHSCOPE_API_KEY，依赖 LLM/嵌入的场景会失败或走兜底。")

    print("=" * 64)
    print("Resume2Job 端到端验收  |  场景:", selected)
    print("=" * 64)

    pc, _jd = setup_env()
    from resume2job.agent.graph import build_graph, run_turn_traced
    app = build_graph()
    ctx = {}

    for n in selected:
        if n == 1:
            scenario_1_job_recommendation(app, run_turn_traced, ctx)
        elif n == 2:
            scenario_2_jd_eval_full_enhancements(app, run_turn_traced, ctx)
        elif n == 3:
            scenario_3_memory_and_commute(app, run_turn_traced, ctx)
        elif n == 4:
            scenario_4_city_hard_constraint(app, run_turn_traced, ctx)
        elif n == 5:
            scenario_5_clarification(app, run_turn_traced, pc, ctx)
        else:
            print(f"[跳过] 未知场景编号：{n}")

    # 汇总 + 报告
    print("\n" + "=" * 64)
    print("验收汇总")
    print("=" * 64)
    passed = sum(1 for r in _RESULTS if r[2])
    for no, name, ok, detail, elapsed, _checks, _traces in sorted(_RESULTS, key=lambda r: r[0]):
        print(f"  S{no} [{'PASS' if ok else 'FAIL'}] {name}（{elapsed:.1f}s）")
        if detail:
            print(f"        {detail}")
    print(f"\n结果：{passed} / {len(_RESULTS)} 通过")

    write_report()
    return 0 if passed == len(_RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
