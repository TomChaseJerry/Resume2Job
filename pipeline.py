"""
pipeline.py

端到端验收脚本（进前端前的统一回归测试）。

覆盖 5 个核心场景：
    S1  上传简历 + 直接 JD 评估（jd_evaluation 全流程，含 FC planner 与 jd_ingest 入库）
    S2  上传简历 + 岗位推荐（job_recommendation 全流程，混合检索）
    S3  JD 批量入库 + 检索（jd_ingest_node 去重 + 向量检索）
    S4  画像缓存命中（先上传保存画像，再不带简历命中缓存）
    S5  Skill Gap / Learning Plan(单次调用) / Interview Questions(3 题) 输出

设计：
    - 每个场景独立 try/except，互不影响，最后汇总 PASS/FAIL；
    - 验收数据隔离到 tmp/acceptance_data（gitignore 内），不污染真实 data/，可重复运行；
    - 需要联网与 DASHSCOPE_API_KEY（部分嵌入/评分能力依赖阿里云百炼）。

用法：
    python pipeline.py            # 跑全部场景
    python pipeline.py 3 4        # 只跑指定场景（按编号）
"""

import os
import sys
import shutil

# Windows 控制台中文输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from resume2job.agent.state import get_initial_state

# ---------------------------------------------------------------------------
# 验收常量与公共工具
# ---------------------------------------------------------------------------
RESUME_PDF = os.path.join("Resumes", "resume_test_1.pdf")
JD_FILES = [os.path.join("JDs", f"jd_test_{i}.txt") for i in range(1, 7)]
ACCEPT_DIR = os.path.join("tmp", "acceptance_data")

# 场景结果收集：[(编号, 名称, 是否通过, 明细)]
_RESULTS = []


def _read_text(path: str) -> str:
    """UTF-8 读取文本（JD 文件在中文 Windows 上必须显式指定编码）。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _record(no: int, name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((no, name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"\n[{flag}] S{no} {name}" + (f" — {detail}" if detail else ""))


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def setup_env():
    """隔离验收数据目录，并把缓存/入库模块指向该目录，保证可复现。"""
    shutil.rmtree(ACCEPT_DIR, ignore_errors=True)
    os.makedirs(ACCEPT_DIR, exist_ok=True)

    import resume2job.storage.profile_cache as pc
    pc.DB_PATH = os.path.join(ACCEPT_DIR, "resume2job.db")
    pc.init_db()

    import resume2job.storage.jd_ingest as jd
    jd.DB_PATH = os.path.join(ACCEPT_DIR, "resume2job.db")
    jd.CHROMA_DIR = os.path.join(ACCEPT_DIR, "chroma_db")
    jd.init_db()
    jd._chroma_collection = jd.init_chroma()
    return pc, jd


# ---------------------------------------------------------------------------
# 场景实现
# ---------------------------------------------------------------------------
def scenario_1_jd_evaluation(app, ctx):
    """S1：上传简历 + 直接粘贴 JD -> jd_evaluation 全流程。"""
    name = "上传简历 + 直接 JD 评估"
    try:
        jd_text = _read_text(JD_FILES[0])
        state = get_initial_state(
            user_query="请帮我评估这份简历和该岗位的匹配度，适合投递吗？",
            pdf_path=RESUME_PDF,
            jd_text=jd_text,
        )
        final = app.invoke(state)
        ctx["s1_final"] = final  # 供 S5 复用

        task_type = final.get("task_type")
        match_results = final.get("match_results") or []
        _assert(task_type == "jd_evaluation", f"task_type 应为 jd_evaluation，实际 {task_type}")
        _assert(len(match_results) >= 1, "match_results 为空")
        mr0 = match_results[0]
        _assert(bool(mr0.get("match_score")), "match_results[0] 缺少 match_score")
        _assert(isinstance(mr0.get("skill_gap"), dict) and mr0["skill_gap"], "match_results[0] 缺少 skill_gap")
        _assert(isinstance(final.get("learning_plan"), dict), "learning_plan 缺失")
        score = (mr0.get("match_score") or {}).get("final_score")
        _record(1, name, True, f"task_type={task_type}, 岗位数={len(match_results)}, final_score={score}")
    except Exception as e:
        _record(1, name, False, f"{type(e).__name__}: {e}")


def scenario_2_job_recommendation(app, ctx):
    """S2：上传简历 + 岗位推荐 -> job_recommendation 全流程（统一向量库 data/chroma_db 检索）。"""
    name = "上传简历 + 岗位推荐"
    try:
        state = get_initial_state(
            user_query="帮我推荐几个北京的大模型/AI 算法实习岗位",
            pdf_path=RESUME_PDF,
            city_filter="北京",
            top_k=3,
        )
        final = app.invoke(state)
        ctx["s2_final"] = final

        task_type = final.get("task_type")
        match_results = final.get("match_results") or []
        _assert(task_type == "job_recommendation", f"task_type 应为 job_recommendation，实际 {task_type}")
        _assert(len(match_results) >= 1, "match_results 为空（data/chroma_db 检索无结果？）")
        _record(2, name, True, f"task_type={task_type}, 召回并评分岗位数={len(match_results)}")
    except Exception as e:
        _record(2, name, False, f"{type(e).__name__}: {e}")


def scenario_3_jd_ingest_and_search(app, jd, ctx):
    """S3：JD 批量入库（含去重）+ 向量检索。"""
    name = "JD 批量入库 + 检索"
    try:
        from resume2job.parsing.jd_parser import parse_jd

        ingested, duplicates = [], 0
        first_job_id = None
        first_parsed = None
        for idx, path in enumerate(JD_FILES):
            jd_text = _read_text(path)
            parsed = parse_jd(jd_text)
            out = jd.jd_ingest_node({"jd_text": jd_text, "jd_profiles": [parsed], "errors": []})
            _assert(bool(out.get("ingested_job_id")), f"{path} 入库未返回 job_id")
            if out.get("jd_is_duplicate"):
                duplicates += 1
            else:
                ingested.append(out["ingested_job_id"])
            if idx == 0:
                first_job_id = out["ingested_job_id"]
                first_parsed = parsed

        # 去重验证：重复提交第一条 JD 应被判为重复并复用同一 job_id
        dup_out = jd.jd_ingest_node({
            "jd_text": _read_text(JD_FILES[0]),
            "jd_profiles": [first_parsed],
            "errors": [],
        })
        _assert(dup_out.get("jd_is_duplicate") is True, "重复提交未被判为重复")
        _assert(dup_out.get("ingested_job_id") == first_job_id, "重复 JD 未复用原 job_id")

        # 检索验证：在 jobs collection 中按语义检索，应有命中
        from resume2job.core.llm import get_embedding
        vec = get_embedding("大模型 Agent 算法工程师")
        res = jd._chroma_collection.query(query_embeddings=[vec], n_results=3)
        hits = (res.get("ids") or [[]])[0]
        _assert(len(hits) >= 1, "向量检索无命中")

        # 入库 metadata 应携带 jd_profile_json（用户粘贴入库的 JD 是检索一等公民）
        sample = jd._chroma_collection.get(ids=[first_job_id], include=["metadatas"])
        meta0 = (sample.get("metadatas") or [{}])[0] or {}
        _assert(bool(meta0.get("jd_profile_json")), "入库 metadata 缺少 jd_profile_json")

        _record(3, name, True,
                f"新入库 {len(ingested)} 条，去重命中 {duplicates}+1 条，检索 top{len(hits)} 命中")
    except Exception as e:
        _record(3, name, False, f"{type(e).__name__}: {e}")


def scenario_4_profile_cache_hit(app, pc, ctx):
    """S4：先保存画像（首次上传），再不带简历跑一次（命中缓存并贯通下游）。

    为聚焦「缓存机制」本身、避免重复解析简历带来的外部波动，
    优先复用 S1/S2 已解析好的画像直接 seed 缓存；只有在缺少现成画像时才回退到带简历跑图。
    """
    name = "画像缓存命中"
    try:
        # 用独立的缓存库，保证第一次为空、第二次命中
        cache_db = os.path.join(ACCEPT_DIR, "cache_only.db")
        if os.path.exists(cache_db):
            os.remove(cache_db)
        pc.DB_PATH = cache_db
        pc.init_db()

        # 第一步：写入一份画像（优先复用已解析结果，避免再次解析简历）
        seed_profile = None
        for key in ("s1_final", "s2_final"):
            f = ctx.get(key)
            if f and f.get("resume_profile"):
                seed_profile = f["resume_profile"]
                break

        if seed_profile is not None:
            saved_id = pc.save_profile(pc.DEFAULT_USER_ID, "", seed_profile)
            save_mode = "seed(复用已解析画像)"
        else:
            # 回退：带简历跑一次图，由 profile_cache 保存新画像
            f1 = app.invoke(get_initial_state(
                user_query="帮我推荐北京的大模型实习岗位",
                pdf_path=RESUME_PDF, city_filter="北京", top_k=2,
            ))
            _assert(f1.get("profile_source") == "new_upload",
                    f"首次应为 new_upload，实际 {f1.get('profile_source')}")
            saved_id = f1.get("profile_id")
            save_mode = "graph(new_upload)"
        _assert(bool(saved_id), "未生成 profile_id")

        # 第二步：不带简历跑图 -> resume_parser 因无 resume_profile 进入兜底，
        #          profile_cache 命中缓存并回填 resume_profile，贯通下游检索/评分
        f2 = app.invoke(get_initial_state(
            user_query="再帮我推荐北京的大模型实习岗位",
            pdf_path=None, city_filter="北京", top_k=2,
        ))
        _assert(f2.get("profile_source") == "cached", f"二次应为 cached，实际 {f2.get('profile_source')}")
        _assert(bool(f2.get("resume_profile")), "命中缓存后 resume_profile 仍为空")
        _assert(f2.get("profile_id") == saved_id, "命中的 profile_id 与保存的不一致")
        _assert(len(f2.get("match_results") or []) >= 1, "用缓存画像未跑出 match_results")

        _record(4, name, True,
                f"保存={save_mode}({saved_id}) -> 二次 cached，缓存画像贯通下游")
    except Exception as e:
        _record(4, name, False, f"{type(e).__name__}: {e}")


def scenario_5_gap_plan_interview(app, ctx):
    """S5：Skill Gap / Learning Plan（单次调用）/ Interview Questions（3 题）输出。"""
    name = "Skill Gap / Learning Plan / Interview Questions"
    try:
        # 复用 S1 的结果；若未跑 S1 则先补跑
        final = ctx.get("s1_final")
        if not final:
            scenario_1_jd_evaluation(app, ctx)
            final = ctx.get("s1_final")
        _assert(bool(final), "无可用的 S1 结果")

        mr0 = (final.get("match_results") or [None])[0]
        _assert(bool(mr0), "S1 无 match_results")

        # Skill Gap
        skill_gap = mr0.get("skill_gap") or {}
        _assert(isinstance(skill_gap, dict) and ("items" in skill_gap or "overall_risk" in skill_gap),
                "skill_gap 结构异常")

        # Learning Plan（单次 LLM 调用生成全部阶段）
        from resume2job.generation.learning_plan import build_learning_plan
        plan = build_learning_plan(mr0, "给我一个 30 天每天 2 小时的学习计划")
        _assert(isinstance(plan, dict) and "overall_suggestion" in plan, "learning_plan 结构异常")
        n_stages = len(plan.get("stages") or [])
        _assert(plan.get("error") is None, f"learning_plan error: {plan.get('error')}")

        # Interview Questions（固定 3 题）
        from resume2job.generation.interview import generate_interview_questions, MAX_QUESTIONS
        q = generate_interview_questions(
            resume_profile=final.get("resume_profile") or {},
            jd_profile=mr0.get("jd_profile") or {},
            match_result=mr0.get("match_score") or {},
            skill_gap=skill_gap,
        )
        questions = q.get("questions") or []
        _assert(1 <= len(questions) <= MAX_QUESTIONS, f"面试题数量异常：{len(questions)}")

        _record(5, name, True,
                f"learning_plan 阶段数={n_stages}, 面试题={len(questions)}（上限 {MAX_QUESTIONS}）")
    except Exception as e:
        _record(5, name, False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        selected = [int(x) for x in argv] if argv else [1, 2, 3, 4, 5]
    except ValueError:
        print("参数应为场景编号，例如：python pipeline.py 3 4")
        return 1

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("[警告] 未配置 DASHSCOPE_API_KEY，依赖 LLM/嵌入的场景会失败或走兜底。")

    print("=" * 64)
    print("Resume2Job 端到端验收  |  场景:", selected)
    print("=" * 64)

    pc, jd = setup_env()
    from resume2job.agent.graph import build_graph
    app = build_graph()
    ctx = {}

    for n in selected:
        if n == 1:
            scenario_1_jd_evaluation(app, ctx)
        elif n == 2:
            scenario_2_job_recommendation(app, ctx)
        elif n == 3:
            scenario_3_jd_ingest_and_search(app, jd, ctx)
        elif n == 4:
            scenario_4_profile_cache_hit(app, pc, ctx)
        elif n == 5:
            scenario_5_gap_plan_interview(app, ctx)
        else:
            print(f"[跳过] 未知场景编号：{n}")

    # 汇总
    print("\n" + "=" * 64)
    print("验收汇总")
    print("=" * 64)
    passed = sum(1 for _, _, ok, _ in _RESULTS if ok)
    for no, name, ok, detail in sorted(_RESULTS):
        print(f"  S{no} [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"        {detail}")
    print(f"\n结果：{passed} / {len(_RESULTS)} 通过")
    return 0 if passed == len(_RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
