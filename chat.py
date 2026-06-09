"""
chat.py

模拟聊天窗口的单程序交互入口（命令行 REPL）。

把每一条用户输入当作一轮对话，复用同一个 session_id 接入工作流，
因此具备多轮记忆：首轮给了简历后续可不再给、依赖上下文的追问可被理解。

用户可以在一句话里自然地带上文件路径，例如：
    我的简历地址是 Resumes/resume_test_1.pdf，我想看看该 JD：JDs/jd_test_1.txt 适合我吗

本入口会自动从输入中识别：
    - *.pdf  -> 简历路径（pdf_path）
    - *.txt  -> JD 文件，读取其内容作为 jd_text
    - 常见城市词 -> 检索城市过滤（city_filter）
其余文字作为用户问题交给 Agent 判断意图（评估 JD / 推荐岗位 / 技能差距）。

默认只输出推荐/评估报告；只有当你的问题里出现「学习计划 / 学习路径」「面试题 / 模拟面试」
等措辞时，才会额外生成阶段化学习计划、模拟面试题。

会话与多轮记忆：
    对话按 session_id 持久化。默认续接「最近一次会话」，因此重启也能接着聊；
    启动参数：--session_id <id> 续接指定会话，--new 强制新建会话。

斜杠命令：
    /help        查看帮助
    /history     查看当前会话的对话记录
    /sessions    列出所有历史会话及其消息数
    /new         切换到一个全新会话（不影响已存记录）
    /clear       清空当前会话的对话记录（session_id 不变）
    /exit /quit  退出
"""

import os
import re
import sys
import uuid
import argparse

# Windows 控制台中文输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from graph import build_graph, run_turn
from storage import conversation_store

# 常见城市词，用于从自然语言里粗提取检索城市过滤
_CITIES = ["北京", "上海", "深圳", "广州", "杭州", "成都", "南京", "武汉",
           "西安", "苏州", "天津", "重庆", "厦门", "合肥", "长沙"]

# 仅匹配真实路径字符（字母/数字/_ - . / \ :），让中文、引号、空格自然成为边界，
# 避免把「我的简历地址是"Resumes/x.pdf"」整段误当成路径。
_PDF_RE = re.compile(r"([A-Za-z0-9_./\\:\-]+\.pdf)", re.IGNORECASE)
_TXT_RE = re.compile(r"([A-Za-z0-9_./\\:\-]+\.txt)", re.IGNORECASE)


def _read_text(path: str) -> str:
    """UTF-8 读取文本文件，失败返回空串。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[警告] 读取文件失败 {path}：{e}")
        return ""


def extract_inputs(text: str):
    """从一句用户输入中提取 (pdf_path, jd_text, city)。识别不到的返回 None。"""
    pdf_path = None
    m = _PDF_RE.search(text)
    if m and os.path.isfile(m.group(1)):
        pdf_path = m.group(1)
    elif m:
        print(f"[提示] 识别到简历路径 {m.group(1)} 但文件不存在，已忽略。")

    jd_text = None
    m = _TXT_RE.search(text)
    if m and os.path.isfile(m.group(1)):
        jd_text = _read_text(m.group(1)) or None
    elif m:
        print(f"[提示] 识别到 JD 文件 {m.group(1)} 但文件不存在，已忽略。")

    city = next((c for c in _CITIES if c in text), None)
    return pdf_path, jd_text, city


def _print_assistant(final_state: dict) -> None:
    """把一轮结果以聊天气泡的形式打印给用户。"""
    print("\n助手>")

    match_results = final_state.get("match_results") or []
    final_response = (final_state.get("final_response") or "").strip()

    if final_response:
        print(f"  {final_response}\n")

    if match_results:
        # 单岗位（JD 评估）：直接展示完整报告；多岗位（推荐）：列表 + 首条报告
        if len(match_results) == 1:
            mr = match_results[0]
            jd = mr.get("jd_profile") or {}
            score = (mr.get("match_score") or {}).get("final_score")
            print(f"  【{jd.get('company', '?')} - {jd.get('title', '?')}】匹配度 {score}")
            report = (mr.get("report") or "").strip()
            if report:
                print("\n  " + report.replace("\n", "\n  "))
        else:
            print(f"  为你匹配到 {len(match_results)} 个岗位：")
            for i, mr in enumerate(match_results[:5], 1):
                jd = mr.get("jd_profile") or {}
                score = (mr.get("match_score") or {}).get("final_score")
                print(f"    {i}. {jd.get('company', '?')} - {jd.get('title', '?')}（匹配度 {score}）")
            top = match_results[0]
            report = (top.get("report") or "").strip()
            if report:
                print(f"\n  ——「{(top.get('jd_profile') or {}).get('title', '首选岗位')}」详情——")
                print("  " + report.replace("\n", "\n  "))

    # 学习计划（若有）
    plan = final_state.get("learning_plan") or {}
    if plan.get("stages"):
        print(f"\n  📚 学习计划（共 {plan.get('total_plan_days', 0)} 天，每天 {plan.get('daily_hours', 0)} 小时）：")
        for st in plan["stages"]:
            print(f"    - {st.get('stage')}（{st.get('days')}）：{st.get('goal')}")
        if plan.get("overall_suggestion"):
            print(f"    建议：{plan['overall_suggestion']}")

    # 模拟面试题（若有）
    interview = final_state.get("interview_prep") or {}
    questions = interview.get("questions") or []
    if questions:
        print(f"\n  🎤 模拟面试题（共 {len(questions)} 道）：")
        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                text = q.get("question") or ""
                kind = q.get("question_type")
            else:
                text, kind = str(q), None
            prefix = f"[{kind}] " if kind else ""
            print(f"    {i}. {prefix}{text}")
        if interview.get("summary"):
            print(f"    小结：{interview['summary']}")

    if not match_results and not final_response:
        print("  本轮没有生成岗位结果。可以试试：带上简历 PDF 路径，并说明想评估某个 JD 或推荐岗位。")

    # 只展示对用户有意义的错误（结构化错误取其 error 文本）
    errors = final_state.get("errors") or []
    cached = final_state.get("profile_source") == "cached"
    msgs = []
    for e in errors:
        msg = e.get("error") if isinstance(e, dict) else str(e)
        msg = str(msg)
        # 命中缓存画像的轮次里，「缺少简历 PDF」类告警已被缓存覆盖，属噪音，不展示
        if cached and ("pdf" in msg.lower() or "简历 PDF" in msg):
            continue
        msgs.append(msg)
    if msgs:
        print("\n  [提示] 本轮存在以下问题：")
        for msg in msgs:
            print(f"    - {msg}")
    print()


def _print_history(session_id: str) -> None:
    history = conversation_store.load_history(session_id, limit=50)
    if not history:
        print("（本会话暂无对话记录）\n")
        return
    print(f"\n——会话 {session_id} 的对话记录——")
    label = {"user": "你", "assistant": "助手"}
    for h in history:
        print(f"  {label.get(h['role'], h['role'])}：{h['content'][:80]}")
    print()


def _new_session_id() -> str:
    return "chat_" + uuid.uuid4().hex[:8]


def _resolve_session(args) -> str:
    """决定本次启动使用哪个 session_id：

        --session_id X  -> 用 X（接着该会话聊）；
        --new           -> 强制新建会话；
        默认            -> 续接最近一次会话（数据库里有则复用，无则新建）。
    """
    if args.session_id:
        return args.session_id
    if args.new:
        return _new_session_id()
    recent = conversation_store.list_sessions()
    return recent[0] if recent else _new_session_id()


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume2Job 模拟聊天窗口（多轮对话）")
    parser.add_argument("--session_id", default=None,
                        help="指定会话 ID（接着该会话的历史继续聊）")
    parser.add_argument("--new", action="store_true",
                        help="强制开启全新会话（默认续接最近一次会话）")
    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("[警告] 未配置 DASHSCOPE_API_KEY，简历解析/评分等能力将失败或走兜底。\n")

    print("=" * 60)
    print("Resume2Job 模拟聊天窗口（多轮对话）")
    print("输入 /help 查看命令；/exit 退出。")
    print("示例：我的简历地址是 Resumes/resume_test_1.pdf，看看 JDs/jd_test_1.txt 适合我吗")
    print("=" * 60)

    app = build_graph()
    session_id = _resolve_session(args)
    history = conversation_store.load_history(session_id, limit=50)
    if history:
        print(f"\n[续接会话 session_id={session_id}，已有 {len(history)} 条历史记录]")
        _print_history(session_id)
    else:
        print(f"\n[新会话 session_id={session_id}]\n")

    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not line:
            continue

        # ---- 斜杠命令 ----
        low = line.lower()
        if low in ("/exit", "/quit"):
            print("再见！")
            break
        if low == "/help":
            print(__doc__)
            continue
        if low == "/history":
            _print_history(session_id)
            continue
        if low == "/sessions":
            sessions = conversation_store.list_sessions()
            if not sessions:
                print("（暂无任何会话记录）\n")
            else:
                print("\n——历史会话（按最近活跃排序）——")
                for s in sessions:
                    n = len(conversation_store.load_history(s, limit=999))
                    mark = "  ← 当前" if s == session_id else ""
                    print(f"  {s}（{n} 条）{mark}")
                print("提示：python chat.py --session_id <id> 可续接指定会话\n")
            continue
        if low == "/new":
            session_id = _new_session_id()
            print(f"[已切换到新会话 session_id={session_id}]\n")
            continue
        if low == "/clear":
            conversation_store.clear_session(session_id)
            print(f"[已清空会话 session_id={session_id} 的对话记录]\n")
            continue

        # ---- 普通消息：解析输入 + 执行一轮 ----
        pdf_path, jd_text, city = extract_inputs(line)

        try:
            final_state = run_turn(
                app,
                user_query=line,
                session_id=session_id,
                pdf_path=pdf_path,
                jd_text=jd_text,
                city_filter=city,
                top_k=3,
            )
            _print_assistant(final_state)
        except Exception as e:
            print(f"[错误] 本轮执行失败：{type(e).__name__}: {e}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
