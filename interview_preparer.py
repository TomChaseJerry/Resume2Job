"""
interview_preparer.py

面试准备增强模块（Skill Gap / Learning Plan 之后的可选增强）。

定位：
    从「这个岗位适不适合我」升级为「如果我投这个岗位，面试官可能怎么问，我该怎么答」。

两层能力：
    第一层 generate_interview_questions：
        基于简历画像 / JD / 匹配评分 / 技能差距，生成面向目标岗位的模拟面试题。
        Python 先按 matched / weak / missing / 项目经历整理候选来源，再交给 LLM 生成，
        最后由 Python 做校验、补 id、去重、截断与兜底。
    第二层 generate_answer_framework：
        用户点击某道题后，按该题 question_type 对应的作答逻辑生成「结构化作答框架」，
        只给框架不给背诵稿。

设计原则（与项目「LLM 负责语义、Python 负责确定性编排」一致）：
    - LLM ：生成问题与作答框架；
    - Python：输入摘要构造、JSON 清洗、输出校验、去重、兜底。

外部依赖：
    - 阿里云百炼（OpenAI 兼容）API，模型 qwen-max，Key 从环境变量 DASHSCOPE_API_KEY 读取。
"""

import os
import re
import sys
import json
from typing import Any, Optional, List, Dict


# ===== LLM 接口常量（与项目其它模块一致）=====
MODEL_NAME = "qwen-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# 若项目已提供统一 LLM 客户端工厂则优先复用；否则按本仓库现有 ChatOpenAI 模式构造。
try:  # pragma: no cover - 取决于项目是否存在 llm_client 模块
    from llm_client import get_llm_client  # type: ignore
except Exception:
    def get_llm_client():
        """构造阿里云百炼（OpenAI 兼容）客户端；缺 Key 或依赖缺失时返回 None。"""
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("[interview_preparer] 未配置 DASHSCOPE_API_KEY，将使用规则兜底")
            return None
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=MODEL_NAME,
                openai_api_key=api_key,
                openai_api_base=BASE_URL,
                temperature=0.4,
            )
        except Exception as exc:
            print(f"[interview_preparer] LLM 客户端初始化失败：{exc}")
            return None


# ===== 合法取值集合 =====
VALID_QUESTION_TYPES = {
    "project_deep_dive",
    "technical_deep_dive",
    "weak_skill_probe",
    "missing_skill_basic",
    "system_design",
    "behavioral_star",
    "risk_challenge",
}
VALID_SOURCES = {"project", "matched_skill", "weak_skill", "missing_skill", "jd_risk"}
VALID_RISK_LEVELS = {"low", "medium", "high"}

DEFAULT_QUESTION_TYPE = "project_deep_dive"
DEFAULT_SOURCE = "project"
DEFAULT_RISK_LEVEL = "medium"


# ===== 各题型作答策略（既用于 Prompt，也用于兜底框架）=====
ANSWER_STRATEGY: Dict[str, List[str]] = {
    "project_deep_dive": ["项目背景", "个人职责", "技术方案", "难点", "结果", "反思"],
    "technical_deep_dive": ["概念解释", "为什么使用", "怎么实现", "如何验证", "局限性"],
    "weak_skill_probe": ["承认没有完整直接经验", "说明已有可迁移基础", "说明如何快速补齐", "回到岗位需求"],
    "missing_skill_basic": ["先给基础认知", "承认目前实践不足", "说明学习计划", "避免夸大"],
    "system_design": ["需求拆解", "模块划分", "数据流", "异常处理", "可扩展性"],
    "behavioral_star": ["Situation 情境", "Task 任务", "Action 行动", "Result 结果", "Reflection 反思"],
    "risk_challenge": ["正面回应短板", "给出已有相关基础", "给出补强路径", "表达可快速上手"],
}


# ---------------------------------------------------------------------------
# 通用工具函数
# ---------------------------------------------------------------------------
def clean_llm_json_output(raw: str) -> str:
    """去除 ```json / ``` 包裹，截取首 '{' 到末 '}'。"""
    if not raw:
        return ""
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return t


def safe_json_parse(raw: str) -> Optional[dict]:
    """清洗 + 解析 LLM 输出为 dict；失败返回 None。"""
    if not raw or not raw.strip():
        return None
    try:
        obj = json.loads(clean_llm_json_output(raw))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _to_name_list(value: Any) -> List[str]:
    """把技能/项目字段统一成去重后的名称字符串列表，兼容 str / dict 列表。"""
    out: List[str] = []
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
            elif isinstance(v, dict):
                name = v.get("skill") or v.get("name") or v.get("title")
                if name and str(name).strip():
                    out.append(str(name).strip())
    elif isinstance(value, str) and value.strip():
        out.append(value.strip())

    seen, result = set(), []
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            result.append(x)
    return result


def _as_str_list(value: Any) -> List[str]:
    """把任意值规整为非空字符串列表。"""
    if isinstance(value, list):
        return [str(x).strip() for x in value if isinstance(x, (str, int, float)) and str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    """安全取字段，d 非 dict 时返回默认值。"""
    return d.get(key, default) if isinstance(d, dict) else default


def _load_json_file(path: str) -> Any:
    """读取并解析 JSON 文件；文件不存在或解析失败时打印错误并退出。"""
    if not os.path.isfile(path):
        print(f"[ERROR] 文件不存在：{path}")
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 解析失败（{path}）：{e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 输入摘要构造（Python，喂给 LLM 前先收敛字段）
# ---------------------------------------------------------------------------
def _collect_skill_buckets(match_result: dict, skill_gap: dict) -> Dict[str, List[str]]:
    """合并 skill_gap 与 match_result 中的 matched / weak / missing 技能名。"""
    skill_score = _safe_get(match_result, "skill_score", {}) or {}

    matched = _to_name_list(_safe_get(skill_gap, "matched_skills")) + \
        _to_name_list(_safe_get(skill_score, "matched_skills"))
    weak = _to_name_list(_safe_get(skill_gap, "weak_skills")) + \
        _to_name_list(_safe_get(skill_score, "weak_matched_skills"))
    missing = _to_name_list(_safe_get(skill_gap, "missing_skills")) + \
        _to_name_list(_safe_get(skill_score, "missing_skills"))

    def _dedup(seq: List[str]) -> List[str]:
        seen, res = set(), []
        for x in seq:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                res.append(x)
        return res

    return {"matched": _dedup(matched), "weak": _dedup(weak), "missing": _dedup(missing)}


def _build_jd_summary(jd_profile: dict) -> dict:
    """提炼岗位关键信息。"""
    jd = jd_profile or {}
    return {
        "company": jd.get("company"),
        "title": jd.get("title"),
        "direction": jd.get("direction"),
        "responsibilities": jd.get("responsibilities"),
        "hard_skills": jd.get("hard_skills"),
        "preferred_skills": jd.get("preferred_skills"),
    }


def _build_resume_summary(resume_profile: dict) -> dict:
    """提炼候选人关键信息。"""
    r = resume_profile or {}
    return {
        "skills": r.get("skills"),
        "skill_groups": r.get("skill_groups"),
        "projects": r.get("projects"),
        "experiences": r.get("experiences"),
        "research": r.get("research"),
        "awards": r.get("awards"),
    }


def _build_match_summary(match_result: dict) -> dict:
    """提炼匹配评分摘要。"""
    m = match_result or {}
    return {
        "final_score": m.get("final_score"),
        "match_level": m.get("match_level"),
        "skill_score": m.get("skill_score"),
        "project_score": m.get("project_score"),
        "direction_score": m.get("direction_score"),
    }


def _build_skill_gap_summary(skill_gap: dict) -> dict:
    """提炼 Skill Gap 摘要。"""
    s = skill_gap or {}
    return {
        "matched_skills": s.get("matched_skills"),
        "weak_skills": s.get("weak_skills"),
        "missing_skills": s.get("missing_skills"),
        "items": s.get("items"),
        "overall_risk": s.get("overall_risk"),
        "overall_risk_reason": s.get("overall_risk_reason"),
    }


# ===========================================================================
# 第一部分：面试题生成
# ===========================================================================
_QUESTIONS_SYSTEM_PROMPT = """你是一位资深 AI 算法岗面试官，擅长根据候选人简历、岗位 JD、匹配评分和技能差距生成有针对性的面试题。

要求：
1. 只输出合法 JSON；
2. 不输出 Markdown、解释或额外文字；
3. 问题必须基于输入材料；
4. 不得编造候选人没有做过的项目或技能；
5. 对 matched 技能生成深挖题；
6. 对 weak 技能生成验证题；
7. 对 missing 技能生成基础认知题或风险挑战题；
8. 对项目经历生成项目深挖题和 STAR 行为题；
9. 每道题都必须包含 interviewer_intent；
10. 问题要像真实面试官会问的问题，不要泛泛而谈。"""


def _build_questions_user_prompt(jd_summary: dict, resume_summary: dict, match_summary: dict,
                                 skill_gap_summary: dict, buckets: Dict[str, List[str]],
                                 projects: List[str], max_questions: int) -> str:
    """构造第一层 User Prompt：岗位 / 候选人 / 评分 / 差距 + 候选来源 + 输出 Schema。"""
    def dump(obj):
        return json.dumps(obj, ensure_ascii=False, indent=2)

    return f"""## 岗位信息
{dump(jd_summary)}

## 候选人信息
{dump(resume_summary)}

## 匹配评分摘要
{dump(match_summary)}

## Skill Gap 摘要
{dump(skill_gap_summary)}

## Python 已整理的候选问题来源（请据此命题，不要超出这些素材）
- matched 技能（出深挖题 technical_deep_dive / project_deep_dive）：{buckets['matched']}
- weak 技能（出验证题 weak_skill_probe / risk_challenge）：{buckets['weak']}
- missing 技能（出基础题 missing_skill_basic / risk_challenge）：{buckets['missing']}
- 项目经历（出 project_deep_dive / system_design / behavioral_star）：{projects}

## 命题要求
1. 总问题数不超过 {max_questions}，信息不足时可减少，不要硬凑；
2. 建议分配：项目深挖题 3~4、matched 深挖题 2~3、weak 验证题 2~3、missing 基础题 1~2、行为/挑战题 1~2；
3. matched 技能出深挖题、weak 技能出验证题、missing 技能出基础认知/风险挑战题；
4. 项目题需覆盖：项目目标、个人职责、技术选型、难点、实验或评估、结果、可迁移到目标岗位的能力；
5. 每道题必须给出 interviewer_intent（面试官考察意图）与 evidence_basis（命题依据，来自输入材料）。

## 输出格式（严格 JSON，不要任何额外文字或 Markdown）
{{
  "questions": [
    {{
      "question_id": "q1",
      "question": "",
      "question_type": "project_deep_dive / technical_deep_dive / weak_skill_probe / missing_skill_basic / system_design / behavioral_star / risk_challenge",
      "source": "project / matched_skill / weak_skill / missing_skill / jd_risk",
      "related_skill": "",
      "related_project": "",
      "risk_level": "low / medium / high",
      "interviewer_intent": "",
      "evidence_basis": []
    }}
  ],
  "summary": "",
  "error": null
}}"""


def _call_llm(llm_client, system_prompt: str, user_prompt: str) -> Optional[str]:
    """调用 LLM 并返回原始文本；任何异常返回 None（交由上层兜底）。"""
    if llm_client is None:
        return None
    try:
        response = llm_client.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        return getattr(response, "content", "") or ""
    except Exception as exc:
        print(f"[interview_preparer] LLM 调用失败：{exc}")
        return None


def _fallback_questions(buckets: Dict[str, List[str]], projects: List[str],
                        max_questions: int) -> List[dict]:
    """LLM 失败时的规则兜底问题列表，按建议分配从各来源取题。"""
    questions: List[dict] = []

    def add(question, qtype, source, skill, project, risk, intent, evidence):
        questions.append({
            "question": question,
            "question_type": qtype,
            "source": source,
            "related_skill": skill,
            "related_project": project,
            "risk_level": risk,
            "interviewer_intent": intent,
            "evidence_basis": evidence,
        })

    # 项目深挖题 / STAR（最多 4）
    for proj in projects[:4]:
        add(f"请用 STAR 方法介绍你在「{proj}」项目中一次关键问题的解决过程，并说明你的具体职责与最终结果。",
            "behavioral_star", "project", "", proj, "medium",
            "考察项目真实性与个人贡献，验证是否真正主导过关键问题。",
            [f"简历项目：{proj}"])

    # matched 技能深挖题（最多 3）
    for sk in buckets["matched"][:3]:
        add(f"你在项目中是如何具体使用 {sk} 的？请讲清楚实现细节和你做过的关键技术决策。",
            "technical_deep_dive", "matched_skill", sk, "", "low",
            f"验证候选人是否真正掌握 {sk}，而非简历堆砌。",
            [f"matched 技能：{sk}"])

    # weak 技能验证题（最多 3）
    for sk in buckets["weak"][:3]:
        add(f"你对 {sk} 目前掌握到什么程度？如果岗位需要，你会如何把已有经验迁移到这个场景？",
            "weak_skill_probe", "weak_skill", sk, "", "medium",
            f"验证 {sk} 是停留在概念了解还是具备迁移能力。",
            [f"weak 技能：{sk}"])

    # missing 技能基础题（最多 2）
    for sk in buckets["missing"][:2]:
        add(f"你了解 {sk} 的基本原理吗？如果岗位要求这部分能力，你打算如何快速补齐？",
            "missing_skill_basic", "missing_skill", sk, "", "high",
            f"考察对缺失技能 {sk} 的基本认知与补齐规划。",
            [f"missing 技能：{sk}"])

    # 风险挑战题（最多 1，优先 missing 再 weak）
    challenge_skill = (buckets["missing"][:1] or buckets["weak"][:1] or [""])[0]
    if challenge_skill:
        add(f"你在 {challenge_skill} 方面经验有限，为什么你认为自己依然能胜任这个岗位？",
            "risk_challenge", "jd_risk", challenge_skill, "", "high",
            "考察候选人面对短板时的应对与抗压表达。",
            [f"关键缺口：{challenge_skill}"])

    return questions[:max_questions]


def _validate_questions(parsed: Optional[dict], max_questions: int,
                        fallback: List[dict]) -> Dict[str, Any]:
    """第一层输出校验：补 id、过滤空题、规范枚举值、去重、截断；空则兜底。"""
    raw_questions = []
    summary = ""
    if isinstance(parsed, dict):
        rq = parsed.get("questions")
        if isinstance(rq, list):
            raw_questions = rq
        if isinstance(parsed.get("summary"), str):
            summary = parsed["summary"].strip()

    # LLM 没给出任何题，直接用兜底
    if not raw_questions:
        raw_questions = fallback

    cleaned: List[dict] = []
    seen_questions = set()
    for q in raw_questions:
        if not isinstance(q, dict):
            continue
        text = str(q.get("question") or "").strip()
        if not text:
            continue
        dedup_key = re.sub(r"\s+", "", text).lower()
        if dedup_key in seen_questions:
            continue
        seen_questions.add(dedup_key)

        qtype = q.get("question_type")
        if qtype not in VALID_QUESTION_TYPES:
            qtype = DEFAULT_QUESTION_TYPE
        source = q.get("source")
        if source not in VALID_SOURCES:
            source = DEFAULT_SOURCE
        risk = q.get("risk_level")
        if risk not in VALID_RISK_LEVELS:
            risk = DEFAULT_RISK_LEVEL

        intent = str(q.get("interviewer_intent") or "").strip()
        if not intent:
            intent = "考察候选人在该问题上的真实经验与表达能力。"

        cleaned.append({
            "question_id": "",  # 占位，稍后统一编号
            "question": text,
            "question_type": qtype,
            "source": source,
            "related_skill": str(q.get("related_skill") or "").strip(),
            "related_project": str(q.get("related_project") or "").strip(),
            "risk_level": risk,
            "interviewer_intent": intent,
            "evidence_basis": _as_str_list(q.get("evidence_basis")),
        })

    # 截断到上限后统一编号 q1, q2, ...
    cleaned = cleaned[:max_questions]
    for i, q in enumerate(cleaned, start=1):
        q["question_id"] = f"q{i}"

    if not summary:
        summary = "本轮面试题围绕项目真实性、核心技能掌握程度与岗位关键缺口展开。"

    return {"questions": cleaned, "summary": summary, "error": None}


def generate_interview_questions(
    resume_profile: dict,
    jd_profile: dict,
    match_result: dict,
    skill_gap: dict,
    max_questions: int = 12,
) -> dict:
    """第一层：生成面向目标岗位的模拟面试题。

    流程：Python 整理候选来源 → LLM 命题 → Python 校验/补 id/去重/截断/兜底。
    任何失败都不会抛异常，error 字段恒存在（正常为 None）。
    """
    try:
        buckets = _collect_skill_buckets(match_result, skill_gap)
        projects = _to_name_list(_safe_get(resume_profile, "projects"))
        fallback = _fallback_questions(buckets, projects, max_questions)

        user_prompt = _build_questions_user_prompt(
            _build_jd_summary(jd_profile),
            _build_resume_summary(resume_profile),
            _build_match_summary(match_result),
            _build_skill_gap_summary(skill_gap),
            buckets, projects, max_questions,
        )

        llm_client = get_llm_client()
        raw = _call_llm(llm_client, _QUESTIONS_SYSTEM_PROMPT, user_prompt)
        parsed = safe_json_parse(raw) if raw else None

        return _validate_questions(parsed, max_questions, fallback)
    except Exception as exc:
        # 极端兜底：构造最小可用结果，不让调用方崩溃
        print(f"[interview_preparer] 面试题生成异常：{exc}")
        return {
            "questions": _fallback_questions(
                _collect_skill_buckets(match_result, skill_gap),
                _to_name_list(_safe_get(resume_profile, "projects")),
                max_questions,
            ),
            "summary": "面试题生成出现异常，已返回规则兜底问题。",
            "error": f"面试题生成异常：{exc}",
        }


# ===========================================================================
# 第二部分：作答思路框架生成
# ===========================================================================
_ANSWER_SYSTEM_PROMPT = """你是一位资深 AI 算法岗面试辅导专家，擅长将面试问题转化为结构化作答框架。

要求：
1. 只输出合法 JSON；
2. 不输出 Markdown、解释或额外文字；
3. 不编造候选人没有做过的经历；
4. 不生成完整背诵稿，只生成作答框架；
5. 作答框架必须体现该题型对应的回答逻辑；
6. 对 missing 或 weak 技能，不要伪装成已经熟练掌握；
7. 回答要真实、克制、可面试表达。"""


def _build_answer_user_prompt(question_item: dict, resume_summary: dict, jd_summary: dict,
                              skill_gap_summary: dict, match_summary: dict,
                              qtype: str) -> str:
    """构造第二层 User Prompt：当前题 + 各摘要 + 该题型回答策略 + 输出 Schema。"""
    def dump(obj):
        return json.dumps(obj, ensure_ascii=False, indent=2)

    strategy = ANSWER_STRATEGY.get(qtype, ANSWER_STRATEGY[DEFAULT_QUESTION_TYPE])
    strategy_text = " → ".join(strategy)

    return f"""## 当前问题
{dump(question_item)}

## 候选人简历摘要
{dump(resume_summary)}

## 岗位 JD 摘要
{dump(jd_summary)}

## Skill Gap 摘要
{dump(skill_gap_summary)}

## 匹配评分摘要
{dump(match_summary)}

## 本题作答策略（question_type = {qtype}）
回答必须遵循以下逻辑顺序：{strategy_text}

## 任务
请生成「作答框架」，不是完整背诵稿：
1. opening：一句话开场，点明核心结论；
2. key_points：至少 3 条要点，体现上面的作答逻辑顺序；
3. example_evidence：可引用的真实证据（来自简历/项目），不得编造，可为空；
4. pitfalls：至少 2 条本题常见的回答误区；
5. closing：一句话收尾，回到岗位价值。

## 输出格式（严格 JSON，不要任何额外文字或 Markdown）
{{
  "question": "",
  "question_type": "",
  "risk_level": "",
  "interviewer_intent": "",
  "answer_framework": {{
    "opening": "",
    "key_points": [],
    "example_evidence": [],
    "pitfalls": [],
    "closing": ""
  }},
  "error": null
}}"""


def _fallback_answer_framework(question_item: dict, qtype: str) -> dict:
    """LLM 失败时的规则兜底作答框架，按题型策略生成 key_points 与 pitfalls。"""
    question = str(_safe_get(question_item, "question", "") or "").strip()
    strategy = ANSWER_STRATEGY.get(qtype, ANSWER_STRATEGY[DEFAULT_QUESTION_TYPE])

    # 题型相关的通用误区
    if qtype in ("weak_skill_probe", "missing_skill_basic", "risk_challenge"):
        pitfalls = ["不要夸大未真正掌握的技能", "不要回避短板，要给出可执行的补强路径"]
    elif qtype == "behavioral_star":
        pitfalls = ["不要只讲团队成果，要讲清个人具体行动", "不要省略可量化的结果"]
    else:
        pitfalls = ["不要泛泛而谈，要结合具体项目、数据或决策", "不要编造没有做过的经历或细节"]

    return {
        "opening": f"针对「{question}」，我会先给出核心结论，再按逻辑展开。" if question
                   else "我会先给出核心结论，再按逻辑展开。",
        "key_points": list(strategy),
        "example_evidence": [],
        "pitfalls": pitfalls,
        "closing": "最后我会总结这段经历/能力对目标岗位的价值，并说明可以快速上手的部分。",
    }


def _validate_answer_framework(parsed: Optional[dict], question_item: dict,
                               qtype: str, risk_level: str) -> dict:
    """第二层输出校验：保证 answer_framework 完整，key_points≥3、pitfalls≥2，空则兜底。"""
    fallback = _fallback_answer_framework(question_item, qtype)

    af = parsed.get("answer_framework") if isinstance(parsed, dict) else None
    if not isinstance(af, dict):
        af = {}

    opening = str(af.get("opening") or "").strip() or fallback["opening"]
    closing = str(af.get("closing") or "").strip() or fallback["closing"]

    key_points = _as_str_list(af.get("key_points"))
    for kp in fallback["key_points"]:
        if len(key_points) >= 3:
            break
        if kp not in key_points:
            key_points.append(kp)

    pitfalls = _as_str_list(af.get("pitfalls"))
    for p in fallback["pitfalls"]:
        if len(pitfalls) >= 2:
            break
        if p not in pitfalls:
            pitfalls.append(p)

    example_evidence = _as_str_list(af.get("example_evidence"))  # 可为空，不编造

    # 顶层字段优先用 LLM 输出，缺失时回退到 question_item
    parsed = parsed if isinstance(parsed, dict) else {}
    question = str(parsed.get("question") or _safe_get(question_item, "question", "") or "").strip()
    intent = str(parsed.get("interviewer_intent")
                 or _safe_get(question_item, "interviewer_intent", "") or "").strip()

    return {
        "question": question,
        "question_type": qtype,
        "risk_level": risk_level,
        "interviewer_intent": intent,
        "answer_framework": {
            "opening": opening,
            "key_points": key_points,
            "example_evidence": example_evidence,
            "pitfalls": pitfalls,
            "closing": closing,
        },
        "error": None,
    }


def generate_answer_framework(
    question_item: dict,
    resume_profile: dict,
    jd_profile: dict,
    match_result: dict,
    skill_gap: dict,
) -> dict:
    """第二层：为某一道题生成结构化作答框架（只给框架，不给背诵稿）。

    按 question_type 选择回答策略；LLM 失败或解析失败时返回规则兜底框架，不抛异常。
    """
    try:
        qtype = _safe_get(question_item, "question_type")
        if qtype not in VALID_QUESTION_TYPES:
            qtype = DEFAULT_QUESTION_TYPE
        risk_level = _safe_get(question_item, "risk_level")
        if risk_level not in VALID_RISK_LEVELS:
            risk_level = DEFAULT_RISK_LEVEL

        user_prompt = _build_answer_user_prompt(
            question_item or {},
            _build_resume_summary(resume_profile),
            _build_jd_summary(jd_profile),
            _build_skill_gap_summary(skill_gap),
            _build_match_summary(match_result),
            qtype,
        )

        llm_client = get_llm_client()
        raw = _call_llm(llm_client, _ANSWER_SYSTEM_PROMPT, user_prompt)
        parsed = safe_json_parse(raw) if raw else None

        return _validate_answer_framework(parsed, question_item or {}, qtype, risk_level)
    except Exception as exc:
        print(f"[interview_preparer] 作答框架生成异常：{exc}")
        qtype = _safe_get(question_item, "question_type")
        if qtype not in VALID_QUESTION_TYPES:
            qtype = DEFAULT_QUESTION_TYPE
        result = _validate_answer_framework(None, question_item or {}, qtype, DEFAULT_RISK_LEVEL)
        result["error"] = f"作答框架生成异常：{exc}"
        return result


# ---------------------------------------------------------------------------
# main 入口（两种调试模式）
# ---------------------------------------------------------------------------
def _print_usage() -> None:
    print("用法：")
    print("  python interview_preparer.py questions <resume.json> <jd.json> <match.json> <skill_gap.json>")
    print("  python interview_preparer.py answer <question_item.json> <resume.json> <jd.json> <match.json> <skill_gap.json>")


def main(argv: Optional[List[str]] = None) -> int:
    """命令行调试入口：questions 生成面试题；answer 生成某题作答框架。"""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        _print_usage()
        return 1

    mode = argv[0].lower()

    if mode == "questions":
        if len(argv) < 5:
            _print_usage()
            return 1
        resume_profile = _load_json_file(argv[1])
        jd_profile = _load_json_file(argv[2])
        match_result = _load_json_file(argv[3])
        skill_gap = _load_json_file(argv[4])
        result = generate_interview_questions(resume_profile, jd_profile, match_result, skill_gap)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if mode == "answer":
        if len(argv) < 6:
            _print_usage()
            return 1
        question_item = _load_json_file(argv[1])
        resume_profile = _load_json_file(argv[2])
        jd_profile = _load_json_file(argv[3])
        match_result = _load_json_file(argv[4])
        skill_gap = _load_json_file(argv[5])
        result = generate_answer_framework(question_item, resume_profile, jd_profile, match_result, skill_gap)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    _print_usage()
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出
    except Exception:
        pass
    sys.exit(main())
