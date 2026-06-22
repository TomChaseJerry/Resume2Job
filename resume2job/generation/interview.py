"""
面试准备模块（Interview Prep，可选增强）。

定位：
    从「这个岗位适不适合我」升级为「如果我投这个岗位，面试官可能怎么问」。
    针对首选岗位生成固定 3 道高针对性模拟面试题：
        1 道项目深挖（验证简历真实性与个人贡献）；
        1 道核心技能题（matched 深挖或 weak 验证，取更有信息量者）；
        1 道缺口挑战题（missing 基础认知 / 风险挑战）。

设计原则（与项目「LLM 负责语义、Python 负责确定性编排」一致）：
    - LLM ：基于整理好的素材命题；
    - Python：候选来源整理、输出校验、补 id、去重、截断与规则兜底。
"""

import re
import json
from typing import Any, Optional, List, Dict

from resume2job.core.config import CHAT_MODEL as MODEL_NAME
from resume2job.core.llm import call_llm as _core_call_llm, safe_json_parse

# 固定生成 3 道题：少而准，覆盖「项目真实性 / 核心技能 / 关键缺口」三个考察面
MAX_QUESTIONS = 3

# ===== 合法取值集合 =====
VALID_QUESTION_TYPES = {
    "project_deep_dive",
    "technical_deep_dive",
    "weak_skill_probe",
    "missing_skill_basic",
    "behavioral_star",
    "risk_challenge",
}
VALID_SOURCES = {"project", "matched_skill", "weak_skill", "missing_skill", "jd_risk"}
VALID_RISK_LEVELS = {"low", "medium", "high"}

DEFAULT_QUESTION_TYPE = "project_deep_dive"
DEFAULT_SOURCE = "project"
DEFAULT_RISK_LEVEL = "medium"


# ---------------------------------------------------------------------------
# 通用工具函数
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 面试题生成
# ---------------------------------------------------------------------------
_QUESTIONS_SYSTEM_PROMPT = """你是一位资深 AI 算法岗面试官，擅长根据候选人简历、岗位 JD 和技能差距生成有针对性的练习题。

你生成的是基于该岗位 JD 与候选人简历定制的「岗位定制面试练习题」，供候选人针对性练习，
**不是、也不要暗示是任何公司的真实面试题库或原题**。

要求：
1. 只输出合法 JSON；
2. 不输出 Markdown、解释或额外文字；
3. 问题必须基于输入材料（岗位职责 / 核心技能要求 / 候选人项目经历 / 主要短板），不得编造候选人没有做过的项目或技能；
4. 每道题都必须包含 interviewer_intent；
5. 问题要贴合岗位、有针对性，不要泛泛而谈。"""


def _build_questions_user_prompt(jd_summary: dict, resume_summary: dict,
                                 buckets: Dict[str, List[str]],
                                 projects: List[str]) -> str:
    """构造 User Prompt：岗位 / 候选人 + 候选来源 + 输出 Schema。"""
    def dump(obj):
        return json.dumps(obj, ensure_ascii=False, indent=2)

    return f"""## 岗位信息
{dump(jd_summary)}

## 候选人信息
{dump(resume_summary)}

## Python 已整理的候选问题来源（请据此命题，不要超出这些素材）
- matched 技能（可出深挖题 technical_deep_dive）：{buckets['matched']}
- weak 技能（可出验证题 weak_skill_probe）：{buckets['weak']}
- missing 技能（可出基础题 missing_skill_basic / 挑战题 risk_challenge）：{buckets['missing']}
- 项目经历（可出 project_deep_dive / behavioral_star）：{projects}

## 命题要求
恰好生成 {MAX_QUESTIONS} 道题，每道覆盖一个不同的考察面：
1. 第 1 题：项目深挖（project_deep_dive 或 behavioral_star），覆盖项目目标、个人职责、难点与结果；
2. 第 2 题：核心技能（matched 深挖或 weak 验证，选对该岗位更关键的技能）；
3. 第 3 题：关键缺口（missing_skill_basic 或 risk_challenge）；missing 为空时改出第二道技能题。
每道题必须给出 interviewer_intent（面试官考察意图）与 evidence_basis（命题依据，来自输入材料）。

## 输出格式（严格 JSON，不要任何额外文字或 Markdown）
{{
  "questions": [
    {{
      "question_id": "q1",
      "question": "",
      "question_type": "project_deep_dive / technical_deep_dive / weak_skill_probe / missing_skill_basic / behavioral_star / risk_challenge",
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


def _fallback_questions(buckets: Dict[str, List[str]], projects: List[str]) -> List[dict]:
    """LLM 失败时的规则兜底：项目深挖 1 题 + 核心技能 1 题 + 缺口挑战 1 题。"""
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

    if projects:
        proj = projects[0]
        add(f"请用 STAR 方法介绍你在「{proj}」项目中一次关键问题的解决过程，并说明你的具体职责与最终结果。",
            "behavioral_star", "project", "", proj, "medium",
            "考察项目真实性与个人贡献，验证是否真正主导过关键问题。",
            [f"简历项目：{proj}"])

    core_skill = (buckets["matched"][:1] or buckets["weak"][:1] or [""])[0]
    if core_skill:
        add(f"你在项目中是如何具体使用 {core_skill} 的？请讲清楚实现细节和你做过的关键技术决策。",
            "technical_deep_dive", "matched_skill", core_skill, "", "low",
            f"验证候选人是否真正掌握 {core_skill}，而非简历堆砌。",
            [f"核心技能：{core_skill}"])

    gap_skill = (buckets["missing"][:1] or buckets["weak"][1:2] or [""])[0]
    if gap_skill:
        add(f"你在 {gap_skill} 方面经验有限，如果岗位要求这部分能力，你打算如何快速补齐？",
            "missing_skill_basic", "missing_skill", gap_skill, "", "high",
            f"考察对缺失技能 {gap_skill} 的基本认知与补齐规划。",
            [f"关键缺口：{gap_skill}"])

    return questions[:MAX_QUESTIONS]


def _validate_questions(parsed: Optional[dict], fallback: List[dict]) -> Dict[str, Any]:
    """输出校验：补 id、过滤空题、规范枚举值、去重、截断到 3 题；空则兜底。"""
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

    # 截断到 3 题后统一编号 q1, q2, q3
    cleaned = cleaned[:MAX_QUESTIONS]
    for i, q in enumerate(cleaned, start=1):
        q["question_id"] = f"q{i}"

    if not summary:
        summary = "本轮面试题覆盖项目真实性、核心技能掌握程度与岗位关键缺口三个考察面。"

    return {"questions": cleaned, "summary": summary, "error": None}


def generate_interview_questions(
    resume_profile: dict,
    jd_profile: dict,
    match_result: dict,
    skill_gap: dict,
) -> dict:
    """生成面向目标岗位的 3 道模拟面试题。

    流程：Python 整理候选来源 → LLM 命题 → Python 校验/补 id/去重/截断/兜底。
    任何失败都不会抛异常，error 字段恒存在（正常为 None）。
    """
    try:
        buckets = _collect_skill_buckets(match_result, skill_gap)
        projects = _to_name_list(_safe_get(resume_profile, "projects"))
        fallback = _fallback_questions(buckets, projects)

        user_prompt = _build_questions_user_prompt(
            _build_jd_summary(jd_profile),
            _build_resume_summary(resume_profile),
            buckets, projects,
        )

        try:
            # 命题任务温度略高（0.4），避免每次生成雷同问题
            raw = _core_call_llm(_QUESTIONS_SYSTEM_PROMPT, user_prompt,
                                 model=MODEL_NAME, temperature=0.4)
        except Exception as exc:
            print(f"[interview] LLM 调用失败：{exc}")
            raw = None
        parsed = safe_json_parse(raw) if raw else None

        return _validate_questions(parsed, fallback)
    except Exception as exc:
        # 极端兜底：构造最小可用结果，不让调用方崩溃
        print(f"[interview] 面试题生成异常：{exc}")
        return {
            "questions": _fallback_questions(
                _collect_skill_buckets(match_result, skill_gap),
                _to_name_list(_safe_get(resume_profile, "projects")),
            ),
            "summary": "面试题生成出现异常，已返回规则兜底问题。",
            "error": f"面试题生成异常：{exc}",
        }
