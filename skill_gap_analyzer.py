"""
技能差距分析模块（Skill Gap Analyzer）

定位：岗位评估流程中的固定内部分析步骤，链路为
    match_scorer → skill_gap_analyzer → recommendation_writer

每次完成岗位匹配评分后调用本模块，对 JD 的核心能力逐条分析，输出结构化技能差距结果，
供 recommendation_writer 生成更有证据、更有建议的推荐报告。

分工：
    - LLM：只负责生成 items（逐项技能的 status / evidence / gap_reason / suggestion）；
    - Python：负责选目标技能、聚合 matched/weak/missing、计算 overall_risk、生成 top_suggestion，
      并保证与 Match Scorer 结论一致、失败时规则兜底。
"""

import os
import re
import sys
import json
import argparse
from typing import Optional, Any


# ===== 模型 / 接口常量 =====
MODEL_NAME = "qwen-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ===== 目标技能选择上限 =====
MAX_MUST = 7        # must 技能最多分析数量
MAX_PREFERRED = 3   # preferred 技能最多分析数量
MAX_ITEMS = 10      # items 总条数上限


# ===== 空结构（异常兜底用）=====
def _empty_result(error: Optional[str] = None) -> dict:
    """返回符合 Schema 的空结构。"""
    return {
        "items": [],
        "matched_skills": [],
        "weak_skills": [],
        "missing_skills": [],
        "overall_risk": "low",
        "overall_risk_reason": "未能产出技能差距分析结果。",
        "top_suggestion": "当前匹配情况较好，建议正常投递，并在简历中突出相关项目证据。",
        "error": error,
    }


# ===== 工具函数 =====
def _dedup_keep_order(items: list) -> list:
    """列表去重并保持原顺序（仅对非空字符串生效）。"""
    seen = set()
    result = []
    for x in items or []:
        if not isinstance(x, str):
            continue
        key = x.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _lower_set(items: list) -> set:
    """转成小写去空格集合，便于不区分大小写比对。"""
    return {x.strip().lower() for x in (items or []) if isinstance(x, str) and x.strip()}


def _safe_str(value: Any, default: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


# ===== 1. 目标技能选择 =====
def _select_target_skills(jd_profile: dict, match_result: dict) -> tuple:
    """选择 5~10 个最关键的能力项。

    返回 (must_selected, preferred_selected)。
    规则：
      - must = hard_skills + tools_or_frameworks（hard_skills 为空时退化到 domain_keywords）；
      - preferred = preferred_skills，最多 3 个；
      - must 内部按「missing > weak > matched > 其它」的优先级稳定排序，取前 MAX_MUST；
      - 这样既优先分析缺口，也保留命中/弱匹配项用于解释优势。
    """
    must = _dedup_keep_order(
        list(jd_profile.get("hard_skills") or []) +
        list(jd_profile.get("tools_or_frameworks") or [])
    )
    # hard_skills 为空时，才用 domain_keywords 兜底
    if not must:
        must = _dedup_keep_order(jd_profile.get("domain_keywords") or [])

    preferred = _dedup_keep_order(jd_profile.get("preferred_skills") or [])

    skill_score = match_result.get("skill_score") or {}
    missing_set = _lower_set(skill_score.get("missing_skills"))
    weak_set = _lower_set(skill_score.get("weak_matched_skills"))
    matched_set = _lower_set(skill_score.get("matched_skills"))

    def _rank(s: str) -> int:
        low = s.strip().lower()
        if low in missing_set:
            return 0
        if low in weak_set:
            return 1
        if low in matched_set:
            return 2
        return 3

    must_sorted = sorted(must, key=_rank)  # sorted 是稳定排序，同级保持原顺序
    must_selected = must_sorted[:MAX_MUST]
    preferred_selected = preferred[:MAX_PREFERRED]

    # 控制总量不超过 MAX_ITEMS
    if len(must_selected) + len(preferred_selected) > MAX_ITEMS:
        preferred_selected = preferred_selected[: max(0, MAX_ITEMS - len(must_selected))]

    return must_selected, preferred_selected


# ===== 2. 简历子集（用于 User Prompt，节省 token 且只暴露可作证据的字段）=====
def _resume_subset_for_prompt(resume_profile: dict) -> dict:
    """裁剪简历，只保留可作为证据的字段。"""
    projects = []
    for p in resume_profile.get("projects") or []:
        if not isinstance(p, dict):
            continue
        projects.append({
            "name": p.get("name"),
            "description": p.get("description"),
            "tech_stack": p.get("tech_stack") or [],
            "tasks": p.get("tasks") or [],
            "keywords": p.get("keywords") or [],
            "evidence": p.get("evidence") or [],
        })

    experiences = []
    for e in resume_profile.get("experiences") or []:
        if not isinstance(e, dict):
            continue
        experiences.append({
            "company": e.get("company"),
            "title": e.get("title"),
            "keywords": e.get("keywords") or [],
            "tasks": e.get("tasks") or [],
        })

    # 学历/课程（弱证据来源）：课程通常写在 educations[].highlights，
    # 提供给 LLM 后，课程里出现过的能力可判为 weak（学过但无项目实践），而非直接 missing。
    educations = []
    for ed in resume_profile.get("educations") or []:
        if not isinstance(ed, dict):
            continue
        educations.append({
            "major": ed.get("major"),
            "courses_or_highlights": ed.get("highlights") or [],
        })

    return {
        "skills": resume_profile.get("skills") or [],
        "skill_groups": resume_profile.get("skill_groups") or [],
        "educations": educations,
        "projects": projects,
        "experiences": experiences,
        "research": resume_profile.get("research") or [],
        "publications": resume_profile.get("publications") or [],
        "awards": resume_profile.get("awards") or resume_profile.get("competitions") or [],
    }


# ===== 3. LLM Prompt =====
SYSTEM_PROMPT = """你是一位专业求职辅导专家，擅长根据候选人简历和岗位要求进行技能差距分析。

要求：
1. 只能输出合法 JSON；
2. 不要输出 Markdown 代码块标记（如 ```json）；
3. 不要输出解释性前后缀；
4. 不得编造简历中不存在的信息；
5. 每个技能都必须判断为 matched、weak 或 missing；
6. resume_evidence 必须来自候选人简历内容（skills / skill_groups / educations 课程 / projects / experiences / research / publications / awards），没有证据时返回空列表 []；
7. suggestion 必须具体、可执行；
8. 只输出如下结构：
{
  "items": [
    {
      "skill": "string",
      "importance": "must / preferred",
      "status": "matched / weak / missing",
      "resume_evidence": ["string"],
      "gap_reason": "string",
      "suggestion": "string"
    }
  ]
}

【判断标准】（三档：matched / weak / missing，弱匹配应优先于直接判缺口）
matched：简历 projects / experiences / research / publications 中有**项目或实践级**直接证据支撑该能力。
weak（有基础但缺直接项目实践——满足以下任一即应判 weak，不得直接判 missing）：
  - 技能栏 skills / skill_groups 中**明确列出**该技能，但没有对应项目 / 实习实践证据；
  - educations.courses_or_highlights（课程）中出现该技能或其上位能力，但无项目实践（如课程「强化学习 / 自然语言处理 / 计算机视觉」）；
  - 简历具备可迁移的相邻能力：GNN / 多源传感器建模 之于「多传感器信息处理」；深度学习之于「模仿学习 / 强化学习」；RAG / Agent 之于「大模型应用」。
missing（缺口）：技能栏、课程、项目、经历中**都完全没有**该技能或其相邻能力的任何描述。
注意：技能栏或课程里明确写了的能力（如机器学习、深度学习、NLP、CV），至少是 weak，不能判为 missing；
仅在简历完全未提及时才判 missing。

【重要约束】
1. 不要因为候选人是计算机专业，就推断其掌握 Python / PyTorch / C++ / ROS 等技能；
2. 编程语言、框架、工具必须在简历中显式出现，才能判为 matched；
3. 组合技能要分别判断（如「C/C++ 和 Python」分别判断 C/C++ 与 Python；「深度学习及模仿学习」分别判断深度学习与模仿学习）；
4. 不要把求职意向当作项目证据；不要把课程背景写成项目经历；不要编造简历中不存在的经历。"""


def _build_user_prompt(
    resume_subset: dict,
    must_skills: list,
    preferred_skills: list,
    match_result: dict,
) -> str:
    """构造 User Prompt：携带简历证据字段、岗位 must/preferred 技能、职责摘要、Match Scorer 结论。"""
    skill_score = match_result.get("skill_score") or {}
    match_ref = {
        "matched_skills": skill_score.get("matched_skills") or [],
        "weak_matched_skills": skill_score.get("weak_matched_skills") or [],
        "missing_skills": skill_score.get("missing_skills") or [],
        "preferred_matched_skills": skill_score.get("preferred_matched_skills") or [],
        "skill_score_evidence": skill_score.get("evidence") or [],
    }
    responsibilities = (match_result.get("responsibilities")
                        or [])  # 兼容；JD 职责放在下方单独传

    return (
        "请对以下「目标技能」逐项进行技能差距分析，每项判断为 matched / weak / missing。\n"
        "只分析给定的目标技能，不要扩展到其它字段。\n\n"
        "===== 候选人简历（仅以下内容可作为证据）=====\n"
        f"{json.dumps(resume_subset, ensure_ascii=False, indent=2)}\n\n"
        "===== 岗位 must（必备）技能 =====\n"
        f"{json.dumps(must_skills, ensure_ascii=False)}\n\n"
        "===== 岗位 preferred（优先）技能 =====\n"
        f"{json.dumps(preferred_skills, ensure_ascii=False)}\n\n"
        "===== 岗位职责摘要 =====\n"
        f"{json.dumps(responsibilities, ensure_ascii=False)}\n\n"
        "===== Match Scorer 已给出的技能结论（请保持一致，避免矛盾）=====\n"
        f"{json.dumps(match_ref, ensure_ascii=False, indent=2)}\n\n"
        "===== 输出 JSON Schema =====\n"
        '{\n'
        '  "items": [\n'
        '    {\n'
        '      "skill": "string",\n'
        '      "importance": "must / preferred",\n'
        '      "status": "matched / weak / missing",\n'
        '      "resume_evidence": ["string"],\n'
        '      "gap_reason": "string",\n'
        '      "suggestion": "string"\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "must 技能的 importance 必须为 must，preferred 技能的 importance 必须为 preferred。\n"
        "请直接输出 JSON 对象本体，不要任何额外文字或 Markdown 包装。"
    )


# ===== 4. LLM 输出清洗与解析 =====
def clean_llm_json_output(raw: str) -> str:
    """去除 ```json / ``` 包裹，截取首 '{' 到末 '}'。"""
    if not raw:
        return ""
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return t


def safe_json_parse(raw: str) -> Optional[dict]:
    """清洗 + 解析。失败返回 None 并打印错误。"""
    if not raw or not raw.strip():
        print("[ERROR] LLM 输出为空字符串")
        return None
    cleaned = clean_llm_json_output(raw)
    try:
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            print(f"[ERROR] LLM 输出不是 JSON 对象，而是 {type(obj).__name__}")
            return None
        return obj
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 解析失败：{e}")
        print(f"[ERROR] LLM 原始输出前 500 字符：\n{raw[:500]}")
        return None


# ===== 5. LLM 调用（只生成 items）=====
def _call_llm_items(
    resume_subset: dict,
    must_skills: list,
    preferred_skills: list,
    match_result: dict,
) -> Optional[list]:
    """调用 LLM 生成 items 列表。失败返回 None（交由上层规则兜底）。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] 环境变量 DASHSCOPE_API_KEY 未设置")
        return None

    user_prompt = _build_user_prompt(resume_subset, must_skills, preferred_skills, match_result)

    try:
        # 延迟导入，避免无依赖环境在 import 阶段失败
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=api_key,
            openai_api_base=BASE_URL,
            temperature=0,
        )
        response = llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        raw = getattr(response, "content", "") or ""
    except Exception as e:
        print(f"[ERROR] LLM 调用失败：{e}")
        return None

    parsed = safe_json_parse(raw)
    if parsed is None:
        return None
    items = parsed.get("items")
    if not isinstance(items, list):
        print("[ERROR] LLM 输出缺少合法 items 列表")
        return None
    return items


# ===== 6. items 校验 =====
_VALID_IMPORTANCE = {"must", "preferred"}
_VALID_STATUS = {"matched", "weak", "missing"}


def _validate_items(items: list, must_skills: list, preferred_skills: list) -> list:
    """对 LLM 返回的 items 做字段校验与规范化。

    - importance 不合法时按技能是否属于 must 列表回填；
    - status 不合法时回退为 missing；
    - resume_evidence 强制为字符串列表；
    - gap_reason / suggestion 强制为字符串；
    - skill 为空的条目丢弃。
    """
    must_lower = _lower_set(must_skills)
    valid = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        skill = _safe_str(it.get("skill"))
        if not skill:
            continue

        importance = it.get("importance")
        if importance not in _VALID_IMPORTANCE:
            importance = "must" if skill.strip().lower() in must_lower else "preferred"

        status = it.get("status")
        if status not in _VALID_STATUS:
            status = "missing"

        evidence = it.get("resume_evidence")
        if isinstance(evidence, list):
            evidence = [str(x).strip() for x in evidence if isinstance(x, (str, int, float)) and str(x).strip()]
        elif isinstance(evidence, str) and evidence.strip():
            evidence = [evidence.strip()]
        else:
            evidence = []
        # matched / weak 必须有证据；无证据则降级（避免与“显式出现才算 matched”冲突）
        if status == "matched" and not evidence:
            status = "weak"

        valid.append({
            "skill": skill,
            "importance": importance,
            "status": status,
            "resume_evidence": evidence,
            "gap_reason": _safe_str(it.get("gap_reason")),
            "suggestion": _safe_str(it.get("suggestion")),
        })
    return valid


# ===== 7. 规则兜底 items（LLM 失败时，用 Match Scorer 的结论生成简化版）=====
def _fallback_items_from_match(
    must_skills: list,
    preferred_skills: list,
    match_result: dict,
) -> list:
    """LLM 失败但 match_result.skill_score 可用时，按命中/弱匹配/缺失生成简化 items。"""
    skill_score = match_result.get("skill_score") or {}
    matched_set = _lower_set(skill_score.get("matched_skills"))
    weak_set = _lower_set(skill_score.get("weak_matched_skills"))
    missing_set = _lower_set(skill_score.get("missing_skills"))

    def _status(skill: str) -> str:
        low = skill.strip().lower()
        if low in matched_set:
            return "matched"
        if low in weak_set:
            return "weak"
        if low in missing_set:
            return "missing"
        return "missing"

    items = []
    for skill in must_skills:
        st = _status(skill)
        items.append({
            "skill": skill,
            "importance": "must",
            "status": st,
            "resume_evidence": [],
            "gap_reason": {
                "matched": "Match Scorer 判定为命中。",
                "weak": "Match Scorer 判定为弱匹配，存在可迁移基础但缺少直接证据。",
                "missing": "Match Scorer 判定为缺口，简历中暂无相关描述。",
            }[st],
            "suggestion": f"建议补充与「{skill}」直接相关的项目或实践经验，并在简历中量化体现。",
        })
    for skill in preferred_skills:
        st = _status(skill)
        items.append({
            "skill": skill,
            "importance": "preferred",
            "status": st,
            "resume_evidence": [],
            "gap_reason": "（优先项）" + {
                "matched": "Match Scorer 判定为命中。",
                "weak": "存在一定相关基础。",
                "missing": "简历中暂无相关描述。",
            }[st],
            "suggestion": f"作为加分项，可逐步补充「{skill}」相关经验以提升竞争力。",
        })
    return items


# ===== 8. Python 聚合计算 =====
def _aggregate_status(items: list) -> tuple:
    """按 status 聚合 matched / weak / missing 技能（去重保序）。"""
    matched = _dedup_keep_order([it["skill"] for it in items if it.get("status") == "matched"])
    weak = _dedup_keep_order([it["skill"] for it in items if it.get("status") == "weak"])
    missing = _dedup_keep_order([it["skill"] for it in items if it.get("status") == "missing"])
    return matched, weak, missing


def _compute_overall_risk(items: list) -> tuple:
    """只统计 must 项，计算 overall_risk 与原因。

    规则：
        must_missing >= 2 → high
        must_missing == 1 或 must_weak >= 2 → medium
        其余 → low
    """
    must_items = [it for it in items if it.get("importance") == "must"]
    must_missing = sum(1 for it in must_items if it.get("status") == "missing")
    must_weak = sum(1 for it in must_items if it.get("status") == "weak")
    must_matched = sum(1 for it in must_items if it.get("status") == "matched")

    if must_missing >= 2:
        risk = "high"
    elif must_missing == 1 or must_weak >= 2:
        risk = "medium"
    else:
        risk = "low"

    level_cn = {"high": "岗位能力缺口较明显", "medium": "存在一定能力缺口", "low": "核心能力基本匹配"}
    reason = (
        f"核心必备技能中有 {must_missing} 项缺失、{must_weak} 项弱匹配、{must_matched} 项命中，"
        f"{level_cn[risk]}。"
    )
    return risk, reason


def _compute_top_suggestion(items: list) -> str:
    """生成 top_suggestion。

    优先级：
        1. 第一个 must + missing 的 suggestion；
        2. 第一个 must + weak 的 suggestion；
        3. 第一个 preferred 中 weak/missing 的 suggestion；
        4. 默认文案。
    """
    def _first(pred) -> Optional[str]:
        for it in items:
            if pred(it):
                s = _safe_str(it.get("suggestion"))
                if s:
                    return s
        return None

    return (
        _first(lambda it: it.get("importance") == "must" and it.get("status") == "missing")
        or _first(lambda it: it.get("importance") == "must" and it.get("status") == "weak")
        or _first(lambda it: it.get("importance") == "preferred" and it.get("status") in {"weak", "missing"})
        or "当前匹配情况较好，建议正常投递，并在简历中突出相关项目证据。"
    )


# ===== 主函数 =====
def analyze_skill_gap(
    resume_profile: dict,
    jd_profile: dict,
    match_result: dict,
) -> dict:
    """技能差距分析主流程。

    LLM 负责生成 items；其余字段（聚合、风险、top_suggestion）由 Python 计算。
    任何异常都不抛出，返回带 error 的空结构或规则兜底结果。
    """
    if not isinstance(resume_profile, dict):
        resume_profile = {}
    if not isinstance(jd_profile, dict):
        jd_profile = {}
    if not isinstance(match_result, dict):
        match_result = {}

    # 1) 选目标技能
    must_skills, preferred_skills = _select_target_skills(jd_profile, match_result)

    # JD 完全没有可分析技能时，直接返回空结构
    if not must_skills and not preferred_skills:
        return _empty_result(error="JD 未提供任何可分析的技能项。")

    # 2) LLM 生成 items（失败则规则兜底）
    resume_subset = _resume_subset_for_prompt(resume_profile)
    error_msg = None
    items = _call_llm_items(resume_subset, must_skills, preferred_skills, match_result)
    if items is None:
        error_msg = "LLM 生成 items 失败，已使用 Match Scorer 结论规则兜底。"
        items = _fallback_items_from_match(must_skills, preferred_skills, match_result)
    else:
        items = _validate_items(items, must_skills, preferred_skills)
        # LLM 返回空 items 时也兜底
        if not items:
            error_msg = "LLM 未返回有效 items，已使用规则兜底。"
            items = _fallback_items_from_match(must_skills, preferred_skills, match_result)

    # 3) Python 聚合与计算
    matched_skills, weak_skills, missing_skills = _aggregate_status(items)
    overall_risk, overall_risk_reason = _compute_overall_risk(items)
    top_suggestion = _compute_top_suggestion(items)

    return {
        "items": items,
        "matched_skills": matched_skills,
        "weak_skills": weak_skills,
        "missing_skills": missing_skills,
        "overall_risk": overall_risk,
        "overall_risk_reason": overall_risk_reason,
        "top_suggestion": top_suggestion,
        "error": error_msg,
    }


# ===== CLI =====
def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """命令行入口：python skill_gap_analyzer.py resume_profile.json jd_profile.json match_result.json"""
    parser = argparse.ArgumentParser(description="Skill Gap Analyzer — 技能差距分析")
    parser.add_argument("resume_json", help="Resume Profile JSON 文件路径")
    parser.add_argument("jd_json", help="JD Profile JSON 文件路径")
    parser.add_argument("match_json", help="Match Result JSON 文件路径")
    parser.add_argument("-o", "--output", default=None, help="（可选）输出 JSON 文件路径")
    args = parser.parse_args()

    for path in (args.resume_json, args.jd_json, args.match_json):
        if not os.path.isfile(path):
            print(f"[ERROR] 文件不存在：{path}")
            sys.exit(1)

    try:
        resume_profile = _load_json_file(args.resume_json)
        jd_profile = _load_json_file(args.jd_json)
        match_result = _load_json_file(args.match_json)
    except json.JSONDecodeError as e:
        print(f"[ERROR] 输入 JSON 解析失败：{e}")
        sys.exit(1)

    if not all(isinstance(x, dict) for x in (resume_profile, jd_profile, match_result)):
        print("[ERROR] 三个输入文件顶层都必须是 JSON 对象")
        sys.exit(1)

    result = analyze_skill_gap(resume_profile, jd_profile, match_result)
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_str)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"\n[INFO] 结果已保存到：{args.output}")


if __name__ == "__main__":
    main()
