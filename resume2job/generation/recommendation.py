"""
推荐报告生成模块（Recommendation Writer）

输入：JD Profile + Match Result（+ 可选 skill_gap）
输出：面向用户的中文纯文本推荐报告

分工：
    - 【推荐岗位】/【综合匹配分】/【匹配亮点】/【风险提示】 → Python 拼接（保证格式稳定）
    - 【推荐理由】/【投递建议】                            → LLM 生成（保证语义自然，兜底文案随语气切换）

新增（V2）：
    - adjust_recommendation_tone()：根据评分降级 maybe → cautious
    - filter_unsafe_skill_claims()：避免无证据的语言类技能被强化
    - format_risk_points 增加优先级排序
    - format_evidence 修正标点重复问题
"""

import os
import re
import sys
import json
import argparse
from typing import Optional

# ===== 模型 / LLM 工具（统一走 core 层）=====
from resume2job.core.config import CHAT_MODEL as MODEL_NAME
from resume2job.core.llm import call_llm as _core_call_llm, safe_json_parse


# ===== 推荐等级映射 =====
MATCH_LEVEL_CN = {
    "recommended": "强烈推荐",
    "maybe": "建议考虑",
    "not_recommended": "暂不推荐",
}


# ===== 统一的「分数 → 档位」映射（单一事实来源）=====
# 推荐等级展示文案、LLM/兜底语气都由这里派生，保证「分数 / 等级 / 投递建议」三者一致，
# 不再出现「72 分却标低优先级」这类分数与等级矛盾的情况。
def _score_tier(final_score: int) -> str:
    if final_score >= 85:
        return "strong"
    if final_score >= 75:
        return "recommend"
    if final_score >= 65:
        return "consider"
    if final_score >= 50:
        return "low"
    return "negative"


_TIER_DISPLAY = {
    "strong": "强烈推荐",
    "recommend": "推荐投递",
    "consider": "建议考虑",
    "low": "低优先级",
    "negative": "暂不推荐",
}

_TIER_TONE = {
    "strong": "positive",
    "recommend": "positive",
    "consider": "neutral",
    "low": "cautious",
    "negative": "negative",
}


# ===== 学历相关风险类型（用于过滤）=====
_DEGREE_RISK_TYPES = {"学历门槛", "学历要求"}


# ===== 风险优先级（数字越小越靠前；未列出的统一为 99）=====
_RISK_PRIORITY = {
    # 高信号的核心能力缺口优先展示
    "机器人运动控制能力缺口": 1,
    "强化学习/模仿学习能力缺口": 2,
    "机器人仿真与ROS/SLAM能力缺口": 3,
    "技能缺口": 4,
    "方向不匹配": 5,
    "技术门槛": 6,
    "时长门槛": 7,
    "经验门槛": 8,
    "科研产出": 9,
    "竞赛经历": 10,
    "学历门槛": 11,
}


# ===== 编程语言类技能（无证据时不在 LLM Prompt 中强化）=====
_LANGUAGE_SKILLS = {
    "python", "java", "c++", "c#", "go", "golang",
    "javascript", "typescript", "ruby", "rust", "kotlin", "scala", "php",
}


# ===== 工具：安全取值 =====
def _safe_str(value, default: str = "") -> str:
    """非空字符串返回 strip 后内容；否则返回 default。"""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _safe_int(value, default: int = 0) -> int:
    """容错地把分数转为 int。"""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            try:
                return int(m.group())
            except ValueError:
                return default
    return default


# ===== 8. 推荐等级映射 =====
def map_match_level(level: str) -> str:
    """recommended/maybe/not_recommended -> 中文；其他或缺失 -> 未知。"""
    if not isinstance(level, str):
        return "未知"
    return MATCH_LEVEL_CN.get(level.strip().lower(), "未知")


# ===== 1. 语气判断（与推荐等级同源，均由 final_score 档位派生）=====
def adjust_recommendation_tone(match_result: dict) -> str:
    """根据综合分档位决定语气：positive / neutral / cautious / negative。

    与 _display_match_level 同源（都走 _score_tier），保证「等级」与「语气/投递建议」
    不会自相矛盾（例如 72 分既显示『推荐投递』又给出『暂不优先投递』的建议）。
    """
    if not isinstance(match_result, dict):
        return "neutral"
    return _TIER_TONE.get(_score_tier(_safe_int(match_result.get("final_score"))), "neutral")


# ===== 7. evidence 格式化（修正标点重复）=====
_TRAILING_PUNCT_RE = re.compile(r"[。；;.,，\s]+$")


def _strip_trailing_punct(text: str) -> str:
    """去掉一段文字末尾的中英文常见标点和空白。"""
    return _TRAILING_PUNCT_RE.sub("", text)


def format_evidence(evidence: list, max_items: int = 2) -> str:
    """将证据列表压缩成一句话：
    1) 每条 evidence 先 strip 再去尾部标点；
    2) 用 `；` 连接；
    3) 末尾补一个 `。`。
    """
    if not isinstance(evidence, list) or not evidence:
        return "暂无明确证据。"

    cleaned = []
    for x in evidence:
        if isinstance(x, str):
            s = x.strip()
        elif x is None:
            continue
        else:
            s = str(x).strip()
        if not s:
            continue
        s = _strip_trailing_punct(s)
        if not s:
            continue
        cleaned.append(s)
        if len(cleaned) >= max_items:
            break

    if not cleaned:
        return "暂无明确证据。"
    return "；".join(cleaned) + "。"


# ===== 6. 风险格式化（含排序 + 去重 + 学历过滤）=====
def _normalize_risk_item(item) -> Optional[dict]:
    """把字符串 / 对象风险点统一为 {type, description, evidence} 形式。"""
    if isinstance(item, dict):
        type_ = _safe_str(item.get("type"), default="其他") or "其他"
        description = _safe_str(item.get("description"))
        evidence = item.get("evidence")
        evidence = _safe_str(evidence) if isinstance(evidence, str) else ""
        if not description:
            return None
        return {"type": type_, "description": description, "evidence": evidence}
    if isinstance(item, str):
        desc = item.strip()
        if not desc:
            return None
        return {"type": "其他", "description": desc, "evidence": ""}
    return None


def _risk_priority(type_: str) -> int:
    """风险类型的展示优先级；未列出的归 99。"""
    return _RISK_PRIORITY.get(type_, 99)


def format_risk_points(jd_profile: dict, match_result: dict) -> list:
    """生成展示用的风险点字符串列表（带优先级排序、去重、学历过滤）。"""
    if not isinstance(jd_profile, dict):
        jd_profile = {}
    if not isinstance(match_result, dict):
        match_result = {}

    edu_score = _safe_int((match_result.get("education_score") or {}).get("score"), default=0)
    edu_satisfied = edu_score == 100

    # 优先 risk_analysis，没有则用 jd_profile.risk_points
    source = match_result.get("risk_analysis")
    if not isinstance(source, list) or not source:
        source = jd_profile.get("risk_points") if isinstance(jd_profile.get("risk_points"), list) else []

    normalized: list = []
    for item in source or []:
        n = _normalize_risk_item(item)
        if n is None:
            continue
        if edu_satisfied and n["type"] in _DEGREE_RISK_TYPES:
            continue
        normalized.append(n)

    if not normalized:
        return ["暂无明显风险。"]

    # 去重：(type, description) 完全相同的保留首条
    seen = set()
    unique = []
    for idx, r in enumerate(normalized):
        key = (r["type"], r["description"])
        if key in seen:
            continue
        seen.add(key)
        # 保留原序号便于稳定排序中"同类保持原有顺序"
        unique.append((idx, r))

    # 稳定排序：先按优先级，再按原始出现顺序
    unique.sort(key=lambda kv: (_risk_priority(kv[1]["type"]), kv[0]))

    lines = []
    for _, r in unique:
        clean_desc = _strip_trailing_punct(r["description"])
        # description 末尾再补 `。`，避免行内拼接歧义
        line_core = f"[{r['type']}] {clean_desc}。"
        if r["evidence"]:
            clean_evi = _strip_trailing_punct(r["evidence"])
            line_core = f"[{r['type']}] {clean_desc}（依据：{clean_evi}）"
        lines.append(line_core)
    return lines


# ===== 6.5 不安全技能声明过滤（仅用于构造 LLM Prompt）=====
def filter_unsafe_skill_claims(match_result: dict) -> dict:
    """返回 skill_score 的浅拷贝，去除"在 evidence 中没有明确来源的语言类技能"。
    用于构造 LLM Prompt，原 match_result 不被修改；
    报告的【匹配亮点】仍保留原始 evidence。
    """
    if not isinstance(match_result, dict):
        return {}
    skill = match_result.get("skill_score") or {}
    matched = list(skill.get("matched_skills") or [])
    evidence_list = skill.get("evidence") or []

    # 把所有 evidence 拼成大写小写不敏感的搜索文本
    evidence_text = " ".join([e for e in evidence_list if isinstance(e, str)]).lower()

    safe_matched = []
    suppressed = []
    for s in matched:
        if not isinstance(s, str):
            continue
        key = s.strip().lower()
        # 语言类技能必须显式出现在 evidence 文本中才视为"有证据"
        if key in _LANGUAGE_SKILLS:
            if key in evidence_text:
                safe_matched.append(s)
            else:
                suppressed.append(s)
        else:
            safe_matched.append(s)

    return {
        "matched_skills": safe_matched,
        "weak_matched_skills": list(skill.get("weak_matched_skills") or []),
        "suppressed_unsafe_skills": suppressed,
        "missing_skills": list(skill.get("missing_skills") or []),
        "preferred_matched_skills": list(skill.get("preferred_matched_skills") or []),
        "score": skill.get("score"),
        "evidence": list(evidence_list),
    }


# ===== 兜底文案（按 tone 切换）=====
def _join_skills(skills: list, limit: int = 4) -> str:
    """把技能列表压成「A、B、C」形式，最多 limit 个。"""
    items = [str(s).strip() for s in (skills or []) if isinstance(s, str) and s.strip()]
    return "、".join(items[:limit])


def _strengths_text(match_result: dict) -> str:
    """汇总候选人的可迁移强项：精确命中 + 弱匹配技能。"""
    skill = match_result.get("skill_score") or {}
    pool = list(skill.get("matched_skills") or []) + list(skill.get("weak_matched_skills") or [])
    # 去重保持顺序
    seen, ordered = set(), []
    for s in pool:
        if isinstance(s, str) and s.strip() and s not in seen:
            seen.add(s)
            ordered.append(s)
    return _join_skills(ordered)


def _fallback_reason(match_result: dict, tone: str) -> str:
    final = _safe_int(match_result.get("final_score"))
    dims = {
        "技能匹配": _safe_int((match_result.get("skill_score") or {}).get("score")),
        "项目相关性": _safe_int((match_result.get("project_score") or {}).get("score")),
        "学历适配": _safe_int((match_result.get("education_score") or {}).get("score")),
        "方向契合": _safe_int((match_result.get("direction_score") or {}).get("score")),
    }
    top = max(dims.items(), key=lambda kv: kv[1])
    bottom = min(dims.items(), key=lambda kv: kv[1])

    strengths = _strengths_text(match_result)
    missing = (match_result.get("skill_score") or {}).get("missing_skills") or []
    missing_text = _join_skills(missing, limit=5)
    edu_ok = _safe_int((match_result.get("education_score") or {}).get("score")) >= 100
    edu_clause = "学历满足要求，" if edu_ok else ""

    if tone == "positive":
        return (
            f"该岗位综合匹配分 {final}，整体适配度较高，"
            f"在{top[0]}（{top[1]}分）上具备明显优势。"
            "项目经历和技能结构与岗位要求基本一致，值得优先投递。"
        )
    if tone in ("cautious", "negative"):
        head = "该岗位可作为低优先级备选" if tone == "cautious" else "该岗位暂不建议优先投递"
        parts = [f"{head}（综合匹配分 {final}）。"]
        if strengths:
            parts.append(f"候选人具备{strengths}等可迁移基础，{edu_clause}")
        else:
            parts.append(f"候选人具备部分可迁移基础，{edu_clause}")
        if missing_text:
            parts.append(
                f"但岗位核心能力（{missing_text}）在当前简历中缺少直接项目证据，"
                f"短板集中在{bottom[0]}（{bottom[1]}分）。"
            )
        else:
            parts.append(f"但岗位核心方向与当前项目经历存在明显差距，短板集中在{bottom[0]}（{bottom[1]}分）。")
        return "".join(parts)
    # neutral
    return (
        f"综合匹配分 {final}。主要优势在于{top[0]}（{top[1]}分），"
        f"主要短板在于{bottom[0]}（{bottom[1]}分），可作为备选投递。"
    )


def _fallback_suggestion(match_result: dict, tone: str) -> str:
    missing = (match_result.get("skill_score") or {}).get("missing_skills") or []
    missing_text = "、".join([str(m) for m in missing[:3]]) if missing else ""

    if tone == "positive":
        base = "可以优先投递该岗位，面试中重点突出已匹配的硬技能与项目经验。"
    elif tone == "cautious":
        base = "建议优先补齐岗位核心技能和相关项目经历后再考虑投递，当前可仅作为低优先级备选。"
    elif tone == "negative":
        base = "建议优先选择与现有项目经历和技能栈更一致的岗位，暂不优先投递。"
    else:
        base = "可作为备选岗位，建议在投递前补强短板。"
    if missing_text and tone in ("cautious", "negative", "neutral"):
        base = base.rstrip("。") + f"。重点补齐：{missing_text}。"
    return base


# ===== 5. LLM 结果校验（带 tone）=====
def validate_writer_result(
    data: Optional[dict],
    match_result: dict,
    tone: str = "neutral",
) -> dict:
    """保证返回 {reason: str, suggestion: str}，缺失时用 tone 适配的兜底文案。"""
    reason = ""
    suggestion = ""
    if isinstance(data, dict):
        if isinstance(data.get("reason"), str):
            reason = data["reason"].strip()
        if isinstance(data.get("suggestion"), str):
            suggestion = data["suggestion"].strip()

    if not reason:
        reason = _fallback_reason(match_result, tone)
    if not suggestion:
        suggestion = _fallback_suggestion(match_result, tone)
    return {"reason": reason, "suggestion": suggestion}


# ===== LLM Prompt =====
SYSTEM_PROMPT_WRITER = """你是一位专业求职顾问，擅长根据岗位 JD 和匹配评分结果撰写简洁、真实、可解释的中文推荐理由和投递建议。

【输出格式】
1. 只能输出合法 JSON，禁止 Markdown 代码块标记（如 ```json），禁止解释性前后缀；
2. 输出格式必须为：
   {
     "reason": "推荐理由文字（2~3 句）",
     "suggestion": "投递建议文字（1~2 句）"
   }

【内容约束】
3. 不得编造输入中不存在的经历、技能、岗位要求或风险；
3.1 【硬性门槛忠实性·强约束】岗位的任何硬性门槛——实习时长 / 到岗月数 / 每周到岗天数 /
   学历 / 工作年限 / 论文 / 竞赛 / 城市 / 户籍 / 届别等——只能引用输入字段中明确给出的原文
   （education_requirement、experience_requirement，或风险信息 risk_analysis / jd_risk_points 中的条目）。
   输入里没有出现的门槛**一律禁止提及**；尤其严禁凭空写出「需至少 N 个月实习」「每周到岗 X 天」
   「需 N 年经验」这类输入中不存在的时间 / 年限要求。拿不准时宁可不写，也不要编造。
4. 描述"项目经历"时，只能基于 projects / project_score.evidence 等项目相关字段；
   - 求职意向（job_preferences.intentions）、技能标签或课程背景**不能**写成项目经历；
   - **严禁**因为 direction_evidence / 求职意向里出现「NLP / 自然语言处理 / CV」等字样，
     就声称候选人有「NLP 项目」「项目集中在自然语言处理」；除非 projects 中确有对应项目，否则禁止此类表述；
   - 若证据中已存在"项目集中在 NLP / CV / RAG 等"这类未必准确的描述，应转写为更保守、更贴合 projects 实际的版本，
     例如"项目主要体现多模态融合、图神经网络、多源传感器建模与 RAG/Agent 系统原型能力"。
5. 对于编程语言类技能（如 Python、Java、C++、Go），如果 evidence 中没有明确来源，
   不要在推荐理由 / 投递建议中重点强调；输入会通过 suppressed_unsafe_skills 显式列出哪些技能不可强化。
6. 撰写 cautious / negative 推荐理由时，应同时说明两面：
   - 先客观承认候选人具备的可迁移基础（取 matched_skills / weak_matched_skills，如深度学习、多模态融合、图神经网络、多源传感器建模），并指出学历是否满足；
   - 再明确指出岗位核心能力缺口（取 missing_skills / 风险信息，如机器人运动控制、轨迹规划、强化学习/模仿学习、仿真平台与真机部署），并说明当前简历缺少直接项目证据。
   投递建议应给出方向性指引：优先投递与现有技能栈一致的岗位；若要转向该方向需补齐哪些核心能力。

【语气规则（recommendation_tone）】
6. positive：推荐理由强调适配优势；投递建议建议优先投递。
7. neutral：推荐理由客观说明优势和短板；投递建议建议作为备选投递。
8. cautious：推荐理由必须体现「可作为低优先级备选，不建议优先投递」；
   投递建议强调先补齐核心短板，再考虑投递。
9. negative：推荐理由说明暂不推荐的主要原因；投递建议建议优先寻找更匹配岗位。

【长度】
- reason 控制在 2~3 句话；
- suggestion 控制在 1~2 句话。"""


def _build_user_prompt_for_writer(
    jd_profile: dict,
    match_result: dict,
    skill_gap: Optional[dict],
    tone: str,
) -> str:
    """构造 User Prompt：携带 tone、裁剪后的评分、过滤后的 skill 信息、风险等。"""
    jd_info = {
        "company": jd_profile.get("company"),
        "title": jd_profile.get("title"),
        "direction": jd_profile.get("direction"),
        "business_area": jd_profile.get("business_area"),
        "education_requirement": jd_profile.get("education_requirement"),
        "experience_requirement": jd_profile.get("experience_requirement"),
    }

    skill = match_result.get("skill_score") or {}
    project = match_result.get("project_score") or {}
    education = match_result.get("education_score") or {}
    direction = match_result.get("direction_score") or {}

    scores = {
        "final_score": match_result.get("final_score"),
        "match_level": match_result.get("match_level"),
        "skill_score": skill.get("score"),
        "project_score": project.get("score"),
        "education_score": education.get("score"),
        "direction_score": direction.get("score"),
    }

    # 过滤后的技能信息（仅用于 Prompt，不影响最终报告中【匹配亮点】展示）
    filtered_skill = filter_unsafe_skill_claims(match_result)

    evidence = {
        "matched_skills": filtered_skill.get("matched_skills") or [],
        "weak_matched_skills": filtered_skill.get("weak_matched_skills") or [],
        "suppressed_unsafe_skills": filtered_skill.get("suppressed_unsafe_skills") or [],
        "missing_skills": filtered_skill.get("missing_skills") or [],
        "skill_evidence": skill.get("evidence") or [],
        "project_evidence": project.get("evidence") or [],
        "education_evidence": education.get("evidence") or [],
        "direction_evidence": direction.get("evidence") or [],
    }

    risks = {
        "risk_analysis": match_result.get("risk_analysis") or [],
        "jd_risk_points": jd_profile.get("risk_points") or [],
    }

    skill_gap_str = (
        json.dumps(skill_gap, ensure_ascii=False, indent=2)
        if isinstance(skill_gap, dict) and skill_gap
        else "暂无额外技能差距分析。"
    )

    return (
        "请根据以下信息生成简洁的中文推荐理由（reason）和投递建议（suggestion）。\n\n"
        f"===== 语气标签 =====\nrecommendation_tone = {tone}\n\n"
        "===== 岗位基本信息 =====\n"
        f"{json.dumps(jd_info, ensure_ascii=False, indent=2)}\n\n"
        "===== 匹配评分 =====\n"
        f"{json.dumps(scores, ensure_ascii=False, indent=2)}\n\n"
        "===== 匹配证据（filtered）=====\n"
        f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
        "===== 风险信息 =====\n"
        f"{json.dumps(risks, ensure_ascii=False, indent=2)}\n\n"
        "===== skill_gap =====\n"
        f"{skill_gap_str}\n\n"
        "===== 输出 JSON Schema =====\n"
        '{\n  "reason": "2~3 句话推荐理由",\n  "suggestion": "1~2 句话投递建议"\n}\n\n'
        "请直接输出 JSON 对象本体，不要任何额外文字或 Markdown 包装。\n"
        "再次提醒：suppressed_unsafe_skills 列出的技能不要在 reason / suggestion 中强化；"
        "描述项目经历时只能引用 projects / project_evidence 相关内容，不要把 NLP/CV 等求职意向当作项目方向；"
        "岗位硬性门槛（实习时长 / 经验年限 / 学历 / 论文 / 竞赛 / 城市等）只能引用上面的"
        "岗位基本信息与风险信息原文，输入中没有的门槛绝对不要写。"
    )


# ===== 2. LLM 生成函数（带 tone）=====
def call_llm_writer(
    jd_profile: dict,
    match_result: dict,
    skill_gap: Optional[dict] = None,
) -> dict:
    """调用 LLM 生成 reason 与 suggestion；任何失败都退回规则兜底。"""
    tone = adjust_recommendation_tone(match_result)
    user_prompt = _build_user_prompt_for_writer(jd_profile, match_result, skill_gap, tone)

    try:
        # 写作任务温度略高（0.3），保持语义自然
        raw = _core_call_llm(SYSTEM_PROMPT_WRITER, user_prompt, model=MODEL_NAME, temperature=0.3)
    except Exception as e:
        print(f"[ERROR] LLM 调用失败：{e}")
        return validate_writer_result(None, match_result, tone=tone)

    parsed = safe_json_parse(raw)
    return validate_writer_result(parsed, match_result, tone=tone)


# ===== 推荐等级展示（统一由 final_score 档位决定）=====
def _display_match_level(match_result: dict) -> str:
    """推荐等级中文展示，直接由综合分档位映射（强烈推荐 / 推荐投递 / 可考虑 / 谨慎投递 / 暂不推荐）。

    取代旧的「maybe + cautious 追加（低优先级）」逻辑，确保同一分数始终对应同一等级，
    分数与等级不再矛盾。
    """
    if not isinstance(match_result, dict):
        return "未知"
    return _TIER_DISPLAY.get(_score_tier(_safe_int(match_result.get("final_score"))), "未知")


# ===== 技能差距分析展示 =====
_SKILL_GAP_RISK_CN = {"high": "高", "medium": "中", "low": "低"}
_SKILL_GAP_STATUS_CN = {"matched": "已具备", "weak": "弱匹配", "missing": "缺口"}
_SKILL_GAP_IMPORTANCE_CN = {"must": "必备", "preferred": "优先"}
# items 展示排序：缺口优先，其次弱匹配，最后已具备
_SKILL_GAP_STATUS_ORDER = {"missing": 0, "weak": 1, "matched": 2}


def format_skill_gap(skill_gap: Optional[dict]) -> list:
    """生成【技能差距分析】展示行列表。

    展示：能力缺口风险等级 + 原因 → 逐项技能（缺口优先）→ 重点建议。
    """
    if not isinstance(skill_gap, dict) or not skill_gap:
        return ["暂无技能差距分析。"]

    lines = []
    risk = skill_gap.get("overall_risk")
    risk_cn = _SKILL_GAP_RISK_CN.get(risk, "未知")
    reason = _safe_str(skill_gap.get("overall_risk_reason"))
    header = f"能力缺口风险：{risk_cn}"
    if reason:
        header += f"。{_strip_trailing_punct(reason)}。"
    lines.append(header)

    items = [it for it in (skill_gap.get("items") or []) if isinstance(it, dict)]
    items_sorted = sorted(
        items, key=lambda it: _SKILL_GAP_STATUS_ORDER.get(it.get("status"), 3)
    )
    for it in items_sorted[:8]:
        skill_name = _safe_str(it.get("skill"))
        if not skill_name:
            continue
        st = _SKILL_GAP_STATUS_CN.get(it.get("status"), "未知")
        imp = _SKILL_GAP_IMPORTANCE_CN.get(it.get("importance"), "")
        tag = f"{imp}/{st}" if imp else st
        gap_reason = _safe_str(it.get("gap_reason"))
        seg = f"[{tag}] {skill_name}"
        if gap_reason:
            seg += f"：{_strip_trailing_punct(gap_reason)}。"
        lines.append(seg)

    top = _safe_str(skill_gap.get("top_suggestion"))
    if top:
        lines.append(f"重点建议：{_strip_trailing_punct(top)}。")
    return lines


# ===== 1. 报告组装（内部）=====
def _compose_report(
    jd_profile: dict,
    match_result: dict,
    skill_gap: Optional[dict],
    include_skill_gap: bool,
) -> str:
    """组装纯文本报告。include_skill_gap=True 时额外渲染【技能差距分析】段落。"""
    if not isinstance(jd_profile, dict):
        jd_profile = {}
    if not isinstance(match_result, dict):
        match_result = {}

    # 1) 头部信息
    company = _safe_str(jd_profile.get("company"), default="未知公司")
    title = _safe_str(jd_profile.get("title"), default="未知岗位")
    final = _safe_int(match_result.get("final_score"))
    level_display = _display_match_level(match_result)

    # 2) 各维度分数与证据
    skill = match_result.get("skill_score") or {}
    project = match_result.get("project_score") or {}
    education = match_result.get("education_score") or {}
    direction = match_result.get("direction_score") or {}

    skill_score = _safe_int(skill.get("score"))
    project_score = _safe_int(project.get("score"))
    education_score = _safe_int(education.get("score"))
    direction_score = _safe_int(direction.get("score"))

    skill_evi = format_evidence(skill.get("evidence") or [])
    project_evi = format_evidence(project.get("evidence") or [])
    education_evi = format_evidence(education.get("evidence") or [])
    direction_evi = format_evidence(direction.get("evidence") or [])

    # 3) 风险点
    risk_lines = format_risk_points(jd_profile, match_result)

    # 4) 推荐理由 / 投递建议（LLM 生成 + 兜底）；skill_gap 一并喂给 LLM 增强证据
    writer_out = call_llm_writer(jd_profile, match_result, skill_gap)
    reason = writer_out["reason"]
    suggestion = writer_out["suggestion"]

    # 5) Python 拼接固定模板
    lines = []
    lines.append(f"【推荐岗位】{company} - {title}")
    lines.append(f"【综合匹配分】{final} 分 / 推荐等级：{level_display}")
    lines.append("")
    lines.append("【推荐理由】")
    lines.append(reason)
    lines.append("")
    lines.append("【匹配亮点】")
    lines.append(f"- 技能匹配（{skill_score}分）：{skill_evi}")
    lines.append(f"- 项目相关性（{project_score}分）：{project_evi}")
    lines.append(f"- 学历适配（{education_score}分）：{education_evi}")
    lines.append(f"- 方向契合（{direction_score}分）：{direction_evi}")

    # 5.5) 技能差距分析（仅 generate_full_report 渲染）
    if include_skill_gap:
        lines.append("")
        lines.append("【技能差距分析】")
        for gl in format_skill_gap(skill_gap):
            if gl.strip() == "暂无技能差距分析。":
                lines.append(gl)
            else:
                lines.append(f"- {gl}")

    lines.append("")
    lines.append("【风险提示】")
    for rl in risk_lines:
        if rl.strip() == "暂无明显风险。":
            lines.append(rl)
        else:
            lines.append(f"- {rl}")
    lines.append("")
    lines.append("【投递建议】")
    lines.append(suggestion)

    return "\n".join(lines)


def generate_recommendation(
    jd_profile: dict,
    match_result: dict,
    skill_gap: Optional[dict] = None,
) -> str:
    """生成纯文本推荐报告（旧版，不渲染技能差距分析段落；保留用于兼容与兜底）。"""
    return _compose_report(jd_profile, match_result, skill_gap, include_skill_gap=False)


def generate_full_report(
    jd_profile: dict,
    match_result: dict,
    skill_gap: Optional[dict] = None,
) -> str:
    """生成 Stage 4 完整报告：在推荐报告基础上额外渲染【技能差距分析】段落。"""
    return _compose_report(jd_profile, match_result, skill_gap, include_skill_gap=True)


# ===== CLI =====
def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """命令行入口：python recommendation_writer.py jd_profile.json match_result.json"""
    parser = argparse.ArgumentParser(description="Recommendation Writer — 生成中文推荐报告")
    parser.add_argument("jd_json", help="JD Profile JSON 文件路径")
    parser.add_argument("match_json", help="Match Result JSON 文件路径")
    parser.add_argument(
        "--skill-gap",
        help="（可选）Skill Gap JSON 文件路径",
        default=None,
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 txt 文件路径；默认保存到当前目录下 recommendation_<jd>__<match>.txt",
        default=None,
    )
    args = parser.parse_args()

    for path in (args.jd_json, args.match_json):
        if not os.path.isfile(path):
            print(f"[ERROR] 文件不存在：{path}")
            sys.exit(1)

    try:
        jd_profile = _load_json_file(args.jd_json)
        match_result = _load_json_file(args.match_json)
        skill_gap = _load_json_file(args.skill_gap) if args.skill_gap else None
    except json.JSONDecodeError as e:
        print(f"[ERROR] 输入 JSON 解析失败：{e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[ERROR] 文件不存在：{e}")
        sys.exit(1)

    if not isinstance(jd_profile, dict) or not isinstance(match_result, dict):
        print("[ERROR] jd_profile / match_result 顶层必须是 JSON 对象")
        sys.exit(1)

    report = generate_recommendation(jd_profile, match_result, skill_gap)
    print(report)

    if args.output:
        out_path = args.output
    else:
        j_base = os.path.splitext(os.path.basename(args.jd_json))[0]
        m_base = os.path.splitext(os.path.basename(args.match_json))[0]
        out_path = os.path.join(os.getcwd(), f"recommendation_{j_base}__{m_base}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[INFO] 报告已保存到：{out_path}")


if __name__ == "__main__":
    main()
