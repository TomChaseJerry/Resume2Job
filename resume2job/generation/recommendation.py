"""
推荐报告生成模块（Recommendation Writer）+ 技能差距视图（原 scoring/skill_gap.py 已并入本模块）。

输入：JD Profile + Match Result；输出：面向用户的中文纯文本推荐报告 + skill_gap dict。

分工（与项目「LLM 负责语义、Python 负责确定性编排」一致）：
    - Python 拼接（格式稳定）：【推荐岗位】/【综合匹配分】(两层 match_score + rank_score)/
      【匹配亮点】/【技能差距分析】/【风险提示】；
    - **一次合并 LLM 调用**（generate_report_and_gap → call_llm_report_and_gap）产出
      {技能差距 items(仅叙述) + 推荐理由 reason + 投递建议 suggestion}，把原本多次调用降为 1 次。

技能差距视图（见下方「技能差距视图」一节）：**status 由规则唯一决定**——取 match_scorer 的
skill_status（同义词/上下位/弱匹配/可替代组全套规则），LLM 只补 gap_reason/suggestion/evidence 叙述，
保证 skill_gap 展示与 skill_score 完全一致；learning_plan / interview 消费同一个 skill_gap dict。

换一批/重排不重调 LLM：recompose_report 用缓存的 reason/suggestion + 当前 match_score 纯 Python 重组。
LLM 不可用处处兜底（rule_based_skill_gap / generate_full_report 走规则文案）。
"""

import os
import re
import sys
import json
import argparse
from typing import Optional, Any

# ===== 模型 / LLM 工具（统一走 core 层）=====
from resume2job.core.config import CHAT_MODEL as MODEL_NAME
from resume2job.core.llm import call_llm as _core_call_llm, safe_json_parse
# 技能差距视图所需的规则层工具：status 由 match_scorer 的 skill_status 唯一权威决定，
# 本模块（见下方「技能差距视图」一节）只按归一名查 status + 组装展示视图，不做任何匹配判定。
from resume2job.scoring.match_scorer import _normalize_skill as _norm_skill
from resume2job.parsing.jd_parser import split_compound_skill, job_cities


# ===== 统一的「分数 → 档位」映射（单一事实来源）=====
# 推荐等级展示文案、LLM/兜底语气都由这里派生，保证「分数 / 等级 / 投递建议」三者一致。
# 入参为 rank_score（最终推荐分）。规范 3 档：≥75 推荐 / 50–74 可酌情 / <50 暂不，
# 这里在其上细分展示档位（strong/recommend/consider/low/negative），等级阈值与规范一致。
def _score_tier(rank_score: int) -> str:
    if rank_score >= 85:
        return "strong"
    if rank_score >= 75:
        return "recommend"
    if rank_score >= 65:
        return "consider"
    if rank_score >= 50:
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


# ===== 1. 语气判断（与推荐等级同源，均由 rank_score 档位派生）=====
def adjust_recommendation_tone(match_result: dict) -> str:
    """根据综合分档位决定语气：positive / neutral / cautious / negative。

    与 _display_match_level 同源（都走 _score_tier），保证「等级」与「语气/投递建议」
    不会自相矛盾（例如 72 分既显示『推荐投递』又给出『暂不优先投递』的建议）。
    """
    if not isinstance(match_result, dict):
        return "neutral"
    return _TIER_TONE.get(_score_tier(_safe_int(match_result.get("rank_score"))), "neutral")


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

    edu_satisfied = (match_result.get("education_gate") or {}).get("gate") == "satisfied"

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
    final = _safe_int(match_result.get("rank_score"))
    dims = {
        "技能匹配": _safe_int((match_result.get("skill_score") or {}).get("score")),
        "项目相关性": _safe_int((match_result.get("project_score") or {}).get("score")),
    }
    top = max(dims.items(), key=lambda kv: kv[1])
    bottom = min(dims.items(), key=lambda kv: kv[1])

    strengths = _strengths_text(match_result)
    missing = (match_result.get("skill_score") or {}).get("missing_skills") or []
    missing_text = _join_skills(missing, limit=5)
    edu_ok = (match_result.get("education_gate") or {}).get("gate") == "satisfied"
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


# ===== LLM Prompt（合并调用：一次产出 技能差距 items + 推荐理由 + 投递建议）=====
SYSTEM_PROMPT_WRITER = """你是一位专业求职顾问，根据岗位 JD、匹配评分与简历证据，一次性产出：
（A）逐技能的技能差距分析 items；（B）推荐理由 reason；（C）投递建议 suggestion。

【输出格式】
1. 只能输出合法 JSON，禁止 Markdown 代码块标记（如 ```json），禁止解释性前后缀；
2. 输出格式必须为：
   {
     "items": [
       {"skill": "string", "resume_evidence": ["string"], "gap_reason": "string", "suggestion": "string"}
     ],
     "reason": "推荐理由文字（2~3 句）",
     "suggestion": "投递建议文字（1~2 句）"
   }

【技能差距 items —— 你只负责「叙述」，不负责「判定」】
- 每个目标技能是否命中 / 弱匹配 / 缺失（status）**已由系统规则判定**，在输入「目标技能及判定状态」里给出，
  你**不得改判、不要输出 status 字段**；只为每个给定技能写：
  * resume_evidence：支撑该判定的简历原文证据（来自 projects / experiences / 技能栏等），没有则空列表 []；
  * gap_reason：针对该技能当前状态的简短原因（matched→指出在哪体现；weak→有何可迁移基础但缺直接实践；missing→简历完全没有）；
  * suggestion：具体可执行的提升建议。
- 只覆盖输入给定的目标技能，不要新增或漏掉；resume_evidence 必须来自简历内容，禁止编造；
- 不要因候选人是计算机专业就臆断掌握 Python/PyTorch/C++/ROS 等；证据须在简历显式出现。

【内容约束】
3. 不得编造输入中不存在的经历、技能、岗位要求或风险；
3.1 【硬性门槛忠实性·强约束】岗位的任何硬性门槛——实习时长 / 到岗月数 / 每周到岗天数 /
   学历 / 工作年限 / 论文 / 竞赛 / 城市 / 户籍 / 届别等——只能引用输入字段中明确给出的原文
   （education_requirement、experience_requirement，或风险信息 risk_analysis / jd_risk_points 中的条目）。
   输入里没有出现的门槛**一律禁止提及**；尤其严禁凭空写出「需至少 N 个月实习」「每周到岗 X 天」
   「需 N 年经验」这类输入中不存在的时间 / 年限要求。拿不准时宁可不写，也不要编造。
4. 描述"项目经历"时，只能基于 projects / project_score.evidence 等项目相关字段；
   - 求职意向（job_preferences.intentions）、技能标签或课程背景**不能**写成项目经历；
   - **严禁**因为求职意向里出现「NLP / 自然语言处理 / CV」等字样，
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
    resume_profile: dict,
    jd_profile: dict,
    match_result: dict,
    tone: str,
) -> str:
    """构造合并 User Prompt：tone + 评分 + 过滤后 skill 证据 + 目标技能 + 简历证据 + 风险，
    一次产出 items（技能差距）+ reason + suggestion。"""
    jd_info = {
        "company": jd_profile.get("company"),
        "title": jd_profile.get("title"),
        "direction": jd_profile.get("direction"),
        "business_area": jd_profile.get("business_area"),
        "education_requirement": jd_profile.get("education_requirement"),
        "experience_requirement": jd_profile.get("experience_requirement"),
        "responsibilities": (jd_profile.get("responsibilities") or [])[:5],
    }

    skill = match_result.get("skill_score") or {}
    project = match_result.get("project_score") or {}
    direction_bonus_info = match_result.get("direction_bonus_info") or {}

    scores = {
        "match_score": match_result.get("match_score"),       # 基础适配分（技能+项目）
        "direction_bonus": match_result.get("direction_bonus"),
        "commute_bonus": match_result.get("commute_bonus"),
        "rank_score": match_result.get("rank_score"),          # 最终推荐分
        "match_level": match_result.get("match_level"),
        "skill_score": skill.get("score"),
        "project_score": project.get("score"),
        "direction_preference_hit": direction_bonus_info.get("reason"),
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
    }

    risks = {
        "risk_analysis": match_result.get("risk_analysis") or [],
        "jd_risk_points": jd_profile.get("risk_points") or [],
    }

    # 技能差距分析所需：目标技能 + **每个技能的规则判定状态**（status 权威，LLM 只叙述）+ 简历证据子集
    must_skills, preferred_skills = select_target_skills(jd_profile, match_result)
    skill_status = (match_result.get("skill_score") or {}).get("skill_status") or {}
    target_status = (
        [{"skill": s, "importance": "must",
          "status": _status_for(s, skill_status, match_result)} for s in must_skills]
        + [{"skill": s, "importance": "preferred",
            "status": _status_for(s, skill_status, match_result)} for s in preferred_skills]
    )
    resume_subset = resume_evidence_subset(resume_profile or {})

    return (
        "请根据以下信息，一次性输出：技能差距 items（仅叙述）+ 推荐理由 reason + 投递建议 suggestion。\n\n"
        f"===== 语气标签 =====\nrecommendation_tone = {tone}\n\n"
        "===== 岗位基本信息 =====\n"
        f"{json.dumps(jd_info, ensure_ascii=False, indent=2)}\n\n"
        "===== 匹配评分 =====\n"
        f"{json.dumps(scores, ensure_ascii=False, indent=2)}\n\n"
        "===== Match Scorer 技能结论（reason/suggestion 参考）=====\n"
        f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
        "===== 风险信息 =====\n"
        f"{json.dumps(risks, ensure_ascii=False, indent=2)}\n\n"
        "===== 目标技能及其判定状态（status 已定，你只为每个技能补 evidence/gap_reason/suggestion）=====\n"
        f"{json.dumps(target_status, ensure_ascii=False, indent=2)}\n\n"
        "===== 候选人简历（items 的 resume_evidence 只能来自这里）=====\n"
        f"{json.dumps(resume_subset, ensure_ascii=False, indent=2)}\n\n"
        "===== 输出 JSON Schema =====\n"
        '{\n  "items": [{"skill":"","resume_evidence":[],"gap_reason":"","suggestion":""}],\n'
        '  "reason": "2~3 句话推荐理由",\n  "suggestion": "1~2 句话投递建议"\n}\n\n'
        "请直接输出 JSON 对象本体，不要任何额外文字或 Markdown 包装。\n"
        "再次提醒：items 覆盖上面每个目标技能、**不要输出 status**（系统已定）；suppressed_unsafe_skills 列出的技能不要在 reason / suggestion 中强化；"
        "描述项目经历只能引用 projects / project_evidence，不要把 NLP/CV 等求职意向当作项目方向；"
        "岗位硬性门槛只能引用上面原文，输入中没有的门槛绝对不要写。"
    )


# ===== 2. 合并 LLM 调用：一次产出 items + reason + suggestion =====
def call_llm_report_and_gap(
    resume_profile: dict,
    jd_profile: dict,
    match_result: dict,
) -> dict:
    """一次 LLM 调用产出 {items, reason, suggestion}；任何失败都退回规则兜底（items=[]）。"""
    tone = adjust_recommendation_tone(match_result)
    user_prompt = _build_user_prompt_for_writer(resume_profile, jd_profile, match_result, tone)

    try:
        raw = _core_call_llm(SYSTEM_PROMPT_WRITER, user_prompt, model=MODEL_NAME, temperature=0.3)
    except Exception as e:
        print(f"[ERROR] LLM 调用失败：{e}")
        out = validate_writer_result(None, match_result, tone=tone)
        return {"items": [], **out}

    parsed = safe_json_parse(raw)
    out = validate_writer_result(parsed, match_result, tone=tone)
    items = parsed.get("items") if isinstance(parsed, dict) and isinstance(parsed.get("items"), list) else []
    return {"items": items, **out}


# ===== 推荐等级展示（统一由 rank_score 档位决定）=====
def _display_match_level(match_result: dict) -> str:
    """推荐等级中文展示，直接由综合分档位映射（强烈推荐 / 推荐投递 / 可考虑 / 谨慎投递 / 暂不推荐）。

    同一分数始终对应同一等级，保证分数与等级一致。
    """
    if not isinstance(match_result, dict):
        return "未知"
    return _TIER_DISPLAY.get(_score_tier(_safe_int(match_result.get("rank_score"))), "未知")


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


# ===== 1. 报告组装（内部，纯 Python，不调 LLM）=====
def _compose_report(
    jd_profile: dict,
    match_result: dict,
    skill_gap: Optional[dict],
    reason: str,
    suggestion: str,
    include_skill_gap: bool = True,
) -> str:
    """组装纯文本报告（reason / suggestion 由合并调用产出后传入，本函数不调 LLM）。
    include_skill_gap=True 时额外渲染【技能差距分析】段落。"""
    if not isinstance(jd_profile, dict):
        jd_profile = {}
    if not isinstance(match_result, dict):
        match_result = {}

    # 1) 头部信息：基础适配分 / 偏好加分 / 最终推荐分（两层评分）
    company = _safe_str(jd_profile.get("company"), default="未知公司")
    title = _safe_str(jd_profile.get("title"), default="未知岗位")
    match_score = _safe_int(match_result.get("match_score"))
    direction_bonus = _safe_int(match_result.get("direction_bonus"))
    commute_bonus = _safe_int(match_result.get("commute_bonus"))
    rank_score = _safe_int(match_result.get("rank_score"))
    level_display = _display_match_level(match_result)

    # 2) 各维度分数与证据（仅技能 / 项目两维）
    skill = match_result.get("skill_score") or {}
    project = match_result.get("project_score") or {}
    direction_bonus_info = match_result.get("direction_bonus_info") or {}

    skill_score = _safe_int(skill.get("score"))
    project_score = _safe_int(project.get("score"))

    skill_evi = format_evidence(skill.get("evidence") or [])
    project_evi = format_evidence(project.get("evidence") or [])

    # 3) 风险点
    risk_lines = format_risk_points(jd_profile, match_result)

    # 4) 推荐理由 / 投递建议由合并调用产出后传入（此处不再调 LLM）
    reason = _safe_str(reason) or _fallback_reason(match_result, adjust_recommendation_tone(match_result))
    suggestion = _safe_str(suggestion) or _fallback_suggestion(match_result, adjust_recommendation_tone(match_result))

    # 5) Python 拼接固定模板
    lines = []
    # 地点未明确（JD 无城市，召回时按 city_status=unknown 进池）：在岗位标题处提示投递前确认
    loc_note = "（地点未明确，建议投递前确认）" if not job_cities(jd_profile) else ""
    lines.append(f"【推荐岗位】{company} - {title}{loc_note}")
    lines.append(
        f"【匹配分】基础适配分 {match_score} ＋ 偏好加分 {direction_bonus + commute_bonus}"
        f"（方向 {direction_bonus} / 通勤 {commute_bonus}）＝ 最终推荐分 {rank_score} 分"
        f" / 推荐等级：{level_display}"
    )
    lines.append("")
    lines.append("【推荐理由】")
    lines.append(reason)
    lines.append("")
    lines.append("【匹配亮点】")
    lines.append(f"- 技能匹配（{skill_score}分）：{skill_evi}")
    lines.append(f"- 项目相关性（{project_score}分）：{project_evi}")
    # 偏好命中说明（方向偏好；通勤命中由 enhancement 追加的【通勤】段体现）
    pref_reason = _safe_str(direction_bonus_info.get("reason"))
    if direction_bonus > 0 and pref_reason:
        lines.append(f"- 偏好命中（方向 +{direction_bonus}）：{pref_reason}")

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


# ===========================================================================
# 技能差距视图（Skill Gap View）—— 原 scoring/skill_gap.py，2026-06-21 并入报告层
#
# 定位：**不做任何匹配判定**。每个技能的 status（命中/弱匹配/缺口）唯一权威是
# match_scorer 的 skill_status（含同义词/上下位/弱匹配/可替代组全套规则）；本节只做
# 「选展示技能 → 按规则 status 组装 items → 配 LLM 叙述（缺则规则模板）→ 汇总风险」，
# 产出供报告渲染、learning_plan、interview 消费的 skill_gap dict。无 LLM、纯 Python。
# ===========================================================================

# ----- 目标技能选择上限 -----
MAX_MUST = 7        # must 技能最多分析数量
MAX_PREFERRED = 3   # preferred 技能最多分析数量
MAX_ITEMS = 10      # items 总条数上限

_VALID_STATUS = {"matched", "weak", "missing"}


def _sg_empty_result(error: Optional[str] = None) -> dict:
    """返回符合 Schema 的空 skill_gap 结构（无目标技能/异常兜底用）。"""
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


def _sg_dedup_keep_order(items: list) -> list:
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


def _sg_lower_set(items: list) -> set:
    """转成小写去空格集合，便于不区分大小写比对。"""
    return {x.strip().lower() for x in (items or []) if isinstance(x, str) and x.strip()}


def _sg_atomize(items: list) -> list:
    """与 match_scorer 同一套原子化（split_compound_skill），保证目标技能 = skill_status 的原子键。
    对已原子化的输入幂等。"""
    out = []
    for it in items or []:
        if isinstance(it, str) and it.strip():
            out.extend(split_compound_skill(it))
    return _sg_dedup_keep_order(out)


def select_target_skills(jd_profile: dict, match_result: dict) -> tuple:
    """选择 5~10 个最关键的能力项。返回 (must_selected, preferred_selected)。

    规则：
      - must = hard_skills + tools_or_frameworks（hard_skills 为空时退化到 domain_keywords）；
      - preferred = preferred_skills，最多 3 个；
      - 全部**原子化**（与 match_scorer 一致），使 _status_for 能在 skill_status 里命中规则权威 status；
      - must 内部按「missing > weak > matched > 其它」的优先级稳定排序，取前 MAX_MUST。
    """
    must = _sg_atomize(
        list(jd_profile.get("hard_skills") or []) +
        list(jd_profile.get("tools_or_frameworks") or [])
    )
    if not must:
        must = _sg_atomize(jd_profile.get("domain_keywords") or [])

    preferred = _sg_atomize(jd_profile.get("preferred_skills") or [])

    skill_score = match_result.get("skill_score") or {}
    missing_set = _sg_lower_set(skill_score.get("missing_skills"))
    weak_set = _sg_lower_set(skill_score.get("weak_matched_skills"))
    matched_set = _sg_lower_set(skill_score.get("matched_skills"))

    def _rank(s: str) -> int:
        low = s.strip().lower()
        if low in missing_set:
            return 0
        if low in weak_set:
            return 1
        if low in matched_set:
            return 2
        return 3

    must_sorted = sorted(must, key=_rank)  # 稳定排序，同级保持原顺序
    must_selected = must_sorted[:MAX_MUST]
    preferred_selected = preferred[:MAX_PREFERRED]

    if len(must_selected) + len(preferred_selected) > MAX_ITEMS:
        preferred_selected = preferred_selected[: max(0, MAX_ITEMS - len(must_selected))]

    return must_selected, preferred_selected


def resume_evidence_subset(resume_profile: dict) -> dict:
    """裁剪简历，只保留可作为证据的字段（供合并调用的 prompt 用，节省 token）。"""
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


def _status_for(skill: str, skill_status: dict, match_result: dict) -> str:
    """取该技能的 status——**唯一权威是 match_scorer 的 skill_status**（按归一名查）。

    老结构无 skill_status 时回退到 skill_score 的 matched/weak/missing 集合（向后兼容）。
    """
    n = _norm_skill(skill)
    if n and isinstance(skill_status, dict) and n in skill_status:
        st = skill_status[n].get("status")
        if st in _VALID_STATUS:
            return st
    # 回退（老 match_result 无 skill_status 时）：两侧都归一（_norm_skill），避免 pytorch/torch 等同义词漏判
    ss = match_result.get("skill_score") or {}
    key = n or (skill or "").strip().lower()

    def _norm_set(xs):
        return {_norm_skill(x) or str(x).strip().lower() for x in (xs or []) if isinstance(x, str) and x.strip()}

    if key in _norm_set(ss.get("matched_skills")):
        return "matched"
    if key in _norm_set(ss.get("weak_matched_skills")):
        return "weak"
    return "missing"


def _sg_default_gap_reason(status: str, importance: str) -> str:
    base = {
        "matched": "Match Scorer 判定为命中。",
        "weak": "Match Scorer 判定为弱匹配，存在可迁移基础但缺少直接项目证据。",
        "missing": "Match Scorer 判定为缺口，简历中暂无相关描述。",
    }.get(status, "")
    return ("（优先项）" + base) if importance == "preferred" else base


def _sg_default_suggestion(skill: str, importance: str) -> str:
    if importance == "preferred":
        return f"作为加分项，可逐步补充「{skill}」相关经验以提升竞争力。"
    return f"建议补充与「{skill}」直接相关的项目或实践经验，并在简历中量化体现。"


def _sg_norm_evidence(value: Any) -> list:
    if isinstance(value, list):
        return [str(x).strip() for x in value if isinstance(x, (str, int, float)) and str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _sg_index_narrative(narrative_items: list) -> dict:
    """把 LLM narrative items 按技能名（小写）索引：{skill_lower: {gap_reason, suggestion, resume_evidence}}。"""
    out = {}
    for it in narrative_items or []:
        if isinstance(it, dict) and isinstance(it.get("skill"), str) and it["skill"].strip():
            out[it["skill"].strip().lower()] = it
    return out


def _sg_aggregate_status(items: list) -> tuple:
    """按 status 聚合 matched / weak / missing 技能（去重保序）。"""
    matched = _sg_dedup_keep_order([it["skill"] for it in items if it.get("status") == "matched"])
    weak = _sg_dedup_keep_order([it["skill"] for it in items if it.get("status") == "weak"])
    missing = _sg_dedup_keep_order([it["skill"] for it in items if it.get("status") == "missing"])
    return matched, weak, missing


def _sg_compute_overall_risk(items: list) -> tuple:
    """只统计 must 项，计算 overall_risk 与原因。

    规则：must_missing >= 2 → high；must_missing == 1 或 must_weak >= 2 → medium；其余 → low。
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


def _sg_compute_top_suggestion(items: list) -> str:
    """生成 top_suggestion：第一个 must+missing > must+weak > preferred 弱/缺 > 默认。"""
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


def _sg_assemble(items: list, error: Optional[str] = None) -> dict:
    """把 items 聚合为完整 skill_gap 结构。"""
    matched, weak, missing = _sg_aggregate_status(items)
    overall_risk, overall_risk_reason = _sg_compute_overall_risk(items)
    return {
        "items": items,
        "matched_skills": matched,
        "weak_skills": weak,
        "missing_skills": missing,
        "overall_risk": overall_risk,
        "overall_risk_reason": overall_risk_reason,
        "top_suggestion": _sg_compute_top_suggestion(items),
        "error": error,
    }


def _sg_assemble_items(jd_profile: dict, match_result: dict, narrative_items: list) -> Optional[list]:
    """对每个目标技能：**status 取自规则权威 skill_status**，叙述（reason/suggestion/evidence）取自
    narrative_items（LLM），缺则规则模板兜底。返回 items 列表；无目标技能返回 None。"""
    must_skills, preferred_skills = select_target_skills(jd_profile, match_result)
    if not must_skills and not preferred_skills:
        return None
    skill_status = (match_result.get("skill_score") or {}).get("skill_status") or {}
    narr = _sg_index_narrative(narrative_items)

    items = []
    for skill, importance in ([(s, "must") for s in must_skills]
                              + [(s, "preferred") for s in preferred_skills]):
        status = _status_for(skill, skill_status, match_result)        # ← 规则权威，LLM 不可改
        nit = narr.get((skill or "").strip().lower()) or {}
        gap_reason = _safe_str(nit.get("gap_reason")) or _sg_default_gap_reason(status, importance)
        suggestion = _safe_str(nit.get("suggestion")) or _sg_default_suggestion(skill, importance)
        items.append({
            "skill": skill,
            "importance": importance,
            "status": status,
            "resume_evidence": _sg_norm_evidence(nit.get("resume_evidence")),
            "gap_reason": gap_reason,
            "suggestion": suggestion,
        })
    return items


def build_skill_gap(narrative_items: list, jd_profile: dict, match_result: dict) -> dict:
    """构建 skill_gap：status 由规则（match_scorer.skill_status）决定，叙述取自 LLM narrative_items。

    narrative_items 仅提供 gap_reason / suggestion / resume_evidence，**不参与 status 判定**——
    保证 skill_gap 展示与 skill_score 完全一致。narrative 缺失时叙述走规则模板（仍有效）。
    """
    if not isinstance(jd_profile, dict):
        jd_profile = {}
    if not isinstance(match_result, dict):
        match_result = {}
    items = _sg_assemble_items(jd_profile, match_result, narrative_items or [])
    if items is None:
        return _sg_empty_result(error="JD 未提供任何可分析的技能项。")
    return _sg_assemble(items)


def rule_based_skill_gap(jd_profile: dict, match_result: dict) -> dict:
    """完全规则、零 LLM 的 skill_gap（合并调用失败时兜底）：status 取规则、叙述走规则模板。"""
    return build_skill_gap([], jd_profile, match_result)


# ===== 对外主入口：合并调用（1 次 LLM）产出 报告 + skill_gap =====
def generate_report_and_gap(
    resume_profile: dict,
    jd_profile: dict,
    match_result: dict,
) -> tuple:
    """一次 LLM 调用产出 技能差距 items + reason + suggestion，再 Python 组装。

    返回 (report, skill_gap, writer_out)：
      - report     ：完整文本报告（含【技能差距分析】段）；
      - skill_gap  ：由 items 聚合（build_skill_gap，见「技能差距视图」一节）；items 缺失则规则兜底；
      - writer_out ：{reason, suggestion}，缓存后供换一批/重排时纯 Python 重组报告（不再调 LLM）。
    """
    combined = call_llm_report_and_gap(resume_profile, jd_profile, match_result)
    skill_gap = build_skill_gap(combined.get("items") or [], jd_profile, match_result)
    writer_out = {"reason": combined["reason"], "suggestion": combined["suggestion"]}
    report = _compose_report(jd_profile, match_result, skill_gap,
                             writer_out["reason"], writer_out["suggestion"], include_skill_gap=True)
    return report, skill_gap, writer_out


def recompose_report(
    jd_profile: dict,
    match_result: dict,
    skill_gap: Optional[dict],
    writer_out: dict,
) -> str:
    """纯 Python 用缓存的 reason/suggestion + 当前 match_score 重组报告（换一批/重排刷新分数，不调 LLM）。"""
    writer_out = writer_out or {}
    return _compose_report(jd_profile, match_result, skill_gap,
                           writer_out.get("reason", ""), writer_out.get("suggestion", ""),
                           include_skill_gap=True)


def generate_recommendation(
    jd_profile: dict,
    match_result: dict,
    skill_gap: Optional[dict] = None,
) -> str:
    """规则兜底报告（无 LLM）：reason/suggestion 走规则文案，技能差距段不渲染。用于合并调用失败时。"""
    return _compose_report(jd_profile, match_result, skill_gap, "", "", include_skill_gap=False)


def generate_full_report(
    jd_profile: dict,
    match_result: dict,
    skill_gap: Optional[dict] = None,
) -> str:
    """规则兜底完整报告（无 LLM）：reason/suggestion 走规则文案，渲染技能差距段。"""
    return _compose_report(jd_profile, match_result, skill_gap, "", "", include_skill_gap=True)


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
