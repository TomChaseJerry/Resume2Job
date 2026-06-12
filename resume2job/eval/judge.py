# -*- coding: utf-8 -*-
"""生成质量评测：LLM-as-judge 对推荐报告打分。

检索指标（retrieval_eval）评「找得对不对」，本模块评「写得好不好」——
用独立的 judge 模型（config.JUDGE_MODEL，可与被评模型不同以减小自评偏置）
按三个维度对推荐报告打 1~5 分并给出理由：

    faithfulness 忠实性：报告中的结论/技能描述是否都有简历或 JD 依据，无编造；
    usefulness   有用性：对「投不投、怎么投」是否给出可执行的判断与建议；
    evidence     证据性：关键论断是否引用了具体证据（项目/技能/JD 原文）。

输出恒为合法结构（解析失败时各维度 0 分 + error），不抛异常。
"""

import json
from typing import Optional

from resume2job.core.config import JUDGE_MODEL
from resume2job.core.llm import call_llm, safe_json_parse

_DIMENSIONS = ("faithfulness", "usefulness", "evidence")

_SYSTEM_PROMPT = (
    "你是一个严格的求职推荐系统质量评审员。给定候选人画像、岗位 JD 与系统生成的"
    "推荐报告，按以下维度逐项打分（1~5 整数，5 最好）并给一句话理由：\n"
    "1. faithfulness 忠实性：报告内容是否都有画像/JD 依据，是否存在编造或夸大；\n"
    "2. usefulness 有用性：是否对「投不投、怎么投、补什么」给出可执行建议；\n"
    "3. evidence 证据性：关键论断是否引用了具体证据。\n"
    "只输出 JSON：{\"faithfulness\": {\"score\": 5, \"reason\": \"...\"}, "
    "\"usefulness\": {...}, \"evidence\": {...}, \"overall_comment\": \"...\"}"
)


def _empty_result(error: Optional[str]) -> dict:
    return {
        **{d: {"score": 0, "reason": ""} for d in _DIMENSIONS},
        "avg_score": 0.0,
        "overall_comment": "",
        "error": error,
    }


def judge_report(resume_profile: dict, jd_profile: dict, report: str) -> dict:
    """对一份推荐报告做三维度 LLM-as-judge 打分。"""
    if not isinstance(report, str) or not report.strip():
        return _empty_result("报告为空，无法评审")

    user_prompt = (
        f"## 候选人画像（节选）\n"
        f"{json.dumps(_trim_profile(resume_profile), ensure_ascii=False)[:2500]}\n\n"
        f"## 岗位 JD（结构化）\n"
        f"{json.dumps(jd_profile or {}, ensure_ascii=False)[:2500]}\n\n"
        f"## 待评审的推荐报告\n{report[:3000]}"
    )

    try:
        raw = call_llm(_SYSTEM_PROMPT, user_prompt, model=JUDGE_MODEL)
    except Exception as e:
        return _empty_result(f"judge LLM 调用失败：{e}")

    parsed = safe_json_parse(raw)
    if not isinstance(parsed, dict):
        return _empty_result("judge 输出解析失败")

    result = {}
    scores = []
    for dim in _DIMENSIONS:
        item = parsed.get(dim) if isinstance(parsed.get(dim), dict) else {}
        score = item.get("score")
        score = int(score) if isinstance(score, (int, float)) and 1 <= score <= 5 else 0
        result[dim] = {"score": score, "reason": str(item.get("reason") or "").strip()}
        if score:
            scores.append(score)

    result["avg_score"] = round(sum(scores) / len(scores), 2) if scores else 0.0
    result["overall_comment"] = str(parsed.get("overall_comment") or "").strip()
    result["error"] = None
    return result


def _trim_profile(resume_profile: dict) -> dict:
    """裁剪画像到评审需要的核心字段，控制 prompt 长度。"""
    r = resume_profile or {}
    return {
        "skills": r.get("skills") or r.get("skill_groups"),
        "projects": r.get("projects"),
        "educations": r.get("educations"),
        "experiences": r.get("experiences"),
    }
