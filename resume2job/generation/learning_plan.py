"""
阶段化学习计划模块（Learning Plan，可选增强）。

定位：
    把岗位评估产出的 skill_gap 结构化结果，转化为针对「当前评估岗位」的、
    可执行的阶段化学习计划。

与旧版的差异：
    旧版按阶段拆分后「每个阶段单独调一次 LLM」，慢且贵；
    现在 Python 先做确定性编排（时间约束提取 → 技能优先级排序 → 阶段骨架划分），
    再用 **单次 LLM 调用** 一次性生成所有阶段的 goal / tasks / resources /
    resume_update_suggestion，失败时整体走规则兜底。

分工（与项目「LLM 负责语义、Python 负责确定性编排」一致）：
    - Python：时间约束提取（正则）、技能优先级排序、阶段划分、输出校验与兜底；
    - LLM   ：仅生成各阶段的语义内容。
"""

import re
import json
from typing import Dict, List, Optional

from resume2job.core.config import CHAT_MODEL as MODEL_NAME
from resume2job.core.llm import call_llm as _core_call_llm, safe_json_parse

# ===== 时间约束默认值与范围保护 =====
_DEFAULT_TARGET_DAYS = 30
_MIN_TARGET_DAYS, _MAX_TARGET_DAYS = 7, 180
_DEFAULT_DAILY_HOURS = 2.0
_MIN_DAILY_HOURS, _MAX_DAILY_HOURS = 0.5, 12.0

# ===== 优先级权重（确定性排序用）=====
_IMPORTANCE_WEIGHT = {"must": 2, "preferred": 1}
_STATUS_WEIGHT = {"missing": 2, "weak": 1}

# ===== 阶段划分参数 =====
_MAX_STAGES = 3          # 最多 3 个阶段（基础补齐 / 重点强化 / 综合冲刺）
_SKILLS_PER_STAGE = 3    # 每阶段最多聚焦技能数


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_CN_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_to_int(token: str) -> Optional[int]:
    """把阿拉伯数字或简单中文数字（一~十、两）转为 int，失败返回 None。"""
    token = (token or "").strip()
    if token.isdigit():
        return int(token)
    if token in _CN_DIGITS:
        return _CN_DIGITS[token]
    # 「十N」「N十」「N十M」形式
    if "十" in token:
        parts = token.split("十")
        tens = _CN_DIGITS.get(parts[0], 1) if parts[0] else 1
        ones = _CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return None


def extract_time_constraints(user_query: str) -> dict:
    """从用户问题中用正则提取时间约束（不调用 LLM）。

    支持：N天 / N周 / 四周 / 两周 / N个月 / 一个月；每天N小时 / 一天Nh / 每天学Nh。
    缺失则用默认值，识别到的异常值做范围保护。
    """
    q = user_query or ""
    target_days: Optional[int] = None
    daily_hours: Optional[float] = None

    # 1) 每日学习时长：每天 / 一天 / 每日 + （学/学习）+ 数字 + 小时/h
    daily_pat = r"(?:每天|每日|一天)\s*(?:学习|学|看)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:个?小时|小時|h|H)"
    m = re.search(daily_pat, q)
    if m:
        try:
            daily_hours = float(m.group(1))
        except ValueError:
            daily_hours = None

    # 2) 天数解析前，先剔除每日时长短语，避免「一天3小时」被误读为「1 天」目标
    q_days = re.sub(daily_pat, " ", q)

    # 优先「N天」，其次「N周」(×7)，再次「N个月」(×30)
    m = re.search(r"([0-9]+|[一二两三四五六七八九十]+)\s*天", q_days)
    if m:
        target_days = _cn_to_int(m.group(1))
    if target_days is None:
        m = re.search(r"([0-9]+|[一二两三四五六七八九十]+)\s*周", q_days)
        if m:
            val = _cn_to_int(m.group(1))
            target_days = val * 7 if val else None
    if target_days is None:
        m = re.search(r"([0-9]+|[一二两三四五六七八九十]+)\s*个?\s*月", q_days)
        if m:
            val = _cn_to_int(m.group(1))
            target_days = val * 30 if val else None

    # 3) 默认值 + 范围保护
    target_days = int(_clamp(target_days or _DEFAULT_TARGET_DAYS,
                             _MIN_TARGET_DAYS, _MAX_TARGET_DAYS))
    daily_hours = round(_clamp(daily_hours or _DEFAULT_DAILY_HOURS,
                               _MIN_DAILY_HOURS, _MAX_DAILY_HOURS), 1)

    return {"target_days": target_days, "daily_hours": daily_hours}


# ---------------------------------------------------------------------------
# 技能优先级排序 + 阶段划分（确定性，不依赖 LLM）
# ---------------------------------------------------------------------------
def prioritize_skills(skill_gap: dict) -> List[dict]:
    """从 skill_gap.items 取 missing / weak 技能并按确定性优先级排序。

    排序键（降序）：importance（must > preferred）→ status（missing > weak）。
    """
    items = (skill_gap or {}).get("items")
    if not isinstance(items, list):
        return []

    to_learn = []
    for it in items:
        if not isinstance(it, dict):
            continue
        skill = str(it.get("skill") or "").strip()
        status = it.get("status")
        if skill and status in ("missing", "weak"):
            to_learn.append({
                "skill": skill,
                "status": status,
                "importance": it.get("importance") if it.get("importance") in _IMPORTANCE_WEIGHT else "preferred",
                "suggestion": str(it.get("suggestion") or "").strip(),
            })

    to_learn.sort(key=lambda it: (-_IMPORTANCE_WEIGHT[it["importance"]],
                                  -_STATUS_WEIGHT[it["status"]]))
    return to_learn


def divide_into_stages(sorted_skills: List[dict], target_days: int) -> List[dict]:
    """把排序后的技能切成至多 3 个阶段骨架，天数按阶段技能数比例分配。"""
    if not sorted_skills:
        return []

    chunks = [sorted_skills[i:i + _SKILLS_PER_STAGE]
              for i in range(0, len(sorted_skills), _SKILLS_PER_STAGE)][:_MAX_STAGES]

    total_skills = sum(len(c) for c in chunks)
    skeletons, used_days = [], 0
    for idx, chunk in enumerate(chunks, start=1):
        if idx == len(chunks):
            stage_days = max(1, target_days - used_days)  # 最后一段兜底吃掉余数
        else:
            stage_days = max(1, round(target_days * len(chunk) / total_skills))
        skeletons.append({
            "stage": idx,
            "days": f"第{used_days + 1}-{used_days + stage_days}天",
            "stage_days": stage_days,
            "focus_skills": [it["skill"] for it in chunk],
            "skill_details": chunk,
        })
        used_days += stage_days
    return skeletons


# ---------------------------------------------------------------------------
# 单次 LLM 调用：一次性生成所有阶段内容
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """你是一位 AI 算法岗求职辅导专家，擅长把技能缺口转化为可执行的学习计划。

要求：
1. 只输出合法 JSON，不输出 Markdown、解释或额外文字；
2. 学习任务必须具体可执行（看什么、做什么、产出什么），不要空泛口号；
3. resources 给出公开可得的课程 / 文档 / 项目实践方向，不编造失效链接；
4. resume_update_suggestion 说明完成该阶段后简历上可以新增/强化的表述。"""


def _build_user_prompt(skeletons: List[dict], daily_hours: float, jd_profile: dict) -> str:
    """构造单次调用的 User Prompt：岗位摘要 + 全部阶段骨架 + 输出 Schema。"""
    jd = jd_profile or {}
    jd_summary = {
        "company": jd.get("company"),
        "title": jd.get("title"),
        "direction": jd.get("direction"),
        "hard_skills": jd.get("hard_skills"),
    }
    stage_brief = [
        {
            "stage": s["stage"],
            "days": s["days"],
            "stage_days": s["stage_days"],
            "focus_skills": s["focus_skills"],
            "skill_status": {it["skill"]: it["status"] for it in s["skill_details"]},
        }
        for s in skeletons
    ]
    return f"""## 目标岗位摘要
{json.dumps(jd_summary, ensure_ascii=False, indent=2)}

## 学习阶段骨架（已按优先级排序与天数分配，请勿改动阶段划分）
{json.dumps(stage_brief, ensure_ascii=False, indent=2)}

## 学习条件
每天可投入约 {daily_hours} 小时。

## 任务
为上面**每个阶段**生成学习内容。每阶段输出：
- goal：一句话阶段目标（与 focus_skills 和岗位要求对应）；
- tasks：3~5 条具体任务（结合每天 {daily_hours} 小时的量力安排）；
- resources：2~3 条学习资源或实践方向；
- resume_update_suggestion：完成后简历可补充的一句表述。

## 输出格式（严格 JSON）
{{
  "stages": [
    {{
      "stage": 1,
      "goal": "",
      "tasks": [],
      "resources": [],
      "resume_update_suggestion": ""
    }}
  ]
}}"""


def _fallback_stage_content(focus_skills: List[str]) -> dict:
    """LLM 失败时的规则兜底阶段内容。"""
    names = "、".join(focus_skills) or "目标技能"
    return {
        "goal": f"系统补齐 {names} 的基础认知与上手实践。",
        "tasks": [
            f"通读 {names} 的官方文档或入门教程，整理核心概念笔记",
            f"完成一个使用 {names} 的最小可运行示例",
            "把所学应用到自己已有项目中，记录遇到的问题与解决过程",
        ],
        "resources": [f"{names} 官方文档", "相关开源项目源码与示例"],
        "resume_update_suggestion": f"可在简历技能/项目部分补充 {names} 的实践经验描述。",
    }


def _as_str_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _merge_llm_stages(skeletons: List[dict], parsed: Optional[dict]) -> List[dict]:
    """把 LLM 输出按 stage 序号对齐回骨架；缺失/非法的阶段用规则兜底。"""
    llm_stages = {}
    if isinstance(parsed, dict) and isinstance(parsed.get("stages"), list):
        for s in parsed["stages"]:
            if isinstance(s, dict) and isinstance(s.get("stage"), int):
                llm_stages[s["stage"]] = s

    stages = []
    for skeleton in skeletons:
        content = llm_stages.get(skeleton["stage"]) or {}
        fallback = _fallback_stage_content(skeleton["focus_skills"])
        stages.append({
            "stage": skeleton["stage"],
            "days": skeleton["days"],
            "stage_days": skeleton["stage_days"],
            "focus_skills": skeleton["focus_skills"],
            "goal": str(content.get("goal") or "").strip() or fallback["goal"],
            "tasks": _as_str_list(content.get("tasks")) or fallback["tasks"],
            "resources": _as_str_list(content.get("resources")) or fallback["resources"],
            "resume_update_suggestion": str(content.get("resume_update_suggestion") or "").strip()
                                        or fallback["resume_update_suggestion"],
        })
    return stages


def _build_overall_suggestion(sorted_skills: List[dict]) -> str:
    """整体建议（确定性模板）。"""
    if not sorted_skills:
        return "当前匹配情况较好，没有需要集中补强的技能缺口，建议正常投递并在面试中突出已有项目证据。"
    missing = [it["skill"] for it in sorted_skills if it["status"] == "missing"]
    weak = [it["skill"] for it in sorted_skills if it["status"] == "weak"]
    parts = []
    if missing:
        parts.append(f"优先补齐缺口技能：{'、'.join(missing[:3])}")
    if weak:
        parts.append(f"强化弱匹配技能：{'、'.join(weak[:3])}")
    parts.append("每完成一个阶段就同步更新简历，并用小项目沉淀可展示的证据。")
    return "；".join(parts) + "。"


# ---------------------------------------------------------------------------
# 对外主函数
# ---------------------------------------------------------------------------
def _empty_plan(target_job_id: str, target_job_title: str, daily_hours: float,
                overall_suggestion: str, error: Optional[str]) -> dict:
    """构造空学习计划（无技能可补强 / 输入缺失场景）。"""
    return {
        "target_job_id": target_job_id,
        "target_job_title": target_job_title,
        "total_plan_days": 0,
        "daily_hours": daily_hours,
        "stages": [],
        "overall_suggestion": overall_suggestion,
        "error": error,
    }


def build_learning_plan(match_result: dict, user_query: str) -> dict:
    """基于选中岗位的 skill_gap + 用户时间约束，生成阶段化学习计划。

    单次 LLM 调用生成全部阶段内容；LLM 失败时所有阶段走规则兜底，不抛异常。
    """
    match_result = match_result or {}
    skill_gap = match_result.get("skill_gap") or {}
    jd_profile = match_result.get("jd_profile") or {}

    target_job_id = match_result.get("job_id") or ""
    target_job_title = jd_profile.get("title") or "目标岗位"

    time_conf = extract_time_constraints(user_query)
    target_days, daily_hours = time_conf["target_days"], time_conf["daily_hours"]

    sorted_skills = prioritize_skills(skill_gap)

    # 无待补强技能（全部 matched）：返回空计划 + 匹配度较高提示
    if not sorted_skills:
        return _empty_plan(target_job_id, target_job_title, daily_hours,
                           _build_overall_suggestion([]), None)

    skeletons = divide_into_stages(sorted_skills, target_days)

    parsed = None
    try:
        raw = _core_call_llm(_SYSTEM_PROMPT,
                             _build_user_prompt(skeletons, daily_hours, jd_profile),
                             model=MODEL_NAME, temperature=0.3)
        parsed = safe_json_parse(raw)
    except Exception as exc:
        print(f"[learning_plan] LLM 调用失败，全部阶段走规则兜底：{exc}")

    stages = _merge_llm_stages(skeletons, parsed)
    print(f"[learning_plan] 生成 {len(stages)} 个阶段（单次 LLM 调用），"
          f"总天数 {sum(s['stage_days'] for s in stages)}，每天 {daily_hours} 小时")

    return {
        "target_job_id": target_job_id,
        "target_job_title": target_job_title,
        "total_plan_days": sum(s["stage_days"] for s in stages),
        "daily_hours": daily_hours,
        "stages": stages,
        "overall_suggestion": _build_overall_suggestion(sorted_skills),
        "error": None,
    }
