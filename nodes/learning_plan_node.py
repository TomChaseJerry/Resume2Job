"""
nodes/learning_plan_node.py

阶段化学习路径规划节点（Skill Gap Analysis 之后的内部增强节点）。

定位：
    把已有的 skill_gap 结构化结果，进一步转化为针对「当前评估岗位」的、
    可执行的阶段化学习计划，写入 state["learning_plan"]。

接入链路（本次不修改 graph.py）：
    match_scorer → skill_gap_analyzer → learning_plan_node → recommendation_writer

设计原则（与项目「LLM 负责语义、Python 负责确定性编排」一致）：
    - Python：时间约束提取、技能优先级排序、阶段划分、整体建议、输出校验与兜底；
    - LLM   ：仅为「单个阶段」生成 goal / tasks / resources / resume_update_suggestion。

健壮性：
    - 单次 LLM 失败只降级该阶段，不中断整个工作流；
    - 输入字段缺失时使用默认值；
    - 节点不抛异常，错误统一追加到 state["errors"]。
"""

import os
import re
import json
from typing import Any, Optional, Dict, List

from state import AgentState


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
            print("[learning_plan] 未配置 DASHSCOPE_API_KEY，阶段内容将使用规则兜底")
            return None
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=MODEL_NAME,
                openai_api_key=api_key,
                openai_api_base=BASE_URL,
                temperature=0.2,
            )
        except Exception as exc:
            print(f"[learning_plan] LLM 客户端初始化失败：{exc}")
            return None


# ===== 排序权重表 =====
# 第一键：importance（兼容新旧词表，未知按 0）
IMPORTANCE_WEIGHT = {
    "must": 3,
    "core": 3,
    "required": 2,
    "preferred": 1,
    "bonus": 1,
}
# 第二键：status（兼容大小写）
STATUS_WEIGHT = {
    "missing": 3,
    "MISSING": 3,
    "weak": 2,
    "WEAK": 2,
    "matched": 0,
    "STRONG": 0,
}

# 阶段名称固定三档
_STAGE_NAMES = ["第一阶段", "第二阶段", "第三阶段"]

# 时间约束默认值与范围保护
_DEFAULT_TARGET_DAYS = 28
_DEFAULT_DAILY_HOURS = 2.0
_MIN_TARGET_DAYS, _MAX_TARGET_DAYS = 7, 90
_MIN_DAILY_HOURS, _MAX_DAILY_HOURS = 0.5, 8.0

# 中文数字（覆盖时间约束常见取值）
_CN_DIGIT = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


# ===== LLM System Prompt =====
_STAGE_SYSTEM_PROMPT = """你是一位资深 AI 求职辅导专家，擅长为候选人生成针对具体岗位的可执行学习计划。

要求：
1. 建议必须围绕候选人的真实技能差距；
2. 不得输出泛泛建议；
3. 每条任务必须包含具体行动动词和可验证产出物；
4. 不得编造候选人已经掌握某项技能；
5. 不得建议学习阶段之外的技能；
6. resources 优先使用官方文档、经典开源项目或通用可信资源；
7. 如果不确定资源 URL，不要编造 URL，可将 url 置为空字符串；
8. 输出严格 JSON；
9. 不输出 Markdown、注释或额外文字。"""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def append_error(state: AgentState, message: str) -> AgentState:
    """安全地把一条错误/警告追加到 errors，返回更新后的浅拷贝 State（不就地改原列表）。"""
    new_state = dict(state)
    errors = list(state.get("errors") or [])
    errors.append(message)
    new_state["errors"] = errors
    return new_state


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


def clamp(value: float, low: float, high: float) -> float:
    """把数值约束到 [low, high]。"""
    return max(low, min(high, value))


def _cn_to_int(token: str) -> Optional[int]:
    """把阿拉伯/中文数字串转为 int，覆盖 1~99 常见写法；无法识别返回 None。"""
    token = (token or "").strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if token in _CN_DIGIT:
        return _CN_DIGIT[token]
    if "十" in token:
        left, _, right = token.partition("十")
        tens = _CN_DIGIT.get(left, 1) if left else 1
        ones = _CN_DIGIT.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def normalize_skill_gap_item(raw: dict) -> dict:
    """把单条 skill_gap item 归一化为统一结构，兼容旧字段。

    归一化：
        importance: must / preferred（旧 core/required → must，preferred/bonus → preferred）；
        status:     matched / weak / missing（旧 MISSING/WEAK/STRONG → 小写 / matched）；
        gap_reason: 取 gap_reason 或旧 gap_description。
    另保留 importance_raw（用于细粒度排序权重）。
    """
    if not isinstance(raw, dict):
        raw = {}

    skill = str(raw.get("skill") or raw.get("name") or "").strip()

    imp_raw = str(raw.get("importance") or "").strip().lower()
    if imp_raw in ("must", "core", "required"):
        importance = "must"
    elif imp_raw in ("preferred", "bonus", "optional", "nice_to_have", "nice-to-have"):
        importance = "preferred"
    else:
        importance = "preferred"  # 未知重要度按非必备处理，避免过度拔高优先级

    status_raw = str(raw.get("status") or "").strip().lower()
    if status_raw == "missing":
        status = "missing"
    elif status_raw == "weak":
        status = "weak"
    elif status_raw in ("matched", "strong"):
        status = "matched"
    else:
        status = "missing"  # 未知状态保守纳入学习

    gap_reason = raw.get("gap_reason") or raw.get("gap_description") or ""
    suggestion = raw.get("suggestion") or ""

    evidence = raw.get("resume_evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(e).strip() for e in evidence if isinstance(e, (str, int, float)) and str(e).strip()]

    return {
        "skill": skill,
        "importance": importance,
        "importance_raw": imp_raw,
        "status": status,
        "gap_reason": str(gap_reason).strip(),
        "suggestion": str(suggestion).strip(),
        "resume_evidence": evidence,
    }


# ---------------------------------------------------------------------------
# 时间约束提取
# ---------------------------------------------------------------------------
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
        val = _cn_to_int(m.group(1))
        target_days = val if val else None
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
    if not target_days:
        target_days = _DEFAULT_TARGET_DAYS
    if not daily_hours:
        daily_hours = _DEFAULT_DAILY_HOURS

    target_days = int(clamp(target_days, _MIN_TARGET_DAYS, _MAX_TARGET_DAYS))
    daily_hours = round(clamp(daily_hours, _MIN_DAILY_HOURS, _MAX_DAILY_HOURS), 1)

    return {"target_days": target_days, "daily_hours": daily_hours}


# ---------------------------------------------------------------------------
# 选取目标 match_result
# ---------------------------------------------------------------------------
def select_target_match_result(state: AgentState) -> Optional[dict]:
    """选择需要生成学习计划的岗位。

    优先级：State.target_job_id 命中 > match_results[0]；无可用结果返回 None。
    """
    match_results = state.get("match_results") or []
    if not match_results:
        return None

    target_job_id = state.get("target_job_id")
    if target_job_id:
        for mr in match_results:
            if isinstance(mr, dict) and mr.get("job_id") == target_job_id:
                return mr

    first = match_results[0]
    return first if isinstance(first, dict) else None


# ---------------------------------------------------------------------------
# 技能优先级排序（确定性，不依赖 LLM）
# ---------------------------------------------------------------------------
def _core_requirement_weights(skill_gap: dict, jd_profile: dict) -> Dict[str, float]:
    """合并 jd_profile / skill_gap 中可选的核心技能权重表，键统一小写。"""
    weights: Dict[str, float] = {}
    for src in (jd_profile, skill_gap):
        cr = src.get("core_requirements") if isinstance(src, dict) else None
        if isinstance(cr, dict):
            for k, v in cr.items():
                try:
                    weights[str(k).strip().lower()] = float(v)
                except (TypeError, ValueError):
                    weights[str(k).strip().lower()] = 1.0
    return weights


def prioritize_skills(skill_gap: dict, jd_profile: dict) -> List[dict]:
    """归一化 items，仅保留 missing/weak，按三键确定性排序后返回。

    排序键（均降序）：importance 权重 → status 权重 → 岗位核心技能权重。
    """
    skill_gap = skill_gap or {}
    jd_profile = jd_profile or {}

    raw_items = skill_gap.get("items")
    if not isinstance(raw_items, list):
        return []

    core_weights = _core_requirement_weights(skill_gap, jd_profile)

    normalized = [normalize_skill_gap_item(it) for it in raw_items]
    to_learn = [it for it in normalized if it["skill"] and it["status"] in ("missing", "weak")]

    def sort_key(item: dict):
        imp_w = IMPORTANCE_WEIGHT.get(item["importance_raw"],
                                      IMPORTANCE_WEIGHT.get(item["importance"], 1))
        status_w = STATUS_WEIGHT.get(item["status"], 0)
        core_w = core_weights.get(item["skill"].strip().lower(), 0.0)
        return (-imp_w, -status_w, -core_w)

    return sorted(to_learn, key=sort_key)


# ---------------------------------------------------------------------------
# 阶段划分（确定性）
# ---------------------------------------------------------------------------
def divide_into_stages(sorted_skills: List[dict], target_days: int) -> List[dict]:
    """把已排序技能划分为最多 3 个阶段，并分配天数。

    分组：
        第一阶段：must 且 missing（核心缺失，最多 4 个，超出顺延第二阶段）；
        第二阶段：must 且 weak（核心弱匹配，含第一阶段溢出）；
        第三阶段：preferred（其余弱/缺失能力）。
    天数：每阶段 ≥ 3 天，总和 ≤ target_days，天数不足时优先保障靠前阶段。
    """
    stage1 = [s for s in sorted_skills if s["importance"] == "must" and s["status"] == "missing"]
    stage2 = [s for s in sorted_skills if s["importance"] == "must" and s["status"] == "weak"]
    stage3 = [s for s in sorted_skills if s["importance"] == "preferred"]

    # 第一阶段技能过多时，最多保留 4 个，其余顺延到第二阶段前部
    if len(stage1) > 4:
        overflow = stage1[4:]
        stage1 = stage1[:4]
        stage2 = overflow + stage2

    grouped = [skills for skills in (stage1, stage2, stage3) if skills]
    if not grouped:
        return []

    # 天数较短时优先保障靠前阶段：只资助前 k 个阶段（每阶段至少 3 天）
    k = min(len(grouped), max(1, target_days // 3))
    grouped = grouped[:k]

    base, rem = divmod(target_days, k)
    stages: List[dict] = []
    cursor = 1
    for idx, skills in enumerate(grouped):
        stage_days = base + (1 if idx < rem else 0)
        end = cursor + stage_days - 1
        stages.append({
            "stage": _STAGE_NAMES[idx],
            "days": f"第 {cursor}-{end} 天",
            "stage_days": stage_days,
            "focus_skills": [s["skill"] for s in skills],
            "skills": skills,  # 完整 dict 列表，供 LLM 使用；输出前会剔除
        })
        cursor = end + 1

    return stages


# ---------------------------------------------------------------------------
# 单阶段 LLM 生成 + 校验 + 降级
# ---------------------------------------------------------------------------
def _build_stage_user_prompt(stage: dict, daily_hours: float, jd_profile: dict) -> str:
    """构造单阶段 User Prompt，skills_json 传入完整技能 dict。"""
    skills_for_llm = [{
        "skill": s.get("skill"),
        "importance": s.get("importance"),
        "status": s.get("status"),
        "gap_reason": s.get("gap_reason"),
        "suggestion": s.get("suggestion"),
        "resume_evidence": s.get("resume_evidence", []),
    } for s in stage.get("skills", [])]
    skills_json = json.dumps(skills_for_llm, ensure_ascii=False, indent=2)

    responsibilities = jd_profile.get("responsibilities")
    if isinstance(responsibilities, list):
        responsibilities_text = "\n".join(f"- {r}" for r in responsibilities if r)
    else:
        responsibilities_text = str(responsibilities or "")

    job_title = jd_profile.get("title") or "目标岗位"
    job_direction = jd_profile.get("direction") or jd_profile.get("business_area") or "未明确"
    business_area = jd_profile.get("business_area") or "未明确"

    return f"""## 当前阶段信息
阶段名称：{stage.get('stage')}
阶段时长：{stage.get('stage_days')} 天
每日可学习时长：{daily_hours} 小时
阶段目标技能：
{skills_json}

## 岗位信息
岗位名称：{job_title}
岗位方向：{job_direction}
业务场景：{business_area}
岗位职责摘要：
{responsibilities_text}

## 任务

请为本阶段生成以下内容，严格基于阶段目标技能，不得引入阶段之外的技能：

1. goal：一句话描述本阶段核心目标，不超过 25 字；
2. tasks：3-5 条可执行学习任务，每条必须包含：
   - 具体操作，动词开头；
   - 明确的学习材料、工具或项目名称；
   - 可验证的产出物，例如“提交到 GitHub”“跑通示例输出”“整理一页复盘笔记”；
3. resources：每个技能对应 1-2 条推荐资源，优先官方文档和开源项目；
4. resume_update_suggestion：完成本阶段后，简历中可新增或修改的具体描述，一句话，必须基于真实将完成的任务，不得夸大。

## 输出格式

{{
  "goal": "",
  "tasks": [],
  "resources": [
    {{
      "type": "官方文档 | 开源项目 | 技术博客 | 课程",
      "name": "",
      "url": "",
      "focus": ""
    }}
  ],
  "resume_update_suggestion": ""
}}"""


def _fallback_stage_content(skill_names: List[str]) -> dict:
    """单阶段 LLM 失败时的规则降级内容。"""
    names = skill_names or ["目标技能"]
    first = names[0]
    joined = ", ".join(names)
    return {
        "goal": f"补强 {joined} 相关能力",
        "tasks": [
            f"阅读 {first} 相关官方文档或高质量教程，并整理一页学习笔记",
            f"完成一个围绕 {first} 的最小可运行 Demo，并保存运行截图或日志",
            f"将 Demo 提交到 GitHub，并在 README 中说明实现目标、关键步骤和结果",
        ],
        "resources": [],
        "resume_update_suggestion": f"完成后可在简历中补充 {joined} 相关学习 Demo 或项目实践",
    }


def _validate_stage_content(parsed: dict, skill_names: List[str]) -> dict:
    """校验/修正 LLM 单阶段输出，缺字段用规则兜底。"""
    fallback = _fallback_stage_content(skill_names)

    goal = parsed.get("goal")
    goal = goal.strip() if isinstance(goal, str) and goal.strip() else fallback["goal"]

    tasks = parsed.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    tasks = [t.strip() for t in tasks if isinstance(t, str) and t.strip()]
    # 少于 3 条用降级任务补足，去重；多于 5 条截断
    for fb in fallback["tasks"]:
        if len(tasks) >= 3:
            break
        if fb not in tasks:
            tasks.append(fb)
    tasks = tasks[:5]

    resources = parsed.get("resources")
    if not isinstance(resources, list):
        resources = []
    resources = [r for r in resources if isinstance(r, dict)]

    resume = parsed.get("resume_update_suggestion")
    resume = resume.strip() if isinstance(resume, str) and resume.strip() else fallback["resume_update_suggestion"]

    return {
        "goal": goal,
        "tasks": tasks,
        "resources": resources,
        "resume_update_suggestion": resume,
    }


def generate_stage_plan(stage: dict, daily_hours: float, jd_profile: dict, llm_client) -> dict:
    """为单个阶段生成 goal/tasks/resources/resume_update_suggestion。

    LLM 失败或解析失败时返回规则降级内容，并通过返回值的 "_error" 字段上报错误（不抛异常）。
    """
    skill_names = stage.get("focus_skills") or [s.get("skill") for s in stage.get("skills", [])]

    if llm_client is None:
        content = _fallback_stage_content(skill_names)
        content["_error"] = f"{stage.get('stage')}: LLM 客户端不可用，使用规则兜底"
        return content

    try:
        user_prompt = _build_stage_user_prompt(stage, daily_hours, jd_profile or {})
        response = llm_client.invoke([
            {"role": "system", "content": _STAGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        raw = getattr(response, "content", "") or ""
    except Exception as exc:
        content = _fallback_stage_content(skill_names)
        content["_error"] = f"{stage.get('stage')}: LLM 调用失败：{exc}"
        return content

    parsed = safe_json_parse(raw)
    if parsed is None:
        content = _fallback_stage_content(skill_names)
        content["_error"] = f"{stage.get('stage')}: LLM 输出解析失败，使用规则兜底"
        return content

    content = _validate_stage_content(parsed, skill_names)
    content["_error"] = None
    return content


# ---------------------------------------------------------------------------
# overall_suggestion（规则生成，不调用 LLM）
# ---------------------------------------------------------------------------
def _build_overall_suggestion(sorted_skills: List[dict], skill_gap: dict) -> str:
    """根据风险与技能差距类型生成整体建议。"""
    if not sorted_skills:
        return "当前技能与岗位匹配度较高，无需额外学习规划，建议直接进入面试准备阶段。"

    overall_risk = str((skill_gap or {}).get("overall_risk") or "").strip().lower()
    has_missing_must = any(s["importance"] == "must" and s["status"] == "missing" for s in sorted_skills)

    if has_missing_must or overall_risk == "high":
        return "建议优先完成第一阶段中可展示的项目产出，面试前一周停止新学习，集中复盘已有项目细节。"
    return "建议围绕弱匹配技能补充可展示证据，例如 Demo、实验记录或项目 README，以提高面试中的说服力。"


# ---------------------------------------------------------------------------
# 组装完整学习计划
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


def build_learning_plan(selected_match_result: dict, user_query: str, llm_client) -> dict:
    """基于选中岗位的 skill_gap + 用户时间约束，组装阶段化学习计划。

    收集到的阶段级错误存放在返回值的私有键 "_errors"（list），由节点统一追加到 errors。
    """
    selected_match_result = selected_match_result or {}
    skill_gap = selected_match_result.get("skill_gap") or {}
    jd_profile = selected_match_result.get("jd_profile") or {}

    target_job_id = selected_match_result.get("job_id") or ""
    target_job_title = jd_profile.get("title") or "目标岗位"

    time_conf = extract_time_constraints(user_query)
    target_days = time_conf["target_days"]
    daily_hours = time_conf["daily_hours"]

    sorted_skills = prioritize_skills(skill_gap, jd_profile)

    # 无待补强技能（全部 matched）：返回空计划 + 匹配度较高提示
    if not sorted_skills:
        plan = _empty_plan(
            target_job_id, target_job_title, daily_hours,
            _build_overall_suggestion([], skill_gap), None,
        )
        plan["_errors"] = []
        return plan

    stage_skeletons = divide_into_stages(sorted_skills, target_days)

    stages: List[dict] = []
    errors: List[str] = []
    for i, skeleton in enumerate(stage_skeletons):
        content = generate_stage_plan(skeleton, daily_hours, jd_profile, llm_client)
        stage_error = content.pop("_error", None)
        if stage_error:
            errors.append(f"learning_plan_node: {stage_error}")

        stage_days = skeleton["stage_days"]
        skill_names = skeleton["focus_skills"]
        print(f"[learning_plan] 阶段 {i + 1} 生成完成，技能：{skill_names}，天数：{stage_days}")

        stages.append({
            "stage": skeleton["stage"],
            "days": skeleton["days"],
            "stage_days": stage_days,
            "focus_skills": skill_names,
            "goal": content["goal"],
            "tasks": content["tasks"],
            "resources": content["resources"],
            "resume_update_suggestion": content["resume_update_suggestion"],
        })

    plan = {
        "target_job_id": target_job_id,
        "target_job_title": target_job_title,
        "total_plan_days": sum(s["stage_days"] for s in stages),
        "daily_hours": daily_hours,
        "stages": stages,
        "overall_suggestion": _build_overall_suggestion(sorted_skills, skill_gap),
        "error": None,
    }
    plan["_errors"] = errors
    return plan


# ---------------------------------------------------------------------------
# LangGraph 节点
# ---------------------------------------------------------------------------
def learning_plan_node(state: AgentState) -> AgentState:
    """
    LangGraph 节点函数。
    从当前岗位的 skill_gap 结果中读取待补强技能，
    结合用户时间约束生成阶段化学习路径，
    并写入 state["learning_plan"]。
    """
    print("[learning_plan_node] 开始执行...")

    # 默认关闭：仅当 planner 依据用户措辞开启 need_learning_plan 时才生成
    plan_cfg = state.get("plan") or {}
    if not plan_cfg.get("need_learning_plan"):
        print("[learning_plan_node] 未请求学习计划，跳过")
        return state

    new_state = dict(state)  # 浅拷贝，保留原有字段
    user_query = state.get("user_query") or ""
    daily_default = extract_time_constraints(user_query)["daily_hours"]

    try:
        selected = select_target_match_result(state)
        if selected is None:
            new_state["learning_plan"] = _empty_plan(
                "", "", daily_default, "",
                "learning_plan_node: 无可用 match_result，跳过学习计划生成",
            )
            return append_error(new_state, "learning_plan_node: 无可用 match_result，跳过学习计划生成")

        skill_gap = selected.get("skill_gap") or {}
        items = skill_gap.get("items")
        if not skill_gap or not isinstance(items, list) or not items:
            job_id = selected.get("job_id") or ""
            job_title = (selected.get("jd_profile") or {}).get("title") or "目标岗位"
            new_state["learning_plan"] = _empty_plan(
                job_id, job_title, daily_default, "",
                "learning_plan_node: 选中岗位缺少 skill_gap 结果，跳过学习计划生成",
            )
            return append_error(new_state, "learning_plan_node: 选中岗位缺少 skill_gap 结果，跳过学习计划生成")

        llm_client = get_llm_client()
        plan = build_learning_plan(selected, user_query, llm_client)
        stage_errors = plan.pop("_errors", [])

        new_state["learning_plan"] = plan
        for msg in stage_errors:
            new_state = append_error(new_state, msg)
        return new_state

    except Exception as exc:
        # 兜底：任何未预期异常都不得中断工作流
        new_state["learning_plan"] = _empty_plan(
            "", "", daily_default, "",
            f"learning_plan_node: 未预期异常：{exc}",
        )
        return append_error(new_state, f"learning_plan_node: 未预期异常：{exc}")
