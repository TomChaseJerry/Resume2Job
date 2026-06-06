"""
岗位 JD 解析模块（JD Parser）

面向「Agentic RAG 实习岗位智能推荐系统」，将 JD 原文解析为稳定的结构化 JSON。

Pipeline：
    LLM 初步抽取 -> JSON 清洗 -> 解析 -> Schema 默认值补全 -> 规则后处理修正 -> 输出
"""

import os
import re
import sys
import json
import argparse
from typing import Optional, Any

from openai import OpenAI


# ===== 模型 / 接口常量 =====
MODEL_NAME = "qwen-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ===== JSON Schema 示例（用于 Prompt 中向 LLM 展示目标结构）=====
JSON_SCHEMA_EXAMPLE = {
    "company": "string or null",
    "title": "string or null",
    "job_type": "实习 / 全职 / 校招 / 日常实习 / 暑期实习 / null",
    "offers_internship": "true / false / null（该岗位是否提供实习机会）",
    "direction": "大模型算法 / NLP / CV / 推荐 / 后端 / ... / null",
    "business_area": "智能客服 / 搜索推荐 / 多模态 / Agent / RAG / 广告 / 风控 / ... / null",
    "responsibilities": ["岗位职责描述"],

    "hard_skills": ["硬技能：算法方向 / 工程能力 / 编程语言 / 工具框架"],
    "soft_skills": ["软素质：学习能力 / 责任心 / 抗压能力 / 沟通 / 团队协作 ..."],
    "preferred_skills": ["JD 中出现『优先 / 加分 / 有经验更好』表述的技能"],
    "tools_or_frameworks": ["PyTorch / TensorFlow / LangChain / FAISS / Qwen / Llama ..."],
    "domain_keywords": ["RAG / Agent / SFT / Post-training / 多模态 / 推荐系统 ..."],

    "education_requirement": "学历要求原文摘要",
    "education_level": "本科 / 硕士 / 博士 / 不限 / null",
    "experience_requirement": "经验要求原文摘要",

    "internship_duration": "明确的实习时长，如「3个月以上」「每周至少4天，持续6个月」；模糊表述（如「可提供实习」）不填这里",
    "min_internship_months": "number or null",
    "weekly_days_requirement": "每周到岗天数要求 or null",
    "internship_note": "原文中模糊的实习相关补充说明，如「该岗位可提供实习岗位」",

    "location": {
        "city": "string or null",
        "district": "string or null",
        "office_address": "string or null"
    },
    "work_mode": "现场 / 远程 / 混合 / null",
    "salary": "薪资 / 补贴信息 or null",
    "benefits": ["福利 / 补贴 / 转正机会 ..."],
    "highlights": ["岗位亮点"],

    "academic_requirements": ["论文 / 会议 / 专利等学术要求"],
    "competition_requirements": ["竞赛 / 获奖等要求"],

    "risk_points": [
        {
            "type": "技术门槛 / 学历门槛 / 经验门槛 / 时长门槛 / 驻场要求 / ...",
            "description": "自然、简洁的中文描述（禁止病句）",
            "evidence": "来自 JD 原文的句子 or null"
        }
    ],

    "evidence": ["从 JD 原文摘取的关键证据片段"]
}


# ===== System Prompt =====
SYSTEM_PROMPT = """你是一名资深招聘分析师和岗位信息结构化专家，擅长将各类公司的招聘 JD 原文抽取为标准化 JSON，供下游岗位推荐系统使用。

【核心原则】
1. 只输出一个合法的 JSON 对象，禁止输出 Markdown 代码块标记（如 ```json）、解释、前后缀文字。
2. 严禁编造 JD 原文中没有的信息；信息不明确时一律置为 null 或空列表 []。
3. 字段缺失时：字符串字段返回 null，列表字段返回 []，布尔字段返回 null，对象字段保留键并内部用 null / [] 填充。

【字段抽取与规范化】
- job_type：规范化到「实习 / 全职 / 校招 / 日常实习 / 暑期实习」之一，无法判断则 null。
- offers_internship：布尔字段。当 JD 出现「可提供实习」「接受实习生」「可实习」「实习岗位」等明确表达时为 true；明确说明不接受实习时为 false；未提及则 null。
- direction：岗位的技术方向（如大模型算法、NLP、CV、推荐算法、数据工程、后端开发等）。
- business_area：业务场景（如智能客服、搜索推荐、多模态、Agent、RAG、广告、风控）。

- hard_skills：**仅** 保留硬技能 —— 算法方向、工程能力、编程语言、工具框架（如「强化学习」「分布式训练」「Python」「PyTorch」）。
  * **必须原子化**：每个元素应是一个独立、可匹配的原子技能，**禁止把一整句话当作一个技能**。
    例如「深度学习及模仿学习相关领域经验」应拆为「深度学习」「模仿学习」；
    「C/C++ 和 Python 编程能力」应拆为「C/C++」「Python」；
    「ROS、SLAM、运动规划等至少两项核心技术」应拆为「ROS」「SLAM」「运动规划」。
  * 注意：`C/C++` 是一个整体技能，不要再拆成 `C` 和 `C++`；`Isaac Gym` 等带空格的工具名保持完整。
- soft_skills：**仅** 保留软素质 —— 学习能力 / 责任心 / 抗压能力 / 沟通能力 / 团队协作能力 等。
- preferred_skills：JD 中出现「优先 / 加分 / 有经验更好 / 熟悉者优先 / 满足以下任一即可加分」表述的技能。
  * 凡是在「优先」「加分项」「实战经验（满足以下任一即可加分）」等语境下出现的技能，
    一律放入 preferred_skills，**不要**作为 hard_skills 必备技能，以免拉低匹配分。
- tools_or_frameworks：具体的工具 / 框架 / 平台 / 模型名称（如 PyTorch、LangChain、Qwen）。
- domain_keywords：领域 / 范式关键词（如 RAG、Agent、SFT、多模态），不要混入工具名。
- **严禁**将「学习能力 / 责任心 / 抗压能力 / 团队协作 / 沟通能力」放入 hard_skills。
- **严禁**将论文 / 会议 / 专利要求放入 hard_skills，必须放入 academic_requirements。
- **严禁**将竞赛 / 获奖要求放入 hard_skills，必须放入 competition_requirements。

- education_level：必须规范化到「本科 / 硕士 / 博士 / 不限 / null」；「本科及以上」→ 本科，「硕士及以上」→ 硕士。

- internship_duration：**只能填写明确的实习时长或到岗要求**，例如「3个月以上」「每周至少4天，持续6个月」。
  * 如果原文只出现「可提供实习岗位」「接受实习生」「可实习」等模糊表述而**未明确时长**，则：
      - offers_internship = true
      - internship_duration = null
      - min_internship_months = null
      - weekly_days_requirement = null
      - internship_note = 原文对应的那句话（保持原文）
- min_internship_months：必须为整数或 null；如「至少3个月」→ 3，「6个月以上」→ 6，「每周至少4天，持续6个月」→ 6。
- weekly_days_requirement：原文摘要，如「每周至少4天」；无则 null。

- location.city / district / office_address：能拆则拆，不强行编造；多地点取首要工作地。
- work_mode：规范化到「现场 / 远程 / 混合 / null」。
- benefits：薪资之外的福利（餐补、住宿、转正机会、五险一金等）。
- highlights：岗位亮点（如「大模型核心业务」「有转正机会」「技术栈贴近前沿」）。

- risk_points：**必须是结构化对象列表**，每个元素包含三个键：
    * type：风险类型，如「技术门槛」「学历门槛」「经验门槛」「时长门槛」「驻场要求」；
    * description：**自然、简洁、规范的中文表达**，禁止「需要强XXX经验」这类病句；
      例如不要写「需要强强化学习经验」，应写「强化学习能力要求较高」或「对强化学习理论与实践要求较高」。
    * evidence：来自 JD 原文的句子（可为 null，但不得编造）。

- evidence（顶层）：从 JD 原文摘取的关键证据片段，用于支撑后续推荐理由，禁止改写或编造。

【输出】
只输出 JSON 对象本体，确保可被 json.loads 直接解析。"""


# ===== 兜底默认值（用于 validate_jd_profile）=====
DEFAULT_JD = {
    "company": None,
    "title": None,
    "job_type": None,
    "offers_internship": None,
    "direction": None,
    "business_area": None,
    "responsibilities": [],

    "hard_skills": [],
    "soft_skills": [],
    "preferred_skills": [],
    "tools_or_frameworks": [],
    "domain_keywords": [],

    "education_requirement": None,
    "education_level": None,
    "experience_requirement": None,

    "internship_duration": None,
    "min_internship_months": None,
    "weekly_days_requirement": None,
    "internship_note": None,

    "location": {"city": None, "district": None, "office_address": None},
    "work_mode": None,
    "salary": None,
    "benefits": [],
    "highlights": [],

    "academic_requirements": [],
    "competition_requirements": [],

    "risk_points": [],
    "evidence": [],
}


# ===== 软素质 / 学术 / 竞赛识别词表（用于后处理）=====
SOFT_SKILL_KEYWORDS = {
    "学习能力", "自驱力", "自我驱动", "抗压能力", "抗压", "责任心", "上进心", "细心",
    "沟通能力", "沟通", "表达能力", "团队协作能力", "团队协作", "团队合作", "协作能力",
    "人际交往能力", "人际交往", "执行力", "主动性", "积极性", "时间管理", "情绪管理",
}
ACADEMIC_KEYWORDS = ("论文", "顶会", "顶刊", "会议", "期刊", "专利", "SCI", "EI",
                    "CCF", "NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "EMNLP", "KDD")
COMPETITION_KEYWORDS = ("竞赛", "比赛", "获奖", "ACM", "Kaggle", "天池", "数学建模")

# ===== 实习模糊表达识别（用于 normalize_internship_fields）=====
INTERNSHIP_VAGUE_PATTERNS = [
    r"可提供实习",
    r"提供实习(岗位|机会)",
    r"接受实习生",
    r"可实习",
    r"招收实习生",
    r"实习岗位",
    r"有实习机会",
]
# 出现这些模式即认为是 "明确时长"，不应被搬到 internship_note
INTERNSHIP_DURATION_HINT = re.compile(
    r"(\d+\s*(个?月|周|天))|每周.*\d+\s*天|至少.*\d+|以上"
)


# ===== Prompt 构造 =====
def build_user_prompt(jd_text: str) -> str:
    """拼接 User Prompt：JD 原文 + JSON Schema 示例 + 明确规则。"""
    schema_str = json.dumps(JSON_SCHEMA_EXAMPLE, ensure_ascii=False, indent=2)
    return (
        "以下是一段岗位招聘 JD 原文，请按 System 指令与给定 JSON Schema 抽取结构化信息。\n\n"
        "===== JD 原文 BEGIN =====\n"
        f"{jd_text}\n"
        "===== JD 原文 END =====\n\n"
        "===== 目标 JSON Schema（仅示意字段与类型，值为占位说明）=====\n"
        f"{schema_str}\n\n"
        "===== 规则 =====\n"
        "1. 严格按 Schema 的字段名与层级输出；\n"
        "2. hard_skills 只放硬技能且必须原子化（禁止把长句当作单个技能）；soft_skills 只放软素质；"
        "出现「优先 / 加分 / 满足以下任一即可加分」语境的技能放 preferred_skills，不要混入 hard_skills；\n"
        "3. 论文 / 会议 / 专利 → academic_requirements；竞赛 / 获奖 → competition_requirements；\n"
        "4. internship_duration 只填明确时长；模糊表述（如「可提供实习岗位」）放 internship_note 并置 offers_internship=true；\n"
        "5. min_internship_months 必须为整数或 null；\n"
        "6. risk_points 必须为对象列表 {type, description, evidence}，description 须为自然简洁中文，禁止病句；\n"
        "7. education_level / job_type / work_mode 必须规范化到枚举值；\n"
        "8. evidence 必须来自 JD 原文，禁止改写或编造；\n"
        "9. 字段缺失：字符串 null，列表 []，布尔 null，对象保留键并内部补 null/[]；\n"
        "10. 直接输出 JSON 对象本体，不要任何额外文字或 Markdown 包装。"
    )


# ===== LLM 调用 =====
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """调用阿里云百炼兼容接口（OpenAI SDK 风格），返回模型纯文本响应。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # 抽取任务低温度，结构稳定
    )
    return (completion.choices[0].message.content or "").strip()


# ===== 输出清洗 =====
def clean_llm_json_output(raw: str) -> str:
    """清洗 LLM 输出：
    1) 去除 ```json / ``` Markdown 包装；
    2) 截取第一个 '{' 到最后一个 '}' 之间的内容；
    3) 返回清洗后的字符串（即使没有 {} 也返回 strip 后的原始内容）。
    """
    if not raw:
        return ""
    t = raw.strip()
    # 去除任意位置的 ```json / ``` 标记
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    # 截取首 '{' 到末 '}' 之间
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return t


def safe_json_parse(raw: str) -> Optional[dict]:
    """清洗 + 解析。失败时打印错误并返回 None。"""
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


# ===== Schema 校验与补默认值 =====
def _ensure_type(value: Any, expected_default: Any) -> Any:
    """若 value 类型与默认值不一致，则返回默认值的深拷贝。"""
    if value is None:
        return json.loads(json.dumps(expected_default))
    if isinstance(expected_default, list) and not isinstance(value, list):
        return []
    if isinstance(expected_default, dict) and not isinstance(value, dict):
        return json.loads(json.dumps(expected_default))
    return value


def _coerce_int_or_none(value: Any) -> Optional[int]:
    """将字段统一转为 int 或 None。字符串会尝试抽取首个数字。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            try:
                return int(m.group())
            except ValueError:
                return None
    return None


def _coerce_bool_or_none(value: Any) -> Optional[bool]:
    """将字段统一转为 bool 或 None。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "是", "1"}:
            return True
        if v in {"false", "no", "否", "0"}:
            return False
    return None


def _dedup_keep_order(items: list) -> list:
    """列表去重并保持原顺序。"""
    seen = set()
    result = []
    for x in items:
        if not isinstance(x, str):
            continue
        key = x.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def validate_jd_profile(data: dict) -> dict:
    """对 LLM 输出做结构校验：
    - 顶层缺失键补默认值；
    - 类型不匹配回退；
    - location 子字段补齐；
    - 列表字段保证为 list；
    - 布尔字段保证为 bool 或 None；
    - risk_points 强制为对象列表（每项含 type/description/evidence）；
    - 枚举字段轻量校验。
    """
    if not isinstance(data, dict):
        data = {}

    # 1) 顶层补齐 & 类型校验
    result: dict = {}
    for key, default in DEFAULT_JD.items():
        result[key] = _ensure_type(data.get(key, default), default)

    # 2) location 子字段补齐
    loc_default = DEFAULT_JD["location"]
    loc_value = result["location"] if isinstance(result["location"], dict) else {}
    merged_loc = {**loc_default, **loc_value}
    for k, v in loc_default.items():
        if not isinstance(merged_loc.get(k), (str, type(None))):
            merged_loc[k] = v
        elif isinstance(merged_loc.get(k), str) and not merged_loc[k].strip():
            merged_loc[k] = None
    result["location"] = merged_loc

    # 3) 标量字段类型修正
    result["min_internship_months"] = _coerce_int_or_none(result.get("min_internship_months"))
    result["offers_internship"] = _coerce_bool_or_none(result.get("offers_internship"))

    # 4) risk_points 强制对象列表
    normalized_risks = []
    for item in result.get("risk_points", []):
        if isinstance(item, dict):
            normalized_risks.append({
                "type": item.get("type") if isinstance(item.get("type"), str) else None,
                "description": item.get("description") if isinstance(item.get("description"), str) else None,
                "evidence": item.get("evidence") if isinstance(item.get("evidence"), str) else None,
            })
        elif isinstance(item, str) and item.strip():
            # 字符串形式 → 转对象（type 暂留 None，留给 normalize_risk_points 修正）
            normalized_risks.append({"type": None, "description": item.strip(), "evidence": None})
    result["risk_points"] = normalized_risks

    # 5) 枚举字段轻量校验
    job_type_allowed = {"实习", "全职", "校招", "日常实习", "暑期实习"}
    if result["job_type"] not in job_type_allowed and result["job_type"] is not None:
        result["job_type"] = None
    education_allowed = {"本科", "硕士", "博士", "不限"}
    if result["education_level"] not in education_allowed and result["education_level"] is not None:
        result["education_level"] = None
    work_mode_allowed = {"现场", "远程", "混合"}
    if result["work_mode"] not in work_mode_allowed and result["work_mode"] is not None:
        result["work_mode"] = None

    # 6) 列表字段去重
    for key in ("responsibilities", "hard_skills", "soft_skills", "preferred_skills",
                "tools_or_frameworks", "domain_keywords", "benefits", "highlights",
                "academic_requirements", "competition_requirements", "evidence"):
        result[key] = _dedup_keep_order(result.get(key, []))

    return result


# ===== 规则后处理 1：实习字段修正 =====
def normalize_internship_fields(data: dict) -> dict:
    """将 internship_duration 中模糊表述（"可提供实习" 等且无明确时长）搬到 internship_note。"""
    duration = data.get("internship_duration")
    if isinstance(duration, str) and duration.strip():
        has_vague = any(re.search(p, duration) for p in INTERNSHIP_VAGUE_PATTERNS)
        has_duration = bool(INTERNSHIP_DURATION_HINT.search(duration))
        if has_vague and not has_duration:
            # 搬迁到 internship_note；保留原文
            if not data.get("internship_note"):
                data["internship_note"] = duration.strip()
            data["internship_duration"] = None
            data["min_internship_months"] = None
            data["weekly_days_requirement"] = None
            if data.get("offers_internship") is not True:
                data["offers_internship"] = True

    # internship_note 中若出现模糊实习表述，也将 offers_internship 置 True
    note = data.get("internship_note")
    if isinstance(note, str) and note.strip():
        if any(re.search(p, note) for p in INTERNSHIP_VAGUE_PATTERNS):
            if data.get("offers_internship") is not True:
                data["offers_internship"] = True

    return data


# ===== 规则后处理 2：技能字段修正 =====
def _is_soft_skill(text: str) -> bool:
    return any(kw in text for kw in SOFT_SKILL_KEYWORDS)


def _is_academic(text: str) -> bool:
    return any(kw in text for kw in ACADEMIC_KEYWORDS)


def _is_competition(text: str) -> bool:
    return any(kw in text for kw in COMPETITION_KEYWORDS)


def normalize_skills(data: dict) -> dict:
    """将软素质 / 学术要求 / 竞赛要求从 hard_skills 中迁出到对应字段。
    所有列表去重并保持原顺序。
    """
    hard = list(data.get("hard_skills") or [])
    soft = list(data.get("soft_skills") or [])
    academic = list(data.get("academic_requirements") or [])
    competition = list(data.get("competition_requirements") or [])

    kept_hard = []
    for item in hard:
        if not isinstance(item, str) or not item.strip():
            continue
        s = item.strip()
        if _is_soft_skill(s):
            soft.append(s)
        elif _is_academic(s):
            academic.append(s)
        elif _is_competition(s):
            competition.append(s)
        else:
            kept_hard.append(s)

    data["hard_skills"] = _dedup_keep_order(kept_hard)
    data["soft_skills"] = _dedup_keep_order(soft)
    data["academic_requirements"] = _dedup_keep_order(academic)
    data["competition_requirements"] = _dedup_keep_order(competition)
    return data


# ===== 规则后处理 2.5：技能原子化（split_compound_skill / normalize_jd_skills）=====
# 不可被 "/" 拆开的复合技能（大小写不敏感保护）
_PROTECTED_COMPOUND_SKILLS = ["C/C++", "C/C++/Python", "TCP/IP", "I/O"]

# 含「优先 / 加分」语义的触发词：命中则把该条整体归入 preferred_skills
_PREFERRED_SIGNAL_WORDS = (
    "优先", "加分", "更佳", "更好", "有经验者优先", "熟悉者优先",
    "满足以下任一", "任一即可", "有相关经验更",
)

# 拆分后需要从尾部剥离的噪声修饰词（按从长到短匹配）
_SKILL_NOISE_SUFFIXES = [
    "相关领域经验", "相关项目经验", "相关研究经验", "相关工作经验",
    "等核心技术", "核心技术", "相关经验", "实战经验", "项目经验",
    "编程能力", "开发能力", "建模能力", "设计能力",
    "相关领域", "等技术", "等经验", "等方向",
    "经验", "能力", "优先", "相关", "方向", "技术", "等",
]

# 无意义的停用词（拆分后若整体等于其中之一则丢弃）
_SKILL_STOPWORDS = {
    "经验", "能力", "相关", "优先", "等", "及", "与", "和", "或", "以上",
    "项目", "技术", "核心", "至少", "两项", "一项", "领域", "以及", "等等",
    "方向", "基础", "实战", "熟悉", "了解", "掌握", "精通",
}


def _strip_skill_noise(text: str) -> str:
    """剥离技能字符串尾部的噪声修饰词，例如「深度学习相关领域经验」→「深度学习」。"""
    s = text.strip()
    # 「等」之后通常是「等至少两项核心技术」之类的修饰，截断保留前半
    if "等" in s:
        head = s.split("等", 1)[0].strip()
        if head:
            s = head
    changed = True
    while changed:
        changed = False
        for suf in _SKILL_NOISE_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                s = s[: -len(suf)].strip()
                changed = True
                break
    return s


def _is_meaningful_skill(text: str) -> bool:
    """过滤过短 / 无意义的技能碎片。"""
    s = text.strip()
    if not s or s in _SKILL_STOPWORDS:
        return False
    # 纯中文至少 2 字；含英文 / 数字的允许长度 >= 2
    if len(s) < 2:
        return False
    return True


def split_compound_skill(skill: str) -> list:
    """将组合技能拆分为原子技能列表。

    支持分隔符：、 / 和 及 与 ; ； , ，；并保护 C/C++ 等不可拆的复合词，
    剥离「编程能力」「相关领域经验」「等核心技术」等尾部噪声，过滤无意义碎片。

    例：
      "深度学习及模仿学习相关领域经验" -> ["深度学习", "模仿学习"]
      "C/C++ 和 Python 编程能力"        -> ["C/C++", "Python"]
      "ROS、SLAM、运动规划等至少两项核心技术优先" -> ["ROS", "SLAM", "运动规划"]
    """
    if not isinstance(skill, str) or not skill.strip():
        return []
    text = skill.strip()

    # 1) 保护不可拆分的复合词
    placeholders: dict = {}
    for i, token in enumerate(_PROTECTED_COMPOUND_SKILLS):
        ph = f"\x00P{i}\x00"
        pat = re.compile(re.escape(token), re.IGNORECASE)
        if pat.search(text):
            text = pat.sub(ph, text)
            placeholders[ph] = token

    # 2) 去掉括号说明
    text = re.sub(r"[（(].*?[)）]", " ", text)

    # 3) 按分隔符拆分（不按普通空格拆，保留 "Isaac Gym" 这类带空格的工具名）
    parts = re.split(r"[、/和及与;；,，]+", text)

    results = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 还原被保护的复合词
        if p in placeholders:
            results.append(placeholders[p])
            continue
        p = _strip_skill_noise(p)
        if _is_meaningful_skill(p):
            results.append(p)

    return _dedup_keep_order(results)


def _has_preferred_signal(text: str) -> bool:
    """判断一条技能描述是否带「优先 / 加分」语义。"""
    return any(w in text for w in _PREFERRED_SIGNAL_WORDS)


def normalize_jd_skills(data: dict) -> dict:
    """技能原子化后处理：
    1. 拆分 hard_skills 中的长句为原子技能；
    2. 含「优先 / 加分 / 满足以下任一即可加分」语义的内容归入 preferred_skills；
    3. preferred_skills 同样原子化；
    4. 去重并保持顺序；
    5. preferred_skills 中剔除已属于 hard_skills 的项，避免重复。
    """
    hard_in = list(data.get("hard_skills") or [])
    pref_in = list(data.get("preferred_skills") or [])

    hard_out: list = []
    pref_out: list = []

    for item in hard_in:
        if not isinstance(item, str) or not item.strip():
            continue
        atoms = split_compound_skill(item)
        if _has_preferred_signal(item):
            # 带「优先 / 加分」语义的必备项实为加分项，移入 preferred
            pref_out.extend(atoms)
        else:
            hard_out.extend(atoms)

    for item in pref_in:
        if not isinstance(item, str) or not item.strip():
            continue
        pref_out.extend(split_compound_skill(item))

    hard_out = _dedup_keep_order(hard_out)
    hard_set = set(hard_out)
    pref_out = [s for s in _dedup_keep_order(pref_out) if s not in hard_set]

    data["hard_skills"] = hard_out
    data["preferred_skills"] = pref_out
    return data


# ===== 规则后处理 3：risk_points 病句修复 =====
# "需要强XXX经验" / "需要强XXX能力" → "XXX能力要求较高"
_BAD_RISK_PATTERN = re.compile(r"需要强([一-龥A-Za-z0-9 ]+?)(经验|能力|基础)")


def _fix_risk_description(text: str) -> str:
    """修正常见 LLM 病句。"""
    if not isinstance(text, str):
        return text
    s = text.strip()

    def _repl(m: re.Match) -> str:
        skill = m.group(1).strip()
        return f"{skill}能力要求较高"

    s = _BAD_RISK_PATTERN.sub(_repl, s)
    # 兜底：去除残留的「需要强」开头
    s = re.sub(r"^需要强\s*", "", s)
    return s


def normalize_risk_points(data: dict) -> dict:
    """保证 risk_points 是结构化对象列表；修正明显病句。"""
    raw_list = data.get("risk_points") or []
    fixed = []
    for item in raw_list:
        if isinstance(item, str):
            # 字符串兜底转对象
            fixed.append({
                "type": None,
                "description": _fix_risk_description(item),
                "evidence": None,
            })
            continue
        if not isinstance(item, dict):
            continue
        rp = {
            "type": item.get("type") if isinstance(item.get("type"), str) else None,
            "description": _fix_risk_description(item.get("description") or ""),
            "evidence": item.get("evidence") if isinstance(item.get("evidence"), str) and item.get("evidence").strip() else None,
        }
        if not rp["description"]:
            continue
        fixed.append(rp)

    # 同 description 去重
    seen = set()
    deduped = []
    for rp in fixed:
        key = (rp["description"] or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(rp)

    data["risk_points"] = deduped
    return data


# ===== 规则后处理 4：job_type 推断 =====
# 正式岗位特征：薪资范围 / 经验要求 N-M 年 / 年限要求
_FULLTIME_SIGNAL_PATTERNS = [
    r"薪资范围",
    r"\d+\s*[-~]\s*\d+\s*K",
    r"经验要求.*\d+\s*[-~]?\s*\d*\s*年",
    r"\d+\s*年.*经验",
    r"五险一金",
    r"正式员工",
]
_INTERN_TYPE_PATTERNS = [
    (r"日常实习", "日常实习"),
    (r"暑期实习", "暑期实习"),
    (r"寒假实习", "实习"),
    (r"校招", "校招"),
]


def infer_job_type(data: dict, jd_text: str) -> dict:
    """根据 JD 原文 + 已有字段推断 job_type / offers_internship。"""
    text = jd_text or ""

    # 1) 明确的实习类型关键词优先
    for pat, jt in _INTERN_TYPE_PATTERNS:
        if re.search(pat, text):
            # 若已经是更具体的类型则不覆盖；否则赋值
            if data.get("job_type") is None or data["job_type"] == "实习":
                data["job_type"] = jt
            # 暑期 / 日常 实习意味着提供实习
            if jt in {"实习", "日常实习", "暑期实习"}:
                if data.get("offers_internship") is not True:
                    data["offers_internship"] = True
            break

    # 2) 正式岗位特征：若文本含薪资范围 / 经验年限等，job_type 倾向于全职
    has_fulltime_signal = any(re.search(p, text) for p in _FULLTIME_SIGNAL_PATTERNS)
    has_internship_signal = (
        data.get("offers_internship") is True
        or any(re.search(p, text) for p in INTERNSHIP_VAGUE_PATTERNS)
    )

    if data.get("job_type") is None and has_fulltime_signal:
        data["job_type"] = "全职"

    # 3) 出现"可提供实习"模糊表达时，offers_internship 至少为 True
    if has_internship_signal and data.get("offers_internship") is None:
        data["offers_internship"] = True

    return data


# ===== 主流程 =====
def parse_jd(jd_text: str) -> dict:
    """主流程：LLM 抽取 -> 清洗 -> 解析 -> Schema 补齐 -> 规则后处理。失败返回 {}。"""
    if not jd_text or not jd_text.strip():
        print("[ERROR] JD 文本为空")
        return {}

    # 步骤 1：LLM 抽取
    user_prompt = build_user_prompt(jd_text)
    try:
        raw_output = call_llm(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        print(f"[ERROR] LLM 调用失败：{e}")
        return {}

    # 步骤 2：清洗 + JSON 解析
    parsed = safe_json_parse(raw_output)
    if parsed is None:
        return {}

    # 步骤 3：Schema 默认值补全
    profile = validate_jd_profile(parsed)

    # 步骤 4：规则后处理（顺序：实习字段 → 技能字段 → 技能原子化 → 风险点 → job_type）
    profile = normalize_internship_fields(profile)
    profile = normalize_skills(profile)
    profile = normalize_jd_skills(profile)  # 拆分长句技能、分流优先/加分项
    profile = normalize_risk_points(profile)
    profile = infer_job_type(profile, jd_text)

    return profile


def main():
    """命令行入口：python jd_parser.py <jd_text_path>
    解析结果同时打印到 stdout，并写入当前工作目录下的 `<jd_basename>.json`。
    """
    parser = argparse.ArgumentParser(description="JD Parser — 岗位 JD 文本转结构化 JSON")
    parser.add_argument("jd_text_path", help="JD 原文文本文件路径（UTF-8 编码）")
    parser.add_argument(
        "-o", "--output",
        help="输出 JSON 文件路径；默认保存到当前目录下 <jd_basename>.json",
        default=None,
    )
    args = parser.parse_args()

    if not os.path.isfile(args.jd_text_path):
        print(f"[ERROR] 文件不存在：{args.jd_text_path}")
        sys.exit(1)

    with open(args.jd_text_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    result = parse_jd(jd_text)
    json_str = json.dumps(result, ensure_ascii=False, indent=2)

    # 打印到 stdout
    print(json_str)

    # 写入当前工作目录（输入文件基名 + .json）
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.jd_text_path))[0]
        out_path = os.path.join(os.getcwd(), f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"\n[INFO] 结果已保存到：{out_path}")


if __name__ == "__main__":
    main()
