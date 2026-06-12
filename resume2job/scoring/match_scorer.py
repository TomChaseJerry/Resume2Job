"""
岗位匹配评分模块（Match Scorer）

输入：Resume Profile JSON + JD Profile JSON
输出：多维度评分 + 风险分析 + 总分 + 推荐等级 + 摘要

流程：
    1. Python 规则计算 skill_score
    2. Python 规则计算 education_score
    3. LLM 计算 project_score（rubric 打分）
    4. LLM 计算 direction_score（rubric 打分）
    5. Python 规则生成 risk_analysis（合并 JD 已有 risk_points）
    6. Python 加权计算 final_score
    7. Python 判断 match_level
    8. 生成简洁 summary
"""

import os
import re
import sys
import json
import argparse
import concurrent.futures
from typing import Optional, Any

# 复用 JD Parser 的技能原子化逻辑，保证必备技能在匹配前也被拆分为原子技能
from resume2job.parsing.jd_parser import split_compound_skill

# ===== 模型 / LLM 工具（统一走 core 层）=====
# 模型统一从 config 读取（含 RESUME2JOB_SCORE_MODEL 等环境变量覆盖逻辑）。
from resume2job.core.config import SCORE_MODEL as SCORE_MODEL_NAME, BASE_URL
from resume2job.core.llm import get_api_key, safe_json_parse


# ===== 评分权重 =====
DEFAULT_WEIGHTS = {
    "skill": 0.40,
    "project": 0.35,
    "education": 0.10,
    "direction": 0.15,
}


# ===== 学历等级 =====
DEGREE_RANK = {"本科": 1, "硕士": 2, "博士": 3}


# ===== 同义词表（双向同义：组内任一别名归一到首个 canonical form）=====
SYNONYM_GROUPS = [
    ["大模型", "大语言模型", "llm", "large language model"],
    ["rag", "检索增强生成", "向量检索"],
    ["agent", "智能体", "tool calling", "tool use"],
    ["pytorch", "torch"],
    ["tensorflow", "tf"],
    ["强化学习", "rl", "reinforcement learning"],
    ["模仿学习", "imitation learning"],
    ["元学习", "meta learning", "meta-learning"],
    ["分布式训练", "distributed training"],
    ["分布式强化学习", "distributed rl"],
    ["计算机视觉", "cv", "computer vision"],
    ["自然语言处理", "nlp", "natural language processing"],
    ["sft", "supervised fine-tuning", "有监督微调"],
    ["rlhf", "人类反馈强化学习"],
    ["langchain", "lang chain"],
    ["faiss", "向量数据库"],
    ["python", "py"],
    ["transformer", "transformers"],
    ["bert"],
    ["qwen", "千问", "通义千问"],
    ["llama", "llama2", "llama3"],
    ["sql", "mysql", "postgresql"],
    ["深度学习", "deep learning", "dl"],
    ["机器学习", "machine learning", "ml"],
    ["多模态", "multimodal", "multi-modal"],
    ["多模态算法", "多模态融合"],
    ["图神经网络", "gnn", "graph neural network"],
    ["推荐系统", "recommendation system", "recsys"],
    ["参数高效微调", "peft", "parameter efficient fine-tuning"],
    ["lora"],
    ["adapter"],
    ["lstm"],
    ["gru"],
    ["cnn"],
    ["rnn"],
    ["gat", "gatv2"],
    ["gcn"],
    ["moe", "mixture of experts", "transformer-moe"],
    ["算法工程化", "算法工程化经验", "工程化能力"],
    ["表征对齐", "representation alignment"],
]


def _build_synonym_map() -> dict:
    """构建 alias -> canonical 的小写映射。"""
    mapping = {}
    for group in SYNONYM_GROUPS:
        canonical = group[0].strip().lower()
        for alias in group:
            mapping[alias.strip().lower()] = canonical
    return mapping


SYNONYM_MAP = _build_synonym_map()


# ===== 技能蕴含表（单向：拥有 key 技能 ⇒ 隐含具备 value 列表中的能力）=====
# 用于把项目里出现的具体技术 / 模型扩展为更上层的能力标签
SKILL_IMPLICATIONS_RAW = {
    "lstm": ["深度学习"],
    "gru": ["深度学习"],
    "rnn": ["深度学习"],
    "cnn": ["深度学习", "计算机视觉"],
    "transformer": ["深度学习"],
    "transformer-moe": ["深度学习", "多模态算法", "moe"],
    "moe": ["深度学习"],
    "bert": ["深度学习", "自然语言处理", "transformer"],
    "gat": ["图神经网络", "深度学习"],
    "gatv2": ["图神经网络", "深度学习"],
    "gcn": ["图神经网络", "深度学习"],
    "gnn": ["深度学习"],
    "多模态融合": ["多模态算法", "多模态"],
    "多模态算法": ["多模态"],
    "lora": ["大模型", "参数高效微调"],
    "adapter": ["表征对齐", "多模态融合", "参数高效微调"],
    "qlora": ["大模型", "参数高效微调"],
    "pytorch": ["深度学习", "算法工程化"],
    "tensorflow": ["深度学习", "算法工程化"],
    "rlhf": ["强化学习", "大模型"],
    "sft": ["大模型"],
    "rag": ["大模型"],
    "langchain": ["大模型", "agent"],
    "agent": ["大模型"],
}


def _build_skill_implications() -> dict:
    """归一化蕴含表的 key 与 value（统一走 SYNONYM_MAP 的 canonical form）。"""
    norm: dict = {}
    for k, vs in SKILL_IMPLICATIONS_RAW.items():
        key = SYNONYM_MAP.get(k.lower(), k.lower())
        for v in vs:
            val = SYNONYM_MAP.get(v.lower(), v.lower())
            norm.setdefault(key, [])
            if val not in norm[key]:
                norm[key].append(val)
    return norm


SKILL_IMPLICATIONS = _build_skill_implications()


# ===== 自由文本扫描：用于从 project.tasks / project.description 中识别技能 =====
# 中文短语用子串匹配；英文用词边界正则匹配（避免 "rl" 误匹配 "control"）
_CHINESE_TERMS = sorted({
    k for k in SYNONYM_MAP.keys() if re.search(r"[一-鿿]", k)
}, key=len, reverse=True)
_ENGLISH_TERMS = sorted({
    k for k in SYNONYM_MAP.keys() if not re.search(r"[一-鿿]", k) and len(k) >= 3
}, key=len, reverse=True)
# 短英文词容易误匹配，单独维护需要词边界的白名单
_SHORT_ENGLISH_TERMS = {"rl", "ml", "dl", "cv", "tf", "py", "moe", "gnn", "gat", "cnn", "rnn", "gru", "bert"}


def _scan_text_for_skills(text: str) -> list:
    """从一段自由文本中扫描出已知技能词，返回 canonical 列表。"""
    if not isinstance(text, str) or not text.strip():
        return []
    low = text.lower()
    found = []
    seen = set()

    def _add(term: str):
        canonical = SYNONYM_MAP.get(term, term)
        if canonical not in seen:
            seen.add(canonical)
            found.append(canonical)

    for term in _CHINESE_TERMS:
        if term and term in low:
            _add(term)
    for term in _ENGLISH_TERMS:
        # 词边界匹配，避免 "py" 在 "pyramid" / "happy" 里被误识别
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
            _add(term)
    for term in _SHORT_ENGLISH_TERMS:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
            _add(term)
    return found


# ===== 工具函数：规范化技能 =====
def _normalize_skill(s: Any) -> Optional[str]:
    """转小写、去空格、按同义词表归一化。非字符串返回 None。"""
    if not isinstance(s, str):
        return None
    key = s.strip().lower()
    if not key:
        return None
    return SYNONYM_MAP.get(key, key)


def _dedup_keep_order(items: list) -> list:
    """列表去重并保持原顺序（仅对字符串生效）。"""
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


def _collect_resume_skills(resume_profile: dict) -> dict:
    """从 resume 多个字段综合收集候选人技能，返回 {normalized: original} 字典。

    采集来源：
      1) skills（顶层扁平）
      2) skill_groups[].items（兼容 list / dict 两种形态）
      3) projects[].tech_stack
      4) projects[].keywords
      5) projects[].tasks（自由文本，扫描已知技能词）
      6) projects[].description（自由文本）
      7) experiences[].keywords / responsibilities（兜底）
    然后通过 SKILL_IMPLICATIONS 做能力蕴含扩展（如 LSTM ⇒ 深度学习）。
    """
    raw_pool: list = []
    text_blobs: list = []

    # 1) 顶层 skills
    raw_pool.extend(resume_profile.get("skills") or [])

    # 2) skill_groups
    sg = resume_profile.get("skill_groups")
    if isinstance(sg, list):
        for group in sg:
            if isinstance(group, dict):
                raw_pool.extend(group.get("items") or [])
    elif isinstance(sg, dict):
        for items in sg.values():
            if isinstance(items, list):
                raw_pool.extend(items)

    # 3) & 4) & 5) & 6) projects
    for proj in resume_profile.get("projects") or []:
        if not isinstance(proj, dict):
            continue
        raw_pool.extend(proj.get("tech_stack") or [])
        raw_pool.extend(proj.get("keywords") or [])
        # 自由文本：tasks 列表 + description 字符串
        for t in proj.get("tasks") or []:
            if isinstance(t, str):
                text_blobs.append(t)
        for t in proj.get("achievements") or []:
            if isinstance(t, str):
                text_blobs.append(t)
        if isinstance(proj.get("description"), str):
            text_blobs.append(proj["description"])

    # 7) experiences
    for exp in resume_profile.get("experiences") or []:
        if not isinstance(exp, dict):
            continue
        raw_pool.extend(exp.get("keywords") or [])
        for t in exp.get("tasks") or []:
            if isinstance(t, str):
                text_blobs.append(t)

    # 先把显式 pool 做 canonical 归一
    mapping: dict = {}
    for raw in raw_pool:
        norm = _normalize_skill(raw)
        if norm and norm not in mapping:
            mapping[norm] = raw.strip() if isinstance(raw, str) else norm

    # 自由文本扫描 → 把识别到的 canonical 技能补入 mapping
    for blob in text_blobs:
        for canonical in _scan_text_for_skills(blob):
            if canonical not in mapping:
                mapping[canonical] = canonical  # 原文不可逐字回显，用 canonical 代替

    # 注意：能力蕴含（LSTM⇒深度学习 等）不在此处展开。蕴含属于「推导证据」，
    # 强于课程、弱于直接命中，应单独按 implied 权重计分，故由 _collect_implied_skills 单列，
    # 不再混入精确命中池（避免推导能力被当成 1.0 实战命中而虚高）。
    return mapping


def _collect_implied_skills(direct_map: dict) -> dict:
    """由直接技能蕴含出的上位能力（implied 证据，弱于直接命中、强于课程）。

    返回 {implied_canonical: "由 X 推导"} 映射；已在直接池出现的不再重复。
    与 _collect_resume_skills（直接 1.0）、_collect_course_skills（课程 0.5）分层，
    供 calculate_skill_score 按各自权重计分。
    """
    implied: dict = {}
    for key, src in direct_map.items():
        src_label = src if isinstance(src, str) else key
        for imp in SKILL_IMPLICATIONS.get(key, []):
            if imp not in direct_map and imp not in implied:
                implied[imp] = f"由 {src_label} 推导"
    return implied


def _collect_course_skills(resume_profile: dict) -> set:
    """从学历经历的课程/亮点（educations[].highlights、major）中扫描出已知技能词。

    课程属于「弱证据」：候选人学过但未必有项目实践。返回 canonical 技能集合，
    用于在精确命中之外补一档弱匹配，避免「简历课程里写了强化学习 / NLP / CV，
    却被判为完全缺失」。不与精确命中（resume skill map）混用，以免课程被当成 1.0 实战命中。
    """
    found: set = set()
    for edu in resume_profile.get("educations") or []:
        if not isinstance(edu, dict):
            continue
        blobs = list(edu.get("highlights") or [])
        if isinstance(edu.get("major"), str):
            blobs.append(edu["major"])
        for blob in blobs:
            if isinstance(blob, str):
                found.update(_scan_text_for_skills(blob))
    return found


def _atomize(items: list) -> list:
    """对一组技能字符串逐条做原子化拆分（兜底：即使 JD 未经 normalize_jd_skills 也能拆分长句）。"""
    out = []
    for it in items or []:
        if isinstance(it, str) and it.strip():
            out.extend(split_compound_skill(it))
    return _dedup_keep_order(out)


def _collect_jd_required_skills(jd_profile: dict) -> list:
    """JD 的**必备**技能 = hard_skills + tools_or_frameworks（原子化后）。
    domain_keywords 不参与必备命中（避免方向词被误判为缺口），由 _collect_jd_reference_skills 单独处理。
    """
    pool = []
    pool.extend(jd_profile.get("hard_skills") or [])
    pool.extend(jd_profile.get("tools_or_frameworks") or [])
    return _atomize(pool)


def _collect_jd_reference_skills(jd_profile: dict) -> list:
    """JD 的**辅助参考**技能（方向词），仅在命中时加分，未命中不计入 missing。"""
    return _atomize(jd_profile.get("domain_keywords") or [])


# ===== 弱匹配（weak match）：可迁移基础但非精确命中 =====
# 同一簇内的技能视为弱相关：某必备技能未精确命中，但候选人具备同簇能力时记为弱匹配。
WEAK_SKILL_CLUSTERS = [
    {"多传感器信息处理", "多传感器信息融合", "多源传感器建模", "多传感器融合",
     "多模态融合", "多模态", "传感器融合", "多源信息融合", "传感器建模"},
    {"深度学习", "神经网络", "deep learning"},
    {"运动控制", "运动规划", "轨迹规划", "轨迹生成", "机器人规划控制", "机器人运动学", "动力学建模"},
    {"强化学习", "模仿学习", "rl", "ppo", "sac"},
    {"动力学建模", "运动学建模", "建模", "时序建模", "时空建模"},
]


def _chinese_fragments(text: str, n: int = 3) -> set:
    """提取字符串中所有长度为 n 的中文连续片段，用于弱匹配（如「传感器」）。"""
    if not isinstance(text, str):
        return set()
    chinese = re.findall(r"[一-鿿]+", text)
    frags = set()
    for seg in chinese:
        if len(seg) >= n:
            for i in range(len(seg) - n + 1):
                frags.add(seg[i:i + n])
    return frags


# ===== 技能大类（用于约束「共享片段」弱匹配，避免只沾「算法 / 学习」等泛词就误判）=====
# 一个技能可命中多个大类；两个技能仅当**共享至少一个大类**时，才允许走「共享中文片段」弱匹配。
_SKILL_CATEGORY_KEYWORDS = {
    "感知融合": ["传感器", "感知", "多模态", "融合", "点云"],
    "视觉": ["视觉", "图像", "视频", "ocr", "检测", "分割"],
    "规划控制": ["规划", "控制", "轨迹", "导航", "运动", "定位", "slam", "飞控", "姿态", "动力学"],
    "学习算法": ["强化学习", "模仿学习", "元学习", "深度学习", "机器学习", "神经网络"],
    "检索搜索": ["搜索", "检索", "召回", "排序", "rag", "向量"],
    "推荐广告": ["推荐", "搜广推", "广告", "recsys"],
    "大模型": ["大模型", "llm", "agent", "智能体", "微调", "预训练", "对齐"],
    "运筹优化": ["运筹", "优化", "调度", "排程", "求解"],
    "自然语言": ["自然语言", "nlp", "文本", "语义", "对话"],
}


def _skill_categories(skill: str) -> set:
    """返回该技能命中的大类标签集合（按关键词子串）。无法归类时返回空集。"""
    if not isinstance(skill, str) or not skill.strip():
        return set()
    low = skill.lower()
    return {cat for cat, kws in _SKILL_CATEGORY_KEYWORDS.items()
            if any(kw in low for kw in kws)}


def _same_skill_category(a: str, b: str) -> bool:
    """两技能是否同属至少一个大类。任一无法归类则视为不同类（保守，避免泛词误匹配）。"""
    ca, cb = _skill_categories(a), _skill_categories(b)
    return bool(ca and cb and (ca & cb))


def _is_weak_match(required_norm: str, required_raw: str,
                   resume_norm_set: set, resume_raw_list: list) -> bool:
    """判断某必备技能是否与候选人技能构成弱匹配（同簇 or 同大类内的共享中文片段）。"""
    req_keys = {required_norm, (required_raw or "").strip().lower()}

    # 1) 同簇弱相关
    for cluster in WEAK_SKILL_CLUSTERS:
        cl = {c.lower() for c in cluster}
        if req_keys & cl:
            if resume_norm_set & cl:
                return True
            if any(isinstance(r, str) and r.strip().lower() in cl for r in resume_raw_list):
                return True

    # 2) 共享中文三元片段，但**必须同属一个技能大类**才算弱匹配。
    #    否则「搜索算法 vs 算法工程化」「控制算法 vs 推荐算法」会因共享「算法」等泛词被误判。
    req_frags = _chinese_fragments(required_raw, 3)
    if req_frags:
        for r in resume_raw_list:
            if not isinstance(r, str):
                continue
            if req_frags & _chinese_fragments(r, 3) and _same_skill_category(required_raw, r):
                return True
    return False


# ===== 技能证据来源权重（越直接越高；蕴含/课程/弱匹配为弱证据，避免虚高）=====
_W_DIRECT = 1.0        # 技能栏 / 项目技术栈 / 关键词 / 项目文本直接出现
_W_IMPLIED = 0.7       # 能力蕴含：由具体技术推导出的上位能力（如 LSTM⇒深度学习）
_W_COURSE = 0.5        # 课程级证据（学过但无项目实践）
_W_COURSE_CORE = 0.35  # 课程级证据，且该技能恰为岗位核心方向时进一步降权
_W_WEAK = 0.5          # 弱簇 / 同大类共享片段
_BONUS_CAP = 20.0      # preferred + domain 加分上限，防止加分项淹没核心缺口


def _is_core_direction_skill(skill_norm: str, jd_profile: dict) -> bool:
    """该技能是否与岗位核心方向（direction）高度一致。

    用于课程降权：学过「强化学习」课程 ≠ 能投强化学习算法岗，核心方向只给课程 0.35。
    """
    if not skill_norm:
        return False
    direction = str(jd_profile.get("direction") or "").strip().lower()
    if not direction:
        return False
    return skill_norm in direction or direction in skill_norm


# ===== 1. skill_score：Python 规则计算 =====
def calculate_skill_score(resume_profile: dict, jd_profile: dict) -> dict:
    """技能匹配评分（原子化 + 分层证据加权 + 加分上限）。

    每个必备技能按证据来源取最高一档权重：
      - 直接命中（技能栏/项目）           1.0
      - 能力蕴含（具体技术推出上位能力）   0.7
      - 课程级证据                         0.5（若恰为岗位核心方向则 0.35）
      - 弱簇 / 同大类共享片段              0.5
      - 缺口                               0
    base = Σ(命中权重) / required 总数 × 100；
    bonus = preferred×5 + domain×3，但封顶 _BONUS_CAP，且必备基础过弱时限制抬分，
    避免加分项淹没核心缺口（如有 Python+深度学习但无 RL 项目，仍不应被拉成高匹配）。
    """
    required_raw = _collect_jd_required_skills(jd_profile)
    preferred_raw = _atomize(jd_profile.get("preferred_skills") or [])
    domain_raw = _collect_jd_reference_skills(jd_profile)

    # JD 完全没有任何技能信号时给保底分
    if not required_raw and not preferred_raw and not domain_raw:
        return {
            "score": 80,
            "matched_skills": [],
            "weak_matched_skills": [],
            "implied_matched_skills": [],
            "missing_skills": [],
            "preferred_matched_skills": [],
            "evidence": ["JD未提供明确硬技能要求，按默认值给 80 分。"],
        }

    # 分层简历技能池：直接（1.0）/ 蕴含（0.7）/ 课程（0.5）
    resume_skill_map = _collect_resume_skills(resume_profile)
    resume_norm_set = set(resume_skill_map.keys())
    implied_skill_set = set(_collect_implied_skills(resume_skill_map).keys())
    course_skill_set = _collect_course_skills(resume_profile)
    # 候选人原始技能文本（含归一 key 与原始值），用于弱匹配片段比对
    resume_raw_list = list(resume_norm_set) + [
        v for v in resume_skill_map.values() if isinstance(v, str)
    ]

    # 必备技能逐项取最高一档权重：直接 > 蕴含 > 课程 > 弱匹配 > 缺口
    matched = []          # 直接命中 1.0
    implied_matched = []  # 蕴含 0.7
    weak_matched = []     # 课程 / 弱簇 / 同大类片段 0.35~0.5
    missing = []
    weight_sum = 0.0
    for raw in required_raw:
        norm = _normalize_skill(raw)
        if norm and norm in resume_norm_set:
            matched.append(raw)
            weight_sum += _W_DIRECT
        elif norm and norm in implied_skill_set:
            implied_matched.append(raw)
            weight_sum += _W_IMPLIED
        elif norm and norm in course_skill_set:
            w = _W_COURSE_CORE if _is_core_direction_skill(norm, jd_profile) else _W_COURSE
            weak_matched.append(raw)
            weight_sum += w
        elif _is_weak_match(norm or "", raw, resume_norm_set, resume_raw_list):
            weak_matched.append(raw)
            weight_sum += _W_WEAK
        else:
            missing.append(raw)

    # 优先技能命中
    preferred_matched = []
    for raw in preferred_raw:
        norm = _normalize_skill(raw)
        if norm and norm in resume_norm_set:
            preferred_matched.append(raw)

    # 方向 / 领域词命中（辅助参考，不计入 missing）
    domain_matched = []
    for raw in domain_raw:
        norm = _normalize_skill(raw)
        if norm and norm in resume_norm_set:
            domain_matched.append(raw)

    # 基础分：按各档证据权重之和归一
    if required_raw:
        base = (weight_sum / len(required_raw)) * 100.0
    elif preferred_raw or domain_raw:
        base = 60.0
    else:
        base = 80.0

    # 加分项：封顶，避免泛化优先词把分数拉过高
    bonus = min(5.0 * len(preferred_matched) + 3.0 * len(domain_matched), _BONUS_CAP)
    raw_score = base + bonus
    # 必备基础过弱（base<40）时，加分不得把岗位抬成高匹配（核心缺口不能被加分掩盖）
    if required_raw and base < 40.0:
        raw_score = min(raw_score, 60.0)
    score = int(round(max(0.0, min(100.0, raw_score))))

    # evidence
    evidence = []
    if required_raw:
        evidence.append(
            f"必备技能直接命中 {len(matched)}/{len(required_raw)}："
            f"[{', '.join(matched) if matched else '无'}]；"
            f"蕴含命中 [{', '.join(implied_matched) if implied_matched else '无'}]（0.7 权重）；"
            f"弱匹配（课程/可迁移）[{', '.join(weak_matched) if weak_matched else '无'}]；"
            f"核心缺口 [{', '.join(missing[:8]) if missing else '无'}]。"
        )
    if preferred_raw:
        evidence.append(
            f"优先 / 加分技能命中 {len(preferred_matched)}/{len(preferred_raw)}："
            f"[{', '.join(preferred_matched) if preferred_matched else '无'}]（加分封顶 {int(_BONUS_CAP)}）。"
        )
    if domain_raw:
        evidence.append(
            f"方向 / 领域弱匹配 {len(domain_matched)}/{len(domain_raw)}："
            f"[{', '.join(domain_matched) if domain_matched else '无'}]（不计入 missing）。"
        )

    # 蕴含与课程都属「有基础但非实战直接证据」，并入 weak_matched_skills 供下游
    # （skill_gap / risk / writer）一致地视为弱匹配，避免被误判为完全缺失。
    weak_for_downstream = _dedup_keep_order(weak_matched + implied_matched)

    return {
        "score": score,
        "matched_skills": matched,
        "weak_matched_skills": weak_for_downstream,
        "implied_matched_skills": implied_matched,
        "missing_skills": missing,
        "preferred_matched_skills": preferred_matched,
        "evidence": evidence,
    }


# ===== 2. education_score：Python 规则计算 =====
def calculate_education_score(resume_profile: dict, jd_profile: dict) -> dict:
    """学历评分。"""
    jd_level = jd_profile.get("education_level")
    resume_degree = resume_profile.get("highest_degree")

    # 1) JD 无学历要求或不限
    if jd_level is None or jd_level == "不限":
        return {
            "score": 90,
            "evidence": [f"JD 学历要求：{jd_level or '未提及'}，按默认值给 90 分。"],
        }

    jd_rank = DEGREE_RANK.get(jd_level)
    resume_rank = DEGREE_RANK.get(resume_degree) if resume_degree else None

    # 2) 无法判断任一方
    if jd_rank is None or resume_rank is None:
        return {
            "score": 60,
            "evidence": [
                f"无法判断学历匹配关系：JD 要求 '{jd_level}'，候选人最高学历 '{resume_degree}'。"
            ],
        }

    # 3) 候选人学历 >= JD 要求
    if resume_rank >= jd_rank:
        return {
            "score": 100,
            "evidence": [f"候选人学历 '{resume_degree}' 满足 JD 要求 '{jd_level}'。"],
        }

    # 4) 候选人学历不足
    return {
        "score": 0,
        "evidence": [f"候选人学历 '{resume_degree}' 低于 JD 要求 '{jd_level}'。"],
    }


# ===== 3 & 4. LLM 评分 =====
SYSTEM_PROMPT_SCORE = """你是一位资深招聘评估专家，请根据提供的评分标准（Rubric）对候选人和岗位的匹配程度进行打分。

要求：
1. 只能输出合法 JSON；
2. 不要输出 Markdown 代码块标记（如 ```json）；
3. 不要输出解释性前后缀；
4. score 必须是 0~100 的整数数字；
5. evidence 必须是字符串列表；
6. 不要编造简历或 JD 中没有的信息；
7. 如果信息不足，请给保守分数，并在 evidence 中说明原因。"""


PROJECT_RUBRIC = """【project_score Rubric】
- 0分：项目经历与岗位完全无关。
- 50分：项目经历与岗位略有相关，但缺少直接支撑。
- 75分：项目经历与岗位方向相关，能够支撑部分岗位职责。
- 100分：项目经历与岗位高度相关，技术栈、任务目标或业务场景高度一致。

【重要约束】
- 只能依据 resume 的 `projects` 字段评估项目相关性；
- **禁止**把 `job_preferences`（求职意向）、课程、研究方向、技能标签当作项目经历；
- evidence 描述项目时必须引用真实的项目名称与内容，禁止臆造「候选人有 NLP 项目」之类未在 projects 中出现的说法；
- 可以指出项目体现的可迁移能力（如多模态融合、图神经网络、多源传感器建模、时序建模、RAG/Agent 系统原型），
  但若岗位核心是机器人运动控制 / 轨迹规划 / 强化学习控制 / 仿真与真机部署，而 projects 中无直接证据，
  则相关性应保守评估（通常 40~55 分），不可因「同属 AI」而高估。
- 【应用层 vs 训练层·重要】当岗位核心包含「大模型训练 / 后训练 / RLHF / 预训练 / 底层算法优化 / 模型加速 /
  垂类大模型研发」等**训练层**职责，而候选人项目偏「大模型应用 / RAG / Agent 系统开发 / 推荐召回应用」等
  **应用层**时：即便方向高度相关，项目相关性也最高给 85~90，**不可给满分 100**——应用层项目无法完整支撑
  训练层职责，给 100 会高估。只有当 projects 中确有训练 / 微调 / 加速等直接证据时，才可接近 100。"""


DIRECTION_RUBRIC = """【direction_score Rubric】（请严格区分「泛 AI 相关」与「岗位核心方向一致」）
- 20分：仅有广义 AI / 机器学习背景，但与岗位核心方向缺少直接关系（弱相关）。
- 35分：有少量可迁移基础（如深度学习、多模态、多源传感器建模），但没有岗位核心方向的直接项目。
- 60分：在岗位核心方向（如机器人、强化学习、控制、规划、仿真）中已有部分直接项目。
- 100分：具备岗位核心方向的直接经验（如运动控制、轨迹规划、RL 控制、Sim2Real 或机器人真机部署）。

【重要约束】
- 对机器人 / 具身智能运动控制类岗位：若候选人没有机器人控制 / 轨迹规划 / 强化学习控制 / ROS / SLAM /
  机器人仿真平台 / 真机部署的直接项目，**最高不超过 35 分**，不得仅因同属 AI 而给到 60；
- 机器学习、多模态、GNN 等与具身智能仅属「弱相关」（约 20~35），不要判为「部分相关」(60)。"""


def _resume_subset_for_project(resume_profile: dict) -> dict:
    """裁剪给 project_score 用的 resume 字段，节省 token。"""
    return {
        "projects": resume_profile.get("projects") or [],
        "experiences": resume_profile.get("experiences") or [],
        "skills": resume_profile.get("skills") or [],
    }


def _resume_subset_for_direction(resume_profile: dict) -> dict:
    """裁剪给 direction_score 用的 resume 字段。"""
    return {
        "job_preferences": resume_profile.get("job_preferences") or {},
        "skills": resume_profile.get("skills") or [],
        "projects": [
            {
                "name": p.get("name"),
                "description": p.get("description"),
                "keywords": p.get("keywords") or [],
            }
            for p in (resume_profile.get("projects") or [])
            if isinstance(p, dict)
        ],
        "experiences": [
            {
                "company": e.get("company"),
                "title": e.get("title"),
                "keywords": e.get("keywords") or [],
            }
            for e in (resume_profile.get("experiences") or [])
            if isinstance(e, dict)
        ],
    }


def _jd_subset_for_project(jd_profile: dict) -> dict:
    return {
        "title": jd_profile.get("title"),
        "direction": jd_profile.get("direction"),
        "business_area": jd_profile.get("business_area"),
        "responsibilities": jd_profile.get("responsibilities") or [],
        "hard_skills": jd_profile.get("hard_skills") or [],
        "tools_or_frameworks": jd_profile.get("tools_or_frameworks") or [],
        "domain_keywords": jd_profile.get("domain_keywords") or [],
    }


def _jd_subset_for_direction(jd_profile: dict) -> dict:
    return {
        "title": jd_profile.get("title"),
        "direction": jd_profile.get("direction"),
        "business_area": jd_profile.get("business_area"),
        "domain_keywords": jd_profile.get("domain_keywords") or [],
        "responsibilities": jd_profile.get("responsibilities") or [],
    }


def _build_user_prompt_for_score(task_type: str, resume_subset: dict, jd_subset: dict) -> str:
    """构造打分 User Prompt。"""
    if task_type == "project_score":
        rubric = PROJECT_RUBRIC
        hint = ("请仅依据 projects 字段评估项目与岗位职责 / 方向 / 业务场景的相关性；"
                "不要把求职意向、课程或技能标签当作项目经历。")
    else:
        rubric = DIRECTION_RUBRIC
        hint = ("请聚焦候选人技能方向 / 项目方向与岗位核心方向的一致性，"
                "严格区分『泛 AI 相关』与『岗位核心方向一致』；缺少岗位核心方向直接项目时从严打分。")

    resume_str = json.dumps(resume_subset, ensure_ascii=False, indent=2)
    jd_str = json.dumps(jd_subset, ensure_ascii=False, indent=2)

    return (
        f"【评分任务】{task_type}\n\n"
        f"{rubric}\n\n"
        f"{hint}\n\n"
        "===== 候选人相关信息 =====\n"
        f"{resume_str}\n\n"
        "===== 岗位相关信息 =====\n"
        f"{jd_str}\n\n"
        "===== 输出 JSON Schema =====\n"
        '{\n  "score": 0~100 的整数,\n  "evidence": ["原因1", "原因2"]\n}\n\n'
        "请直接输出 JSON 对象本体，不要任何额外文字或 Markdown 包装。"
    )


def validate_llm_score_result(data: Optional[dict], default_score: int = 50) -> dict:
    """校验 LLM 评分结果：score 限制到 [0,100] 整数；evidence 强制为 list。"""
    if not isinstance(data, dict):
        return {"score": default_score, "evidence": ["LLM 输出无效，使用默认保守分数。"]}

    # score 规范化
    raw_score = data.get("score")
    score: int
    if isinstance(raw_score, bool):
        score = default_score
    elif isinstance(raw_score, (int, float)):
        score = int(round(float(raw_score)))
    elif isinstance(raw_score, str):
        m = re.search(r"\d+", raw_score)
        score = int(m.group()) if m else default_score
    else:
        score = default_score
    score = max(0, min(100, score))

    # evidence 规范化
    raw_evidence = data.get("evidence")
    if isinstance(raw_evidence, list):
        evidence = [str(x).strip() for x in raw_evidence if isinstance(x, (str, int, float)) and str(x).strip()]
    elif isinstance(raw_evidence, str) and raw_evidence.strip():
        evidence = [raw_evidence.strip()]
    else:
        evidence = []
    if not evidence:
        evidence = ["LLM 未返回 evidence，默认填充。"]

    return {"score": score, "evidence": evidence}


def call_llm_score(task_type: str, resume_profile: dict, jd_profile: dict) -> dict:
    """LLM 评分入口。task_type ∈ {project_score, direction_score}。失败返回保守分数。"""
    if task_type not in {"project_score", "direction_score"}:
        raise ValueError(f"unsupported task_type: {task_type}")

    # 1) 构造裁剪后的输入
    if task_type == "project_score":
        resume_subset = _resume_subset_for_project(resume_profile)
        jd_subset = _jd_subset_for_project(jd_profile)
        default_score = 50
    else:
        resume_subset = _resume_subset_for_direction(resume_profile)
        jd_subset = _jd_subset_for_direction(jd_profile)
        # 方向分默认值改为 30：LLM 失败时给「弱相关」保守分，避免回退到偏高的 60
        default_score = 30

    user_prompt = _build_user_prompt_for_score(task_type, resume_subset, jd_subset)

    # 2) 调 LLM，捕获异常返回保守分
    try:
        api_key = get_api_key()
    except RuntimeError:
        print("[ERROR] 环境变量 DASHSCOPE_API_KEY 未设置，使用保守分数。")
        return {"score": default_score, "evidence": ["未配置 API Key，使用保守分数。"]}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        completion = client.chat.completions.create(
            model=SCORE_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SCORE},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            # 关闭思考模式：评分输出仅 score + evidence，无需推理过程，省时省钱。
            # 仅 Qwen3/flash 等混合推理模型识别该参数，旧模型会忽略。
            extra_body={"enable_thinking": False},
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[ERROR] LLM 调用失败（{task_type}）：{e}")
        return {"score": default_score, "evidence": [f"LLM 调用异常：{e}，使用保守分数。"]}

    # 3) 解析 + 校验
    parsed = safe_json_parse(raw)
    return validate_llm_score_result(parsed, default_score=default_score)


# ===== 5. risk_analysis：Python 规则生成 =====
_EXPERIENCE_YEAR_PATTERN = re.compile(r"(\d+)\s*[-~]?\s*(\d+)?\s*年")

# 高信号技能缺口类别（领域启发式）：命中则单列为独立风险类型，便于按优先级展示。
# 每个类别附带 context_keywords：仅当 **JD 本身确实属于该领域**（方向/业务/标题/领域词命中）时才启用，
# 否则一律回退到通用「技能缺口」。这样可避免「搜索算法 / 采样算法」这类多义词在大模型、
# 推荐、运筹等岗位里被误标成「机器人运动控制能力缺口」（典型的模板串岗污染）。
# signal/context 关键词均为小写；signal 与 _normalize_skill 归一形式比对。
_SKILL_GAP_CATEGORIES = [
    ("机器人运动控制能力缺口",
     {"机器人规划控制", "运动控制", "运动规划", "轨迹规划", "轨迹生成",
      "导航", "机器人运动学", "动力学建模", "搜索算法", "采样算法"},
     {"机器人", "具身", "运动控制", "规划控制", "轨迹", "导航", "无人机",
      "自动驾驶", "robot", "embodied", "控制算法", "运动规划"}),
    ("强化学习/模仿学习能力缺口",
     {"强化学习", "模仿学习", "ppo", "sac", "rlhf"},
     {"强化学习", "模仿学习", "reinforcement", " rl ", "具身", "机器人",
      "决策", "控制", "智能体决策", "多智能体"}),
    ("机器人仿真与ROS/SLAM能力缺口",
     {"isaac gym", "mujoco", "pybullet", "ros", "slam", "sim2real",
      "嵌入式系统开发", "嵌入式"},
     {"机器人", "具身", "slam", "ros", "仿真", "sim2real", "robot",
      "自动驾驶", "无人机", "嵌入式", "真机"}),
]


def _jd_domain_context(jd_profile: dict) -> str:
    """把 JD 的方向 / 业务场景 / 标题 / 领域关键词拼成一段小写文本，
    用于判断某个专项缺口类别是否适用于该岗位（领域门控）。"""
    parts = [
        str(jd_profile.get("direction") or ""),
        str(jd_profile.get("business_area") or ""),
        str(jd_profile.get("title") or ""),
        " ".join(str(x) for x in (jd_profile.get("domain_keywords") or [])),
    ]
    return " ".join(parts).lower()


def _make_risk(type_: str, description: str, evidence: Optional[str] = None) -> dict:
    return {"type": type_, "description": description, "evidence": evidence}


_DEGREE_RISK_TYPES = {"学历门槛", "学历要求"}


def _risk_richness(r: dict) -> int:
    """风险条目的信息量评分：有 evidence 加权 + description 长度。用于同类去重时选优。"""
    desc_len = len(r.get("description") or "")
    has_evidence = bool(r.get("evidence"))
    return (10 if has_evidence else 0) + desc_len


def generate_risk_analysis(
    resume_profile: dict,
    jd_profile: dict,
    skill_score_result: dict,
    education_score_result: dict,
) -> list:
    """根据规则 + JD 中已有 risk_points 生成结构化风险列表。

    去重策略：
      - 同 (type, description) 完全重复：保留首条；
      - 同 type 不同 description：保留信息量更高的一条（优先有 evidence、其次 description 更长）。
      - 候选人已满足学历要求时，过滤掉所有 type 为「学历门槛」的风险（包括 JD 自带）。
    """
    risks: list = []
    edu_score = education_score_result.get("score", 0)
    edu_satisfied = edu_score == 100  # 候选人学历完全满足 JD 要求

    # 1) 学历门槛：仅在候选人确实不满足时才触发
    jd_level = jd_profile.get("education_level")
    resume_degree = resume_profile.get("highest_degree")
    if jd_level in {"硕士", "博士"} and edu_score == 0 and not edu_satisfied:
        risks.append(_make_risk(
            type_="学历门槛",
            description=f"岗位要求 {jd_level}，候选人最高学历为 {resume_degree or '未知'}，存在学历差距。",
            evidence=jd_profile.get("education_requirement"),
        ))

    # 2) 经验门槛
    exp_req = jd_profile.get("experience_requirement")
    if isinstance(exp_req, str) and exp_req.strip():
        m = _EXPERIENCE_YEAR_PATTERN.search(exp_req)
        if m:
            min_year = int(m.group(1))
            if min_year >= 1:
                risks.append(_make_risk(
                    type_="经验门槛",
                    description=f"岗位要求约 {min_year} 年及以上工作经验，对在校 / 应届候选人门槛较高。",
                    evidence=exp_req.strip(),
                ))

    # 3) 科研产出
    academic_reqs = jd_profile.get("academic_requirements") or []
    if academic_reqs:
        pubs = resume_profile.get("publications") or []
        if not pubs:
            risks.append(_make_risk(
                type_="科研产出",
                description="岗位对论文 / 会议 / 专利等学术产出有明确要求，候选人当前未提供相关产出。",
                evidence="；".join([str(x) for x in academic_reqs if isinstance(x, str)]) or None,
            ))

    # 4) 竞赛经历
    comp_reqs = jd_profile.get("competition_requirements") or []
    if comp_reqs:
        comps = resume_profile.get("competitions") or []
        if not comps:
            risks.append(_make_risk(
                type_="竞赛经历",
                description="岗位提及竞赛 / 获奖偏好，候选人当前未提供相关经历。",
                evidence="；".join([str(x) for x in comp_reqs if isinstance(x, str)]) or None,
            ))

    # 5) 技能缺口：先按高信号类别归类（便于风险提示优先展示核心缺口），再补一条通用缺口
    missing = skill_score_result.get("missing_skills") or []
    if missing:
        missing_norm = {(_normalize_skill(m) or str(m).strip().lower()): m for m in missing
                        if isinstance(m, str)}
        categorized_raw = set()  # 已被某个类别覆盖的原始技能，避免在通用缺口里重复
        jd_ctx = _jd_domain_context(jd_profile)  # 领域门控上下文

        for cat_type, signal_set, context_keywords in _SKILL_GAP_CATEGORIES:
            # 领域门控：JD 本身不属于该领域时，跳过该专项类别（缺口仍会进通用「技能缺口」），
            # 避免把多义词（如「搜索算法」）在非机器人岗位里错标成机器人运动控制缺口。
            if not any(ck in jd_ctx for ck in context_keywords):
                continue
            hit_raw = [orig for norm, orig in missing_norm.items() if norm in signal_set]
            if hit_raw:
                categorized_raw.update(hit_raw)
                risks.append(_make_risk(
                    type_=cat_type,
                    description=f"候选人缺少{cat_type.replace('能力缺口', '')}相关直接经验："
                                f"{', '.join(hit_raw[:6])}。",
                    evidence=None,
                ))

        # 其余未归类的缺口，合并成一条通用「技能缺口」
        rest = [m for m in missing if isinstance(m, str) and m not in categorized_raw]
        if rest:
            risks.append(_make_risk(
                type_="技能缺口",
                description=f"候选人在以下硬技能上存在缺口：{', '.join(rest[:5])}。",
                evidence=None,
            ))

    # 6) 合并 JD 已有结构化 risk_points
    for rp in jd_profile.get("risk_points") or []:
        if not isinstance(rp, dict):
            continue
        type_ = rp.get("type") if isinstance(rp.get("type"), str) else "其他"
        description = rp.get("description")
        if not (isinstance(description, str) and description.strip()):
            continue
        # 候选人已满足学历时，跳过 JD 自带的学历类风险
        if edu_satisfied and type_ in _DEGREE_RISK_TYPES:
            continue
        evidence = rp.get("evidence") if isinstance(rp.get("evidence"), str) and rp.get("evidence").strip() else None
        risks.append(_make_risk(type_=type_ or "其他", description=description.strip(), evidence=evidence))

    # 去重 1：完全相同的 (type, description) 仅保留首条
    seen = set()
    unique = []
    for r in risks:
        key = (r["type"], r["description"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    # 去重 2：同 type 多条 description 时，按信息量选优
    by_type: dict = {}
    order: list = []
    for r in unique:
        t = r["type"]
        if t not in by_type:
            by_type[t] = r
            order.append(t)
        else:
            if _risk_richness(r) > _risk_richness(by_type[t]):
                by_type[t] = r

    deduped = [by_type[t] for t in order]
    return deduped


# ===== 6 & 7. 总分与等级 =====
def calculate_final_score(
    skill_score: int,
    project_score: int,
    education_score: int,
    direction_score: int,
) -> int:
    """加权计算 final_score，四舍五入到整数并限制 [0, 100]。"""
    raw = (
        skill_score * DEFAULT_WEIGHTS["skill"]
        + project_score * DEFAULT_WEIGHTS["project"]
        + education_score * DEFAULT_WEIGHTS["education"]
        + direction_score * DEFAULT_WEIGHTS["direction"]
    )
    return max(0, min(100, int(round(raw))))


def judge_match_level(final_score: int) -> str:
    """final_score >= 75 -> recommended；50 <= s < 75 -> maybe；< 50 -> not_recommended。"""
    if final_score >= 75:
        return "recommended"
    if final_score >= 50:
        return "maybe"
    return "not_recommended"


# ===== 8. summary 生成 =====
def generate_summary(result: dict) -> str:
    """简洁说明是否推荐 + 主要优势 + 主要风险，不超过 100 字。"""
    level = result.get("match_level")
    final = result.get("final_score", 0)
    level_text = {
        "recommended": "推荐投递",
        "maybe": "可酌情投递",
        "not_recommended": "暂不建议投递",
    }.get(level, "暂无结论")

    # 主要优势：取最高分维度
    dim_scores = {
        "技能": result.get("skill_score", {}).get("score", 0),
        "项目": result.get("project_score", {}).get("score", 0),
        "学历": result.get("education_score", {}).get("score", 0),
        "方向": result.get("direction_score", {}).get("score", 0),
    }
    top_dim = max(dim_scores.items(), key=lambda kv: kv[1])
    bottom_dim = min(dim_scores.items(), key=lambda kv: kv[1])

    # 主要风险：取 risk_analysis 中首项 type
    risks = result.get("risk_analysis") or []
    risk_text = risks[0].get("type") if risks and isinstance(risks[0], dict) else None

    parts = [f"{level_text}（总分 {final}）"]
    parts.append(f"优势：{top_dim[0]}({top_dim[1]})")
    if bottom_dim[1] < 60:
        parts.append(f"短板：{bottom_dim[0]}({bottom_dim[1]})")
    if risk_text:
        parts.append(f"主要风险：{risk_text}")

    summary = "；".join(parts) + "。"
    # 截断到 100 字以内
    if len(summary) > 100:
        summary = summary[:99] + "…"
    return summary


# ===== 主函数 =====
def score_match(resume_profile: dict, jd_profile: dict) -> dict:
    """完整评分流程。"""
    # 1. 技能分（规则）
    skill_result = calculate_skill_score(resume_profile, jd_profile)
    # 2. 学历分（规则）
    education_result = calculate_education_score(resume_profile, jd_profile)
    # 3 & 4. 项目分 / 方向分：两次独立 LLM 调用，彼此无依赖，并发执行以省去一次串行等待。
    # call_llm_score 内部各自新建 client、无共享可变状态，线程安全。
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _ex:
        _f_project = _ex.submit(call_llm_score, "project_score", resume_profile, jd_profile)
        _f_direction = _ex.submit(call_llm_score, "direction_score", resume_profile, jd_profile)
        project_result = _f_project.result()
        direction_result = _f_direction.result()
    # 5. 风险分析（规则 + JD risk_points 合并）
    risks = generate_risk_analysis(resume_profile, jd_profile, skill_result, education_result)
    # 6. 总分（规则加权）
    final_score = calculate_final_score(
        skill_result["score"], project_result["score"],
        education_result["score"], direction_result["score"],
    )
    # 7. 等级
    match_level = judge_match_level(final_score)

    result = {
        "skill_score": skill_result,
        "project_score": project_result,
        "education_score": education_result,
        "direction_score": direction_result,
        "risk_analysis": risks,
        "final_score": final_score,
        "match_level": match_level,
        "summary": "",
    }
    # 8. summary
    result["summary"] = generate_summary(result)
    return result


# ===== CLI =====
def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """命令行入口：python match_scorer.py resume_profile.json jd_profile.json"""
    parser = argparse.ArgumentParser(description="Match Scorer — 简历与 JD 的多维匹配评分")
    parser.add_argument("resume_json", help="Resume Profile JSON 文件路径")
    parser.add_argument("jd_json", help="JD Profile JSON 文件路径")
    parser.add_argument(
        "-o", "--output",
        help="输出 JSON 文件路径；默认保存到当前目录下 match_<resume>__<jd>.json",
        default=None,
    )
    args = parser.parse_args()

    for path in (args.resume_json, args.jd_json):
        if not os.path.isfile(path):
            print(f"[ERROR] 文件不存在：{path}")
            sys.exit(1)

    # 读取两个 JSON
    try:
        resume_profile = _load_json_file(args.resume_json)
        jd_profile = _load_json_file(args.jd_json)
    except json.JSONDecodeError as e:
        print(f"[ERROR] 输入 JSON 解析失败：{e}")
        sys.exit(1)

    if not isinstance(resume_profile, dict) or not isinstance(jd_profile, dict):
        print("[ERROR] 输入文件顶层必须是 JSON 对象")
        sys.exit(1)

    result = score_match(resume_profile, jd_profile)
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_str)

    # 写入当前目录
    if args.output:
        out_path = args.output
    else:
        r_base = os.path.splitext(os.path.basename(args.resume_json))[0]
        j_base = os.path.splitext(os.path.basename(args.jd_json))[0]
        out_path = os.path.join(os.getcwd(), f"match_{r_base}__{j_base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"\n[INFO] 结果已保存到：{out_path}")


if __name__ == "__main__":
    main()
