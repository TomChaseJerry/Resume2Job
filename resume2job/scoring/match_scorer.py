"""
岗位匹配评分模块（Match Scorer）

输入：Resume Profile JSON + JD Profile JSON
输出：两层评分（基础适配分 + 排序分）+ 风险分析 + 推荐等级 + 摘要

两层评分（见 job_matching_and_ranking.md）：
    match_score = skill_score×0.55 + project_score×0.45            # 基础适配分
    rank_score  = min(100, match_score + direction_bonus + commute_bonus)   # 排序分
其中 direction_bonus ≤ 6、commute_bonus ≤ 4；城市 / 学历 / 岗位类型属硬约束（召回前过滤），
不参与评分。

流程：
    1. Python 规则计算 skill_score（分层证据 + 可替代必备）
    2. LLM 计算 project_score（5 档 rubric 打分）——候选池**惰性评分**（with_project=False）时
       跳过此步、用 skill 作 match_score 代理；展示批次由 apply_project_scores_batch **1 次批量
       LLM**（1 简历项目摘要 + N 精简 JD → N 分，逐岗独立、禁止横向比较）精算，省 LLM 调用次数
    3. Python 加权得到 match_score
    4. Python 规则计算 direction_bonus（方向标签映射，确定性）
    5. Python 规则计算 commute_bonus（通勤时长分级；评分阶段通常为 0，增强阶段回填）
    6. Python 计算 rank_score 与推荐等级 match_level
    7. Python 规则生成 risk_analysis（含学历门槛，合并 JD 已有 risk_points）
    8. 生成简洁 summary
"""

import os
import re
import sys
import json
import argparse
from typing import Optional, Any

# 复用 JD Parser 的技能原子化逻辑，保证必备技能在匹配前也被拆分为原子技能
from resume2job.parsing.jd_parser import split_compound_skill, DEGREE_RANK

# ===== 模型 / LLM 工具（统一走 core 层）=====
# 模型统一从 config 读取（含 RESUME2JOB_SCORE_MODEL 等环境变量覆盖逻辑）。
from resume2job.core.config import SCORE_MODEL as SCORE_MODEL_NAME, BASE_URL
from resume2job.core.llm import get_api_key, safe_json_parse
# 方向标签词表（粗粒度方向层，仅供 direction_bonus；单独成文件便于扩词，见 direction_tags.py）
from resume2job.scoring.direction_tags import DIRECTION_TAG_VOCAB, DIRECTION_PARENTS
# 技能分类法词表（同义词/蕴含/弱匹配簇/大类/缺口类别）——细粒度技能匹配层，见 skill_taxonomy.py
from resume2job.scoring.skill_taxonomy import (
    SYNONYM_MAP, SKILL_IMPLICATIONS, WEAK_SKILL_CLUSTERS,
    SKILL_CATEGORY_KEYWORDS, SKILL_GAP_CATEGORIES,
)


# ===== 基础适配分权重（仅技能 + 项目两维）=====
DEFAULT_WEIGHTS = {
    "skill": 0.55,
    "project": 0.45,
}

# ===== 偏好加分上限（rank_score = match_score + direction_bonus + commute_bonus）=====
DIRECTION_BONUS_CAP = 6.0   # 岗位方向偏好加分上限
COMMUTE_BONUS_CAP = 4.0     # 通勤偏好加分上限


# 学历等级 DEGREE_RANK（仅用于风险分析的学历门槛判定，不参与评分）从 jd_parser 单一事实源导入（见顶部 import）。


# 同义词表 SYNONYM_GROUPS/SYNONYM_MAP 与技能蕴含表 SKILL_IMPLICATIONS_RAW/SKILL_IMPLICATIONS
# 已移至 scoring/skill_taxonomy.py（见顶部 import）。


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


def _collect_resume_skills(resume_profile: dict) -> tuple:
    """从 resume 多个字段综合收集候选人技能，返回 (skill_map, demonstrated_norms)。

    skill_map         ：{normalized: original} 全部技能（仅声明 ∪ 已佐证）。
    demonstrated_norms：在**项目 / 实习 / 工作经历**里实际出现的技能归一名集合（强证据）。

    证据强弱分层（job_matching_and_ranking.md §3 / 用户 2026-06-21 要求）：
      - **已佐证**（项目 tech_stack/keywords/tasks/description、经历 keywords/tasks）→ 直接证据 1.0；
      - **仅声明**（只在技能栏 skills / skill_groups 出现、经历里查无）→ 弱证据 0.7（status=weak）。
    采集来源：
      1) skills（顶层扁平，**仅声明**）
      2) skill_groups[].items（兼容 list / dict，**仅声明**）
      3) projects[].tech_stack / keywords（**已佐证**）
      4) projects[].tasks / achievements / description（自由文本扫描，**已佐证**）
      5) experiences[].keywords / tasks（**已佐证**）
    能力蕴含（LSTM⇒深度学习）不在此处展开，见 _collect_implied_skills。
    """
    declared_pool: list = []      # 仅声明：技能栏 + skill_groups
    demonstrated_pool: list = []  # 已佐证：项目 / 经历的结构化技能字段
    text_blobs: list = []         # 已佐证：项目 / 经历自由文本

    # 1) 顶层 skills（仅声明）
    declared_pool.extend(resume_profile.get("skills") or [])

    # 2) skill_groups（仅声明）
    sg = resume_profile.get("skill_groups")
    if isinstance(sg, list):
        for group in sg:
            if isinstance(group, dict):
                declared_pool.extend(group.get("items") or [])
    elif isinstance(sg, dict):
        for items in sg.values():
            if isinstance(items, list):
                declared_pool.extend(items)

    # 3) & 4) projects（已佐证）
    for proj in resume_profile.get("projects") or []:
        if not isinstance(proj, dict):
            continue
        demonstrated_pool.extend(proj.get("tech_stack") or [])
        demonstrated_pool.extend(proj.get("keywords") or [])
        # 自由文本：tasks / achievements 列表 + description 字符串
        for t in proj.get("tasks") or []:
            if isinstance(t, str):
                text_blobs.append(t)
        for t in proj.get("achievements") or []:
            if isinstance(t, str):
                text_blobs.append(t)
        if isinstance(proj.get("description"), str):
            text_blobs.append(proj["description"])

    # 5) experiences（已佐证）
    for exp in resume_profile.get("experiences") or []:
        if not isinstance(exp, dict):
            continue
        demonstrated_pool.extend(exp.get("keywords") or [])
        for t in exp.get("tasks") or []:
            if isinstance(t, str):
                text_blobs.append(t)

    mapping: dict = {}
    demonstrated: set = set()

    # 已佐证先入：结构化字段 + 自由文本扫描，记入 demonstrated 集合（强证据 1.0）
    for raw in demonstrated_pool:
        norm = _normalize_skill(raw)
        if norm:
            if norm not in mapping:
                mapping[norm] = raw.strip() if isinstance(raw, str) else norm
            demonstrated.add(norm)
    for blob in text_blobs:
        for canonical in _scan_text_for_skills(blob):
            if canonical not in mapping:
                mapping[canonical] = canonical  # 原文不可逐字回显，用 canonical 代替
            demonstrated.add(canonical)

    # 仅声明后补：只在技能栏出现、经历查无的技能不计入 demonstrated（弱证据 0.7）
    for raw in declared_pool:
        norm = _normalize_skill(raw)
        if norm and norm not in mapping:
            mapping[norm] = raw.strip() if isinstance(raw, str) else norm

    # 注意：能力蕴含（LSTM⇒深度学习 等）不在此处展开。蕴含属于「推导证据」，
    # 强于课程、弱于直接命中，应单独按 implied 权重计分，故由 _collect_implied_skills 单列，
    # 不再混入精确命中池（避免推导能力被当成 1.0 实战命中而虚高）。
    return mapping, demonstrated


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


# ===== 可替代必备需求（「Python / C++ 其一」命中任一即满足）=====
# 仅当原始技能条目带这些显式「任选其一」语义时，才把其原子拆分视为可替代组；
# 「及 / 和 / 与」等并列连词不在此列（那是「都要」，仍各自独立必备）。
_ALT_REQUIREMENT_SIGNALS = ("其一", "任一", "二选一", "二选其一", "任选", "之一", "其中之一", "任意一")


# ===== 通用基础能力（不主导技能分）=====
# Python/PyTorch/Linux/Git/C++/SQL/Docker 等语言/框架/开发环境区分度低、许多岗位仅要求若干项，
# 不应作为「核心方向必备」主导技能分（scenario_overview §3 / job_matching_and_ranking.md §3.1）。
# 它们单列做「基础能力校验」：命中小幅加分、未命中小幅扣分并提示，但不淹没核心方向匹配。
_GENERAL_BASE_SKILLS = {
    _normalize_skill(s) for s in
    ["python", "pytorch", "tensorflow", "linux", "git", "c++", "c", "c/c++",
     "sql", "docker", "numpy", "pandas", "shell", "bash", "java"]
    if _normalize_skill(s)
}


def _is_base_skill(skill: str) -> bool:
    """该技能是否属于通用基础能力。"""
    n = _normalize_skill(skill)
    return bool(n) and n in _GENERAL_BASE_SKILLS


def _collect_required_units(jd_profile: dict) -> tuple:
    """收集必备「需求单元」，并区分核心方向技能与通用基础能力。

    返回 (core_units, base_units)，每个 unit 是一组原子技能（list）：
      - 单元素 [skill]       ：普通必备（命中该技能才满足）；
      - 多元素 [s1, s2, ...]  ：可替代组（命中任一即满足，其余不算缺口）。
    core_units = 核心方向 / 工程技能（主导技能分）；base_units = 通用基础能力（不主导，仅小幅加减）。

    可替代组来源：jd_profile.required_alternatives（解析层产出）+ 原始 hard_skill 条目「其一」兜底。
    domain_keywords 不计入必备（由 _collect_jd_reference_skills 处理）。
    """
    alt_groups: list = []
    grouped_norm: set = set()

    def _add_group(atoms: list) -> None:
        atoms = _dedup_keep_order(atoms)
        if len(atoms) < 2:
            return
        key = frozenset(_normalize_skill(a) or a.strip().lower() for a in atoms)
        if any(frozenset(_normalize_skill(x) or x.strip().lower() for x in g) == key for g in alt_groups):
            return
        alt_groups.append(atoms)
        for a in atoms:
            n = _normalize_skill(a)
            if n:
                grouped_norm.add(n)

    # 1) 解析层给出的可替代组（首选）
    for grp in jd_profile.get("required_alternatives") or []:
        _add_group(_atomize(grp) if isinstance(grp, list) else _atomize([grp]))

    pool = list(jd_profile.get("hard_skills") or []) + list(jd_profile.get("tools_or_frameworks") or [])

    # 2) 兜底：原始条目仍含「其一」时识别为组（解析层未填字段、或被直接调用时）
    for raw in pool:
        if isinstance(raw, str) and any(sig in raw for sig in _ALT_REQUIREMENT_SIGNALS):
            cleaned = raw
            for sig in sorted(_ALT_REQUIREMENT_SIGNALS, key=len, reverse=True):
                cleaned = cleaned.replace(sig, " ")
            cleaned = cleaned.replace("或", "/")
            atoms = split_compound_skill(cleaned)
            if len(atoms) >= 2:
                _add_group(atoms)

    # 3) 单项必备（排除已在可替代组里的原子）
    singles: list = []
    seen: set = set(grouped_norm)
    for raw in pool:
        if not isinstance(raw, str):
            continue
        # 含「其一」信号的条目其原子已由步骤 1/2 的可替代组覆盖，跳过以免「C++ 其一」等残片漏成单项
        if any(sig in raw for sig in _ALT_REQUIREMENT_SIGNALS):
            continue
        for atom in split_compound_skill(raw):
            n = _normalize_skill(atom)
            if n and n not in seen:
                seen.add(n)
                singles.append([atom])

    # 4) 分类：整组都是基础能力 -> base；否则 -> core
    core_units, base_units = [], []
    for unit in alt_groups + singles:
        if all(_is_base_skill(a) for a in unit):
            base_units.append(unit)
        else:
            core_units.append(unit)
    return core_units, base_units


def _collect_jd_reference_skills(jd_profile: dict) -> list:
    """JD 的**辅助参考**技能（方向词），仅在命中时加分，未命中不计入 missing。"""
    return _atomize(jd_profile.get("domain_keywords") or [])


# 弱匹配「同簇」表 WEAK_SKILL_CLUSTERS 已移至 scoring/skill_taxonomy.py（见顶部 import）。


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


# 技能大类护栏 SKILL_CATEGORY_KEYWORDS 已移至 scoring/skill_taxonomy.py（见顶部 import）。


def _skill_categories(skill: str) -> set:
    """返回该技能命中的大类标签集合（按关键词子串）。无法归类时返回空集。"""
    if not isinstance(skill, str) or not skill.strip():
        return set()
    low = skill.lower()
    return {cat for cat, kws in SKILL_CATEGORY_KEYWORDS.items()
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
_W_DIRECT = 1.0        # 已佐证：项目 / 实习 / 工作经历里实际出现（技术栈 / 关键词 / 经历文本）
_W_DECLARED = 0.7      # 仅声明：只在技能栏 / skill_groups 出现、经历里查无佐证（status=weak）
_W_IMPLIED = 0.7       # 能力蕴含：由具体技术推导出的上位能力（如 LSTM⇒深度学习）
_W_COURSE = 0.5        # 课程级证据（学过但无项目实践）
_W_COURSE_CORE = 0.35  # 课程级证据，且该技能恰为岗位核心方向时进一步降权
_W_WEAK = 0.5          # 弱簇 / 同大类共享片段
_BONUS_CAP = 10.0      # preferred + domain 加分上限（再按核心覆盖率缩放），防止加分项淹没核心缺口

# 证据档 → skill_gap 三态（status 的唯一权威：规则层决定，LLM 只叙述不决定）
# declared（仅声明未佐证）与 implied/weak 同归 weak：候选人列了该技能但无项目实证。
_BUCKET_TO_STATUS = {"matched": "matched", "declared": "weak", "implied": "weak", "weak": "weak", "missing": "missing"}

# 通用基础能力对技能分的影响（小幅、封顶，不主导核心方向匹配，见 §3.1）
_W_BASE_HIT = 2.0      # 每命中一项 JD 必备基础能力的加分
_BASE_HIT_CAP = 8.0    # 基础能力命中加分上限
_W_BASE_MISS = 4.0     # 每未命中一项 JD 必备基础能力的扣分
_BASE_MISS_CAP = 12.0  # 基础能力缺口扣分上限


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


def _evaluate_required_atom(raw: str, resume_norm_set: set, demonstrated_set: set,
                            implied_skill_set: set, course_skill_set: set,
                            resume_raw_list: list, jd_profile: dict) -> tuple:
    """对单个必备原子技能取最高一档证据权重，返回 (weight, bucket)。

    bucket ∈ {matched(已佐证 1.0) / declared(仅声明 0.7) / implied(蕴含 0.7) /
              weak(课程/弱匹配 0.35~0.5) / missing(0)}。
    已佐证（demonstrated_set）= 经历/项目里实际出现；仅声明 = 在技能栏出现但经历查无。
    """
    norm = _normalize_skill(raw)
    if norm and norm in demonstrated_set:
        return _W_DIRECT, "matched"
    if norm and norm in resume_norm_set:          # 仅声明（在简历技能池但非已佐证）
        return _W_DECLARED, "declared"
    if norm and norm in implied_skill_set:
        return _W_IMPLIED, "implied"
    if norm and norm in course_skill_set:
        w = _W_COURSE_CORE if _is_core_direction_skill(norm, jd_profile) else _W_COURSE
        return w, "weak"
    if _is_weak_match(norm or "", raw, resume_norm_set, resume_raw_list):
        return _W_WEAK, "weak"
    return 0.0, "missing"


# ===== 1. skill_score：Python 规则计算 =====
def calculate_skill_score(resume_profile: dict, jd_profile: dict) -> dict:
    """技能匹配评分（核心方向技能主导 + 通用基础能力小幅加减 + 可替代必备 + 加分上限）。

    每个必备「需求单元」按证据来源取最高一档权重（可替代组取组内任一最高）：
      已佐证命中 1.0 / 仅声明 0.7 / 能力蕴含 0.7 / 课程级 0.5（核心方向 0.35）/ 弱匹配 0.5 / 缺口 0。
      （已佐证=项目/实习/工作经历里实际出现；仅声明=只在技能栏出现、经历查无 → status=weak）

    分层（job_matching_and_ranking.md §3.1 + scenario_overview §3）：
      - **核心方向技能（core_units）主导技能分**：base = Σ(core 单元命中权重) / core 单元数 × 100；
      - **通用基础能力（base_units，Python/PyTorch/Linux/Git…）不主导**：命中小幅加分（≤+8）、
        未命中小幅扣分（≤−12）并计入缺口提示；
      - 若 JD 只列基础能力（无 core）：基础能力才作为主信号；
      - 「Python / C++ 其一」可替代必备：命中任一即满足、其余不计缺口（见 _collect_required_units）；
      - preferred×5 + domain×3 加分封顶 _BONUS_CAP；核心基础过弱（base<40）时限制加分抬分。
    """
    core_units, base_units = _collect_required_units(jd_profile)
    preferred_raw = _atomize(jd_profile.get("preferred_skills") or [])
    domain_raw = _collect_jd_reference_skills(jd_profile)

    # JD 完全没有任何技能信号时给保底分
    if not core_units and not base_units and not preferred_raw and not domain_raw:
        return {
            "score": 80,
            "matched_skills": [],
            "weak_matched_skills": [],
            "implied_matched_skills": [],
            "missing_skills": [],
            "preferred_matched_skills": [],
            "evidence": ["JD未提供明确硬技能要求，按默认值给 80 分。"],
        }

    # 分层简历技能池：已佐证（1.0）/ 仅声明（0.7）/ 蕴含（0.7）/ 课程（0.5）
    resume_skill_map, demonstrated_set = _collect_resume_skills(resume_profile)
    resume_norm_set = set(resume_skill_map.keys())
    implied_skill_set = set(_collect_implied_skills(resume_skill_map).keys())
    course_skill_set = _collect_course_skills(resume_profile)
    # 候选人原始技能文本（含归一 key 与原始值），用于弱匹配片段比对
    resume_raw_list = list(resume_norm_set) + [
        v for v in resume_skill_map.values() if isinstance(v, str)
    ]

    matched = []          # 直接命中 1.0
    implied_matched = []  # 蕴含 0.7
    weak_matched = []     # 课程 / 弱簇 / 同大类片段 0.35~0.5
    missing = []
    # 逐技能状态明细（规则层权威，供 skill_gap 直接用、避免与 LLM 判定不一致）
    skill_status: dict = {}

    def _mark_status(unit: list, bucket: str, importance: str) -> None:
        """把单元（含可替代组所有原子）标成同一 status（组满足则成员都不算缺口）。"""
        st = _BUCKET_TO_STATUS.get(bucket, "missing")
        for a in unit:
            n = _normalize_skill(a)
            if n and n not in skill_status:
                skill_status[n] = {"label": a, "importance": importance, "status": st}

    def _eval_unit(unit: list) -> tuple:
        """可替代组取组内命中证据最强的原子，返回 (best_w, best_bucket, best_atom)。"""
        best_w, best_bucket, best_atom = 0.0, "missing", unit[0]
        for atom in unit:
            w, bucket = _evaluate_required_atom(
                atom, resume_norm_set, demonstrated_set, implied_skill_set,
                course_skill_set, resume_raw_list, jd_profile,
            )
            if w > best_w:
                best_w, best_bucket, best_atom = w, bucket, atom
        return best_w, best_bucket, best_atom

    def _record(bucket: str, atom: str, label: str) -> None:
        if bucket == "matched":
            matched.append(atom)
        elif bucket == "implied":
            implied_matched.append(atom)
        elif bucket in ("weak", "declared"):   # 仅声明并入弱匹配展示（status=weak）
            weak_matched.append(atom)
        else:
            missing.append(label)

    # 1) 核心方向技能：主导技能分
    core_weight = 0.0
    for unit in core_units:
        w, bucket, atom = _eval_unit(unit)
        core_weight += w
        _record(bucket, atom, "/".join(unit) if len(unit) > 1 else unit[0])
        _mark_status(unit, bucket, "must")

    # 2) 通用基础能力：单独统计命中 / 缺口（不进 core 分母）
    base_weight = 0.0
    base_hit = base_miss = 0
    for unit in base_units:
        w, bucket, atom = _eval_unit(unit)
        base_weight += w
        label = "/".join(unit) if len(unit) > 1 else unit[0]
        if w > 0:
            base_hit += 1
            _record(bucket, atom, label)   # 基础能力命中也计入匹配，供下游展示
        else:
            base_miss += 1
            missing.append(label)          # 未命中基础必备：计入缺口提示
        _mark_status(unit, bucket, "must")

    # 优先技能：命中（直接）计入加分；并按证据档为每项记 status（供 skill_gap 展示）
    preferred_matched = []
    for raw in preferred_raw:
        norm = _normalize_skill(raw)
        if norm and norm in resume_norm_set:
            preferred_matched.append(raw)
        w, bucket = _evaluate_required_atom(
            raw, resume_norm_set, demonstrated_set, implied_skill_set,
            course_skill_set, resume_raw_list, jd_profile,
        )
        _mark_status([raw], bucket, "preferred")

    # 方向 / 领域词命中（辅助参考，不计入 missing）
    domain_matched = []
    for raw in domain_raw:
        norm = _normalize_skill(raw)
        if norm and norm in resume_norm_set:
            domain_matched.append(raw)

    # 基础分：核心方向技能主导；基础能力仅小幅加减、不主导
    n_core = len(core_units)
    n_base = len(base_units)
    if n_core:
        base = (core_weight / n_core) * 100.0
        base += min(_BASE_HIT_CAP, base_hit * _W_BASE_HIT)
        base -= min(_BASE_MISS_CAP, base_miss * _W_BASE_MISS)
        base = max(0.0, min(100.0, base))
    elif n_base:
        # JD 仅列通用基础能力：此时基础能力作为主信号
        base = (base_weight / n_base) * 100.0
    elif preferred_raw or domain_raw:
        base = 60.0
    else:
        base = 80.0

    # 加分项：封顶后**按核心覆盖率缩放**——核心方向覆盖弱时加分按比例缩水，
    # 平滑地防止「只命中泛优先词 / 领域词却被加分抬成高匹配」（替代旧的 base<40 生硬封顶）。
    bonus = min(5.0 * len(preferred_matched) + 3.0 * len(domain_matched), _BONUS_CAP)
    # 核心覆盖率：有 core 用其命中率；无 core 但有 base（纯工具型 JD）给满；
    # 纯 preferred/domain（既无 core 也无 base）→ 0，不让加分项把 base=60 抬高（加分不主导）。
    core_coverage = (core_weight / n_core) if n_core else (1.0 if n_base else 0.0)
    bonus *= core_coverage
    raw_score = base + bonus
    score = int(round(max(0.0, min(100.0, raw_score))))

    # evidence
    evidence = []
    if n_core:
        evidence.append(
            f"核心方向必备 {n_core} 项："
            f"直接命中 [{', '.join(matched) if matched else '无'}]；"
            f"蕴含 [{', '.join(implied_matched) if implied_matched else '无'}]（0.7）；"
            f"弱匹配 [{', '.join(weak_matched) if weak_matched else '无'}]；"
            f"核心缺口 [{', '.join(missing[:8]) if missing else '无'}]。"
        )
    if n_base:
        evidence.append(
            f"通用基础能力命中 {base_hit}/{n_base}（命中加分≤{int(_BASE_HIT_CAP)}、缺口扣分≤{int(_BASE_MISS_CAP)}，不主导技能分）。"
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
        # 逐技能状态明细（规则权威）：{归一技能: {label, importance(must/preferred), status(matched/weak/missing)}}
        # skill_gap 直接据此定 status，LLM 只补叙述，保证 skill_score 与 skill_gap 一致。
        "skill_status": skill_status,
        "evidence": evidence,
    }


# ===== 2. education_gate：学历门槛三态（不计入评分）=====
def calculate_education_score(resume_profile: dict, jd_profile: dict) -> dict:
    """学历门槛三态 gate（**不计入匹配分**，仅供风险提示与报告语气）。

    返回 {"gate": "satisfied"|"indeterminate"|"insufficient", "evidence": [...]}：
      - satisfied    ：JD 无学历要求 / 不限，或候选人学历 ≥ 要求；
      - indeterminate：JD 或候选人学历无法判断（按规范不硬过滤，报告标「待确认」）；
      - insufficient ：候选人学历明确低于 JD 要求。
    学历本是召回前的硬约束（jobs_store.get_eligible_jobs 资格预筛），此函数只为穿过过滤的
    岗位（无要求 / 不可判定）在报告里标注门槛状态，不参与评分（原 0~100 score 是旧加权模型遗留）。
    """
    jd_level = jd_profile.get("education_level")
    resume_degree = resume_profile.get("highest_degree")

    # 1) JD 无学历要求或不限 → 无门槛
    if jd_level is None or jd_level == "不限":
        return {"gate": "satisfied",
                "evidence": [f"JD 学历要求：{jd_level or '未提及'}，无学历门槛。"]}

    jd_rank = DEGREE_RANK.get(jd_level)
    resume_rank = DEGREE_RANK.get(resume_degree) if resume_degree else None

    # 2) 任一方无法判断 → 待确认（不硬过滤）
    if jd_rank is None or resume_rank is None:
        return {"gate": "indeterminate",
                "evidence": [f"无法判断学历匹配关系：JD 要求 '{jd_level}'，候选人最高学历 '{resume_degree}'，报告标待确认。"]}

    # 3) 候选人学历 >= JD 要求 → 满足
    if resume_rank >= jd_rank:
        return {"gate": "satisfied",
                "evidence": [f"候选人学历 '{resume_degree}' 满足 JD 要求 '{jd_level}'。"]}

    # 4) 候选人学历不足 → 门槛不满足
    return {"gate": "insufficient",
            "evidence": [f"候选人学历 '{resume_degree}' 低于 JD 要求 '{jd_level}'。"]}


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


PROJECT_RUBRIC = """【project_score Rubric】（5 档：0 / 25 / 50 / 75 / 100）
- 0分：无相关项目或经历证据。
- 25分：仅有泛领域相关性（例如都属于 AI，但任务本身无关）。
- 50分：存在部分任务、方法或技术重叠，但核心职责不同。
- 75分：项目任务、方法或业务场景高度相关，能支撑部分岗位职责。
- 100分：项目直接覆盖岗位的核心工作内容与关键技术。

【重要约束】
- 只能依据 resume 的 `projects`（含科研 / 实习经历）评估项目相关性；
- **禁止**把 `job_preferences`（求职意向）、课程、研究方向、技能标签当作项目经历；
- evidence 描述项目时必须引用真实的项目名称与内容，禁止臆造「候选人有 NLP 项目」之类未在 projects 中出现的说法；
- 泛 AI 相关 ≠ 岗位核心方向匹配；应用层项目 ≠ 训练层项目；
- RAG、Agent 等应用项目不能直接等价于预训练、后训练、RLHF 或强化学习项目；
- 若岗位核心是机器人控制 / 强化学习 / 轨迹规划 / SLAM / 真机部署，而 projects 中无对应证据，
  只能给低分或中低分（通常 ≤50）；
- 【应用层 vs 训练层】岗位核心为大模型训练 / 后训练 / RLHF / 预训练 / 模型加速等**训练层**职责，
  而候选人项目偏「大模型应用 / RAG / Agent / 推荐召回应用」等**应用层**时：即便方向高度相关也最高给 75，
  **不可给 100**；只有 projects 中确有训练 / 微调 / 加速等直接证据时才可接近 100；
- 评分结果需在 evidence 中返回匹配证据、主要缺口与简短理由。"""


def _resume_subset_for_project(resume_profile: dict) -> dict:
    """裁剪给 project_score 用的 resume 字段，节省 token。"""
    return {
        "projects": resume_profile.get("projects") or [],
        "experiences": resume_profile.get("experiences") or [],
        "skills": resume_profile.get("skills") or [],
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


def _build_user_prompt_for_score(resume_subset: dict, jd_subset: dict) -> str:
    """构造 project_score 打分 User Prompt。"""
    rubric = PROJECT_RUBRIC
    hint = ("请仅依据 projects（含科研 / 实习）字段评估项目与岗位职责 / 方向 / 业务场景的相关性；"
            "不要把求职意向、课程或技能标签当作项目经历；缺少岗位核心方向直接项目时从严打分。")

    resume_str = json.dumps(resume_subset, ensure_ascii=False, indent=2)
    jd_str = json.dumps(jd_subset, ensure_ascii=False, indent=2)

    return (
        "【评分任务】project_score\n\n"
        f"{rubric}\n\n"
        f"{hint}\n\n"
        "===== 候选人相关信息 =====\n"
        f"{resume_str}\n\n"
        "===== 岗位相关信息 =====\n"
        f"{jd_str}\n\n"
        "===== 输出 JSON Schema =====\n"
        '{\n  "score": 0/25/50/75/100 之一,\n  "evidence": ["匹配证据", "主要缺口", "简短理由"]\n}\n\n'
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


def call_llm_score(resume_profile: dict, jd_profile: dict) -> dict:
    """LLM 项目匹配度评分（project_score，5 档 rubric）。方向不再走 LLM（改 compute_direction_bonus）。失败返回保守分数。"""
    # 1) 构造裁剪后的输入
    resume_subset = _resume_subset_for_project(resume_profile)
    jd_subset = _jd_subset_for_project(jd_profile)
    default_score = 50

    user_prompt = _build_user_prompt_for_score(resume_subset, jd_subset)

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
        print(f"[ERROR] LLM 调用失败（project_score）：{e}")
        return {"score": default_score, "evidence": [f"LLM 调用异常：{e}，使用保守分数。"]}

    # 3) 解析 + 校验
    parsed = safe_json_parse(raw)
    return validate_llm_score_result(parsed, default_score=default_score)


# ===== 3b. 批量 project_score（1 简历项目摘要 + N 精简 JD → 1 次 LLM → N 分，逐岗独立）=====
def _build_user_prompt_for_score_batch(resume_subset: dict, jd_subsets: list) -> str:
    """构造**批量** project_score 打分 User Prompt：候选人项目摘要只放一份，N 个岗位各自独立打分。

    关键约束：逐岗独立、**禁止横向比较**（不得因其他岗位高/低而调整某岗位分）。
    """
    resume_str = json.dumps(resume_subset, ensure_ascii=False, indent=2)
    jobs_blocks = []
    for i, jd in enumerate(jd_subsets, start=1):
        jobs_blocks.append(f"--- 岗位 id={i} ---\n" + json.dumps(jd, ensure_ascii=False, indent=2))
    jobs_str = "\n\n".join(jobs_blocks)
    n = len(jd_subsets)

    return (
        "【评分任务】project_score（批量，逐岗独立）\n\n"
        f"{PROJECT_RUBRIC}\n\n"
        "【独立评分要求（必须严格遵守）】\n"
        "请分别独立评估每个岗位与候选人项目经历的相关性。\n"
        "不得因其他岗位得分高或低而调整某岗位分数。\n"
        "每个岗位只能依据该岗位职责、核心技能与候选人项目证据评分。\n\n"
        "===== 候选人项目经历（所有岗位共用，仅此一份）=====\n"
        f"{resume_str}\n\n"
        f"===== 待评分岗位（共 {n} 个，逐个独立打分）=====\n"
        f"{jobs_str}\n\n"
        "===== 输出 JSON Schema =====\n"
        '{\n'
        '  "scores": [\n'
        '    {"id": 1, "score": 0/25/50/75/100 之一, "evidence": ["匹配证据", "主要缺口", "简短理由"]}\n'
        '    // ... 每个岗位一项，id 与上面岗位对应 ...\n'
        '  ]\n'
        '}\n\n'
        f"必须为上面全部 {n} 个岗位各返回一项（id 与岗位一一对应），共 {n} 项；"
        "直接输出 JSON 对象本体，不要任何额外文字或 Markdown 包装。"
    )


def _distribute_batch_scores(parsed, n: int) -> list:
    """把批量 LLM 输出还原成长度 n、按岗位顺序对齐的 [{score, evidence}]。

    优先按 id 映射（容忍 LLM 调序）；无可用 id 但数量吻合则按顺序兜底；缺失/多余/畸形一律补保守默认。
    每项复用 validate_llm_score_result 做 score/evidence 规范化。
    """
    default = {"score": 50, "evidence": ["批量评分未覆盖该岗位，使用保守分数。"]}
    out = [None] * n
    if isinstance(parsed, dict):
        items = parsed.get("scores")
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = None

    if isinstance(items, list):
        # 1) 优先按 id（1-based）映射，容忍乱序
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                idx = int(it.get("id")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n and out[idx] is None:
                out[idx] = validate_llm_score_result(it, default_score=50)
        # 2) 无可用 id 但条数吻合 → 按顺序兜底
        if all(o is None for o in out) and len(items) == n:
            for i, it in enumerate(items):
                if isinstance(it, dict):
                    out[i] = validate_llm_score_result(it, default_score=50)

    return [o if o is not None else dict(default) for o in out]


def call_llm_score_batch(resume_profile: dict, jd_profiles: list) -> list:
    """**批量** LLM 项目评分：1 个候选人项目摘要 + N 个精简 JD → **1 次 LLM 调用** → N 个 project_score。

    逐岗独立打分（prompt 明确禁止横向比较）。返回与 jd_profiles 等长、一一对应的 [{score, evidence}]。
    任何失败 → 全部回退保守默认（与 call_llm_score 一致），**绝不抛**（单次调用顶替原 N 次/岗）。
    """
    n = len(jd_profiles)
    default = {"score": 50, "evidence": ["批量项目评分不可用，使用保守分数。"]}
    if n == 0:
        return []

    resume_subset = _resume_subset_for_project(resume_profile)
    jd_subsets = [_jd_subset_for_project(jd if isinstance(jd, dict) else {}) for jd in jd_profiles]

    try:
        api_key = get_api_key()
    except RuntimeError:
        print("[ERROR] 环境变量 DASHSCOPE_API_KEY 未设置，使用保守分数（批量 project_score）。")
        return [dict(default) for _ in range(n)]

    user_prompt = _build_user_prompt_for_score_batch(resume_subset, jd_subsets)
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
            extra_body={"enable_thinking": False},
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[ERROR] LLM 调用失败（project_score 批量）：{e}")
        return [dict(default) for _ in range(n)]

    parsed = safe_json_parse(raw)
    return _distribute_batch_scores(parsed, n)


# ===== 5. risk_analysis：Python 规则生成 =====
_EXPERIENCE_YEAR_PATTERN = re.compile(r"(\d+)\s*[-~]?\s*(\d+)?\s*年")

# 高信号技能缺口类别 SKILL_GAP_CATEGORIES（含领域门控）已移至 scoring/skill_taxonomy.py（见顶部 import）。


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
    education_gate_result: dict,
) -> list:
    """根据规则 + JD 中已有 risk_points 生成结构化风险列表。

    去重策略：
      - 同 (type, description) 完全重复：保留首条；
      - 同 type 不同 description：保留信息量更高的一条（优先有 evidence、其次 description 更长）。
      - 候选人已满足学历要求时，过滤掉所有 type 为「学历门槛」的风险（包括 JD 自带）。
    """
    risks: list = []
    edu_gate = education_gate_result.get("gate")
    edu_satisfied = edu_gate == "satisfied"  # 候选人学历满足 JD 要求（或无门槛）

    # 1) 学历门槛：候选人学历明确低于岗位要求即触发（insufficient 已含专科→本科等差距）
    jd_level = jd_profile.get("education_level")
    resume_degree = resume_profile.get("highest_degree")
    if edu_gate == "insufficient":
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

        for cat_type, signal_set, context_keywords in SKILL_GAP_CATEGORIES:
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


# ===== 6. 基础适配分 match_score（技能 + 项目两维加权）=====
def calculate_match_score(skill_score: int, project_score: int) -> int:
    """match_score = skill×0.55 + project×0.45，四舍五入到整数并限制 [0, 100]。"""
    raw = (
        skill_score * DEFAULT_WEIGHTS["skill"]
        + project_score * DEFAULT_WEIGHTS["project"]
    )
    return max(0, min(100, int(round(raw))))


# ===== 6.1 方向偏好加分 direction_bonus（确定性标签映射，取代旧 direction LLM 评分）=====
# 方向标签词表 DIRECTION_TAG_VOCAB / DIRECTION_PARENTS 见 scoring/direction_tags.py（便于扩词，
# 含「不放泛词 / 一词只归一标签」两条边界说明）。JD 与用户偏好都映射到标签集合后比较匹配程度。
def _kw_in_text(kw: str, low: str) -> bool:
    """关键词命中判断：纯英文 / 缩写用**词边界**匹配（避免 rag⊂storage、bev⊂beverage 等子串误命中），
    含中文的关键词用子串匹配（中文无词边界）。low 已是小写并首尾加空格。"""
    k = kw.strip().lower()
    if not k:
        return False
    if k.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", low) is not None
    return k in low


def _map_direction_tags(text: str) -> set:
    """把一段文本映射到方向标签集合（英文词边界 / 中文子串命中即归入）。"""
    if not isinstance(text, str) or not text.strip():
        return set()
    low = f" {text.lower()} "
    tags = set()
    for tag, kws in DIRECTION_TAG_VOCAB.items():
        if any(_kw_in_text(kw, low) for kw in kws):
            tags.add(tag)
    return tags


def _jd_direction_tags(jd_profile: dict) -> set:
    """JD 的方向标签集合：综合 direction / business_area / title / domain_keywords。"""
    parts = [
        str(jd_profile.get("direction") or ""),
        str(jd_profile.get("business_area") or ""),
        str(jd_profile.get("title") or ""),
        " ".join(str(x) for x in (jd_profile.get("domain_keywords") or [])),
    ]
    return _map_direction_tags(" ".join(parts))


def compute_direction_bonus(jd_profile: dict, user_direction_tags) -> dict:
    """岗位方向偏好加分（确定性，≤ DIRECTION_BONUS_CAP）。

    user_direction_tags：用户偏好原始标签（list 或 {tag: weight} dict 的键）。
    分级（job_matching_and_ranking.md §4.1）：
      - 核心方向完全一致（标签集合有交集）         +6
      - 同一上层方向、子方向接近                   +4
      - 存在弱相关技术交集（偏好词直接出现在 JD 文本）+1~2
      - 无明显关联                                 +0
    返回 {bonus, matched_tags, reason}。
    """
    if isinstance(user_direction_tags, dict):
        pref_tags_raw = list(user_direction_tags.keys())
    elif isinstance(user_direction_tags, (list, tuple, set)):
        pref_tags_raw = list(user_direction_tags)
    else:
        pref_tags_raw = []
    pref_tags_raw = [str(t).strip() for t in pref_tags_raw if str(t).strip()]
    if not pref_tags_raw:
        return {"bonus": 0.0, "matched_tags": [], "reason": "用户未表达方向偏好。"}

    user_tags = _map_direction_tags(" ".join(pref_tags_raw))
    jd_tags = _jd_direction_tags(jd_profile)

    # 1) 核心方向完全一致：标签集合有交集
    overlap = user_tags & jd_tags
    if overlap:
        return {"bonus": DIRECTION_BONUS_CAP, "matched_tags": sorted(overlap),
                "reason": f"岗位方向与偏好标签一致（{', '.join(sorted(overlap))}）。"}

    # 2) 同一上层方向、子方向接近：父标签相同
    user_parents = {DIRECTION_PARENTS.get(t) for t in user_tags if DIRECTION_PARENTS.get(t)}
    jd_parents = {DIRECTION_PARENTS.get(t) for t in jd_tags if DIRECTION_PARENTS.get(t)}
    if user_parents & jd_parents:
        return {"bonus": 4.0, "matched_tags": sorted(user_tags),
                "reason": "岗位与偏好同属一个上层方向，子方向接近。"}

    # 3) 弱相关：偏好原词直接出现在 JD 文本中
    hay = " ".join([
        str(jd_profile.get("direction") or ""),
        str(jd_profile.get("business_area") or ""),
        str(jd_profile.get("title") or ""),
        " ".join(str(x) for x in (jd_profile.get("domain_keywords") or [])),
        " ".join(str(x) for x in (jd_profile.get("hard_skills") or [])),
    ]).lower()
    weak_hits = [t for t in pref_tags_raw if t.lower() in hay]
    if weak_hits:
        bonus = 2.0 if len(weak_hits) >= 2 else 1.0
        return {"bonus": bonus, "matched_tags": weak_hits,
                "reason": f"岗位与偏好存在弱相关技术交集（{', '.join(weak_hits[:3])}）。"}

    return {"bonus": 0.0, "matched_tags": [], "reason": "岗位方向与用户偏好无明显关联。"}


# ===== 6.2 通勤偏好加分 commute_bonus（确定性时长分级，≤ COMMUTE_BONUS_CAP）=====
def compute_commute_bonus(commute_info: Optional[dict], max_minutes: Optional[int]) -> dict:
    """通勤偏好加分（job_matching_and_ranking.md §4.2）。

    仅当用户设定了通勤时长上限 max_minutes 时才加分；只要求「展示通勤时间路线」（max_minutes 为 None）→ +0。
    分级：满足上限 +4 / 接近（≤1.25×）+2 / 超出可接受（≤1.5×）+1 / 明显超出或不可计算 +0。
    commute_info：{commute_time_minutes, within_limit, ...}（来自 tools/commute）。
    """
    if not max_minutes or not isinstance(commute_info, dict):
        return {"bonus": 0.0, "reason": "未设定通勤时长偏好，仅展示通勤信息不加分。"}
    minutes = commute_info.get("commute_time_minutes")
    if not isinstance(minutes, (int, float)):
        return {"bonus": 0.0, "reason": "通勤时长不可计算，不加分。"}
    if minutes <= max_minutes:
        return {"bonus": COMMUTE_BONUS_CAP, "reason": f"通勤约 {int(minutes)} 分钟，满足 {max_minutes} 分钟上限。"}
    if minutes <= max_minutes * 1.25:
        return {"bonus": 2.0, "reason": f"通勤约 {int(minutes)} 分钟，接近目标时长。"}
    if minutes <= max_minutes * 1.5:
        return {"bonus": 1.0, "reason": f"通勤约 {int(minutes)} 分钟，超出目标但可接受。"}
    return {"bonus": 0.0, "reason": f"通勤约 {int(minutes)} 分钟，明显超出目标。"}


# ===== 6.3 排序分 rank_score =====
def calculate_rank_score(match_score: int, direction_bonus: float, commute_bonus: float) -> int:
    """rank_score = min(100, match_score + direction_bonus + commute_bonus)。"""
    raw = match_score + (direction_bonus or 0.0) + (commute_bonus or 0.0)
    return max(0, min(100, int(round(raw))))


# ===== 7. 推荐等级（按 rank_score）=====
def judge_match_level(rank_score: int) -> str:
    """rank_score >= 75 -> recommended（推荐投递）；50~74 -> maybe（可酌情投递）；<50 -> not_recommended（暂不建议）。"""
    if rank_score >= 75:
        return "recommended"
    if rank_score >= 50:
        return "maybe"
    return "not_recommended"


# ===== 8. summary 生成 =====
def generate_summary(result: dict) -> str:
    """简洁说明是否推荐 + 主要优势 + 主要风险，不超过 100 字。"""
    level = result.get("match_level")
    rank = result.get("rank_score", 0)
    level_text = {
        "recommended": "推荐投递",
        "maybe": "可酌情投递",
        "not_recommended": "暂不建议投递",
    }.get(level, "暂无结论")

    # 主要优势 / 短板：在技能、项目两维中取
    dim_scores = {
        "技能": result.get("skill_score", {}).get("score", 0),
        "项目": result.get("project_score", {}).get("score", 0),
    }
    top_dim = max(dim_scores.items(), key=lambda kv: kv[1])
    bottom_dim = min(dim_scores.items(), key=lambda kv: kv[1])

    # 主要风险：取 risk_analysis 中首项 type
    risks = result.get("risk_analysis") or []
    risk_text = risks[0].get("type") if risks and isinstance(risks[0], dict) else None

    parts = [f"{level_text}（推荐分 {rank}）"]
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
def score_match(resume_profile: dict, jd_profile: dict,
                user_direction_tags=None, with_project: bool = True) -> dict:
    """完整评分流程（两层评分）。

    user_direction_tags：用户方向偏好标签（list 或 {tag: weight}），用于 direction_bonus；
    通勤加分 commute_bonus 在评分阶段未知（通勤在增强节点计算），此处置 0，
    由 enhancement 节点算完通勤后回填并重算 rank_score。

    with_project：是否立即调 LLM 算 project_score。
      - True（默认，单 JD 评估 / CLI / eval）：完整两层评分；
      - False（候选池**惰性评分**）：跳过 project LLM，用 skill_score 作 match_score 的中性代理
        （假定 project≈skill，与 skill 同 0~100 量纲，仅供候选池排序），并置 project_pending=True；
        该岗位进入**展示批次**时再由 apply_project_score 精算 project 并刷新 match_score/rank_score。
        project LLM 是评分主瓶颈，全候选池每岗一次会拖垮大候选池场景，故只对展示岗位付费
        （见 nodes/executor 的 _finalize_projects）。
    """
    # 1. 技能分（规则）
    skill_result = calculate_skill_score(resume_profile, jd_profile)
    # 2. 学历分（规则）—— 仅供风险分析的学历门槛判定，不计入评分
    education_result = calculate_education_score(resume_profile, jd_profile)
    # 3. 项目分（LLM，5 档 rubric）；惰性模式跳过，用 skill 作中性代理
    if with_project:
        project_result = call_llm_score(resume_profile, jd_profile)
        match_score = calculate_match_score(skill_result["score"], project_result["score"])
        project_pending = False
    else:
        project_result = {"score": skill_result["score"],
                          "evidence": ["项目分惰性未计算（仅对展示岗位精算）。"], "pending": True}
        match_score = skill_result["score"]   # 中性代理：假定 project≈skill，仅供候选池排序
        project_pending = True
    # 4. 风险分析（规则 + JD risk_points 合并）
    risks = generate_risk_analysis(resume_profile, jd_profile, skill_result, education_result)
    # 5. 方向偏好加分（确定性标签映射）；通勤加分评分阶段为 0
    direction_bonus_info = compute_direction_bonus(jd_profile, user_direction_tags)
    direction_bonus = direction_bonus_info["bonus"]
    commute_bonus = 0.0
    # 6. 排序分 rank_score + 推荐等级
    rank_score = calculate_rank_score(match_score, direction_bonus, commute_bonus)
    match_level = judge_match_level(rank_score)

    result = {
        "skill_score": skill_result,
        "project_score": project_result,
        "education_gate": education_result,  # 学历门槛三态，仅供风险/语气，不计入评分
        "risk_analysis": risks,
        "match_score": match_score,
        "direction_bonus": direction_bonus,
        "direction_bonus_info": direction_bonus_info,
        "commute_bonus": commute_bonus,
        "rank_score": rank_score,
        "match_level": match_level,
        "project_pending": project_pending,  # True=项目分待惰性精算（apply_project_score）
        "summary": "",
    }
    # 7. summary
    result["summary"] = generate_summary(result)
    return result


def _recompute_after_project(result: dict, project_result: dict) -> dict:
    """用精算出的 project_result 刷新 result 的 project_score / match_score / rank_score /
    match_level / summary 并清 project_pending。就地修改并返回。单/批量精算共用。"""
    skill_score = (result.get("skill_score") or {}).get("score", 0)
    result["project_score"] = project_result
    result["match_score"] = calculate_match_score(skill_score, project_result.get("score", 50))
    result["rank_score"] = calculate_rank_score(
        result["match_score"], result.get("direction_bonus", 0.0), result.get("commute_bonus", 0.0))
    result["match_level"] = judge_match_level(result["rank_score"])
    result["project_pending"] = False
    result["summary"] = generate_summary(result)
    return result


def apply_project_score(resume_profile: dict, jd_profile: dict, result: dict) -> dict:
    """惰性精算（**单岗**）：对 with_project=False 产出的评分补算 project_score（一次 LLM）并刷新
    match_score / rank_score / match_level / summary。已精算（project_pending=False）直接返回。

    **就地修改并返回** result（result 即候选池项的 match_score 字典）。展示**批次**请用
    apply_project_scores_batch（1 次 LLM 出 N 分），此单岗版供单 JD/特殊路径兜底。
    """
    if not isinstance(result, dict) or not result.get("project_pending"):
        return result
    return _recompute_after_project(result, call_llm_score(resume_profile, jd_profile))


def apply_project_scores_batch(resume_profile: dict, results: list, jd_profiles: list) -> list:
    """惰性精算（**批量**）：对一批 (result, jd) 用 **1 次 LLM** 算出各自 project_score 并分别刷新。

    results 与 jd_profiles 一一对应。只精算其中 project_pending 的（已精算的跳过），把展示批次原本
    「每岗一次 project LLM」降为「整批一次」。逐岗独立打分（call_llm_score_batch 内禁止横向比较）。
    就地修改 results 并返回；**绝不抛**（批量调用失败时由 _distribute/兜底给保守默认，不留 pending 残值）。
    """
    pending_idx = [i for i, r in enumerate(results)
                   if isinstance(r, dict) and r.get("project_pending")]
    if not pending_idx:
        return results
    try:
        project_results = call_llm_score_batch(
            resume_profile, [jd_profiles[i] for i in pending_idx])
    except Exception as e:   # call_llm_score_batch 已自兜底，这里再兜一层防御
        print(f"[ERROR] 批量 project 评分异常，全部保守默认：{e}")
        project_results = []
    for k, i in enumerate(pending_idx):
        pr = (project_results[k] if k < len(project_results)
              else {"score": 50, "evidence": ["批量评分缺失，保守默认。"]})
        _recompute_after_project(results[i], pr)
    return results


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
