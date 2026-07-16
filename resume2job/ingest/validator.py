# -*- coding: utf-8 -*-
"""ingest/validator.py — 岗位质量校验（纯规则，零 LLM）。

判断一条已解析的 jd_profile 是否「足够完整、可信、值得进入召回池」。这是**入库前的质量闸门**，
与 jd_parser.validate_jd_profile 的职责严格区分：
    - jd_parser.validate_jd_profile：保证**结构合法**（补默认值、类型修正、枚举归一）——任何 JD 都能过；
    - 本模块 validate_job：评估**内容质量**（字段是否齐全、JD 是否过短、地点 / 学历是否缺失、
      技能是否异常），给出 is_valid + quality_score + 具名 warnings。供 lifecycle 决定入库 / 隔离，
      供 ranking 特征（jd_quality_score）与 eval/fairness_audit（按来源统计缺字段率）复用。

只做规则判断，不调用任何模型；可对全库批量跑（backfill_lifecycle_fields 即用它回填 quality_score）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from resume2job.parsing.jd_parser import job_cities


# ===== 阈值 =====
MIN_JD_TEXT_LEN = 60          # JD 原文少于此长度视为过短（占位 / 无正文）
MAX_HARD_SKILLS = 30          # 必备硬技能数超过此值视为异常（抽取脏 / 把整段塞进来）

# ===== 具名 warning 码（下游按来源 / 城市统计缺字段率时按码聚合）=====
W_TITLE_MISSING = "title_missing"
W_COMPANY_MISSING = "company_missing"
W_JD_TOO_SHORT = "jd_too_short"
W_LOCATION_UNKNOWN = "location_unknown"
W_JOB_TYPE_MISSING = "job_type_missing"
W_SKILLS_EMPTY = "skills_empty"
W_RESPONSIBILITIES_MISSING = "responsibilities_missing"
W_REQUIRED_SKILLS_EXCESSIVE = "required_skills_excessive"
W_EDUCATION_UNKNOWN = "education_unknown"

# 各 warning 对 quality_score 的扣分（从 1.0 起逐项扣，clamp 到 [0,1]）。
# 关键内容缺失扣得多（标题 / 过短 / 无技能），可缺省的元数据扣得少（公司 / 地点 / 学历）。
_PENALTY = {
    W_TITLE_MISSING: 0.40,
    W_JD_TOO_SHORT: 0.30,
    W_SKILLS_EMPTY: 0.30,
    W_RESPONSIBILITIES_MISSING: 0.15,
    W_REQUIRED_SKILLS_EXCESSIVE: 0.10,
    W_COMPANY_MISSING: 0.10,
    W_LOCATION_UNKNOWN: 0.10,
    W_JOB_TYPE_MISSING: 0.05,
    W_EDUCATION_UNKNOWN: 0.05,
}


@dataclass
class QualityReport:
    """一条 JD 的质量评估结果。"""
    is_valid: bool
    quality_score: float          # [0,1]，1=字段齐全可信
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"is_valid": self.is_valid, "quality_score": self.quality_score,
                "warnings": list(self.warnings)}


def _is_blank(s) -> bool:
    return not (isinstance(s, str) and s.strip())


def _is_unknown(s) -> bool:
    """空 / None / 含 'unknown' / 占位词 视为缺失。"""
    if _is_blank(s):
        return True
    low = s.strip().lower()
    return ("unknown" in low) or low in {"未知", "未知公司", "未知岗位", "n/a", "na", "null", "none"}


def validate_job(jd_profile: dict, jd_text: str = "") -> QualityReport:
    """评估 jd_profile 的内容质量。

    is_valid 闸门（宽松，只挡明显残缺）：必须有标题、有可匹配内容（硬技能 / 工具 / 职责其一）、
    且 JD 原文不过短。公司 / 地点 / 学历缺失只记 warning、**不**判失败——集团统招 JD（无公司名）
    与远程 / 多地岗位（地点不明确）仍应进库，质量分体现其完整度即可。
    """
    if not isinstance(jd_profile, dict):
        jd_profile = {}
    warnings: List[str] = []

    title_missing = _is_unknown(jd_profile.get("title"))
    company_missing = _is_unknown(jd_profile.get("company"))

    hard = [s for s in (jd_profile.get("hard_skills") or []) if isinstance(s, str) and s.strip()]
    tools = [s for s in (jd_profile.get("tools_or_frameworks") or []) if isinstance(s, str) and s.strip()]
    resp = [s for s in (jd_profile.get("responsibilities") or []) if isinstance(s, str) and s.strip()]
    skills_empty = not (hard or tools)

    text_len = len((jd_text or "").strip())
    too_short = text_len < MIN_JD_TEXT_LEN

    if title_missing:
        warnings.append(W_TITLE_MISSING)
    if company_missing:
        warnings.append(W_COMPANY_MISSING)
    if too_short:
        warnings.append(W_JD_TOO_SHORT)
    if not job_cities(jd_profile):
        warnings.append(W_LOCATION_UNKNOWN)
    if _is_blank(jd_profile.get("job_type")):
        warnings.append(W_JOB_TYPE_MISSING)
    if skills_empty:
        warnings.append(W_SKILLS_EMPTY)
    if not resp:
        warnings.append(W_RESPONSIBILITIES_MISSING)
    if len(hard) > MAX_HARD_SKILLS:
        warnings.append(W_REQUIRED_SKILLS_EXCESSIVE)
    if _is_blank(jd_profile.get("education_level")):
        warnings.append(W_EDUCATION_UNKNOWN)

    penalty = sum(_PENALTY.get(w, 0.0) for w in warnings)
    quality_score = round(max(0.0, min(1.0, 1.0 - penalty)), 3)

    is_valid = (not title_missing) and (not skills_empty or bool(resp)) and (not too_short)
    return QualityReport(is_valid=is_valid, quality_score=quality_score, warnings=warnings)
