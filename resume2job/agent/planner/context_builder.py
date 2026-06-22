# -*- coding: utf-8 -*-
"""组装 PlannerContext —— 喂给 nlu_extractor 的结构化上下文。

只注入结构化、必要的信号；其中 last_results_summary 是带序号的上一轮岗位列表，
使「第二个」「那个腾讯的」这类指代能被稳定解析为 job_id。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from resume2job.storage import session_store


@dataclass
class PlannerContext:
    current_message: str                                   # 本轮用户原话
    has_jd: bool                                            # 本轮是否带 JD 原文
    has_pdf: bool                                           # 本轮是否带简历 PDF
    session_state: Dict[str, Any] = field(default_factory=dict)   # 会话短期状态（含活跃约束）
    last_results: List[dict] = field(default_factory=list)        # 上一轮结果 [{rank,job_id,title,city}]
    active_resume_summary: Optional[str] = None            # 已有画像一句话摘要（缓存命中时）
    active_jd_summary: Optional[str] = None                # 当前 JD 摘要
    explicit_ui_filters: Dict[str, Any] = field(default_factory=dict)  # CLI/UI 显式传入的过滤
    recent_dialogue: str = ""                              # 简短对话兜底（结构化信号缺失时参考）
    resume_degree: Optional[str] = None                    # 简历最高学历（学历硬约束默认取此，不从对话推断）

    def last_results_block(self) -> str:
        """渲染带序号的上一轮结果，供 prompt 解析指代。空则返回 ""。"""
        if not self.last_results:
            return ""
        lines = []
        for r in self.last_results:
            rank = r.get("rank")
            lines.append(f"{rank}. job_id={r.get('job_id')}, {r.get('title')}（{r.get('city') or '城市未知'}）")
        return "\n".join(lines)


def _resume_summary(resume_profile: Optional[dict]) -> Optional[str]:
    if not isinstance(resume_profile, dict) or not resume_profile:
        return None
    skills = resume_profile.get("skills") or resume_profile.get("skill_groups") or []
    flat = []
    if isinstance(skills, list):
        for s in skills[:8]:
            flat.append(s if isinstance(s, str) else str(s.get("skill") or s.get("name") or ""))
    intents = ((resume_profile.get("job_preferences") or {}).get("intentions")) or []
    parts = []
    if flat:
        parts.append("技能：" + "、".join(x for x in flat if x))
    if intents:
        parts.append("意向：" + "、".join(str(x) for x in intents[:3]))
    return "；".join(parts) or None


def _jd_summary(jd_text: Optional[str], jd_profiles: Optional[list]) -> Optional[str]:
    if jd_profiles and isinstance(jd_profiles[0], dict):
        p = jd_profiles[0]
        return f"{p.get('company') or '?'} - {p.get('title') or '?'}（{p.get('direction') or ''}）"
    if jd_text:
        return (jd_text[:60] + "…") if len(jd_text) > 60 else jd_text
    return None


def _format_recent(messages: Optional[list], max_chars: int = 400) -> str:
    """简短对话兜底（仅在结构化信号不足时供 LLM 参考）。"""
    if not messages:
        return ""
    label = {"user": "用户", "assistant": "助手"}
    lines = [f"{label.get(m.get('role'), m.get('role'))}：{str(m.get('content') or '').strip()}"
             for m in messages if isinstance(m, dict) and str(m.get("content") or "").strip()]
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


def _probe_resume(state: dict):
    """探测「本轮是否有可用画像」。

    planner 在 profile_cache 节点**之前**运行，state.resume_profile 此刻通常为空
    （缓存画像要到 profile_cache 才回填）。若只看 state.resume_profile，多轮里不带
    简历的轮次会被误判「无简历」而错误澄清。故这里主动探测缓存：本轮已有 > 缓存命中。
    """
    rp = state.get("resume_profile")
    if rp:
        return rp
    try:
        from resume2job.storage import profile_cache
        cached = profile_cache.load_latest_profile(profile_cache.DEFAULT_USER_ID)
        if cached:
            return cached.get("structured_profile")
    except Exception:
        pass
    return None


def build_context(state: dict) -> PlannerContext:
    """从 AgentState 组装 PlannerContext（读会话短期状态 + 结构化摘要）。"""
    session_id = state.get("session_id") or ""
    sess = session_store.get_session(session_id) if session_id else {}

    rc = state.get("retrieval_config") or {}
    ui_filters = {k: rc.get(k) for k in ("city_filter", "education_filter", "job_type_filter")
                  if rc.get(k)}

    probed = _probe_resume(state)
    return PlannerContext(
        current_message=state.get("user_query") or "",
        has_jd=bool(state.get("jd_text")),
        has_pdf=bool(state.get("pdf_path")),
        session_state=sess,
        last_results=sess.get("last_results") or [],
        active_resume_summary=_resume_summary(probed),
        active_jd_summary=_jd_summary(state.get("jd_text"), state.get("jd_profiles")),
        explicit_ui_filters=ui_filters,
        recent_dialogue=_format_recent(state.get("messages")),
        resume_degree=(probed.get("highest_degree") if isinstance(probed, dict) else None),
    )
