# -*- coding: utf-8 -*-
"""observability/redaction.py —— 脱敏与训练数据清洗（纯规则，0 LLM）。

从日志 / trace / 简历画像里移除对推荐**非必要**的个人身份信息（PII），再用于 SFT / 偏好对 / 错误分析 /
离线评测 / 项目展示。这是「企业级对齐」里最容易被忽略、却最该做的一步。

策略：
    - **保留**对推荐有用的信息：学历 / 专业 / 技能 / 项目经历 / 城市偏好 / 岗位偏好 / 实习时长 等；
    - **替换**个人身份信息为占位符：姓名 <NAME> / 邮箱 <EMAIL> / 手机 <PHONE> / 微信 <WECHAT> /
      QQ <QQ> / 身份证 <ID> / 精确住址 <ADDRESS> / 个人主页·GitHub <URL>。
正则只命中明确的 PII 形态（邮箱 / 手机 / 身份证 / URL），不会误伤技能名 / 城市 / 公司名等正常文本，
故可对任意结构递归扫描；已知身份字段（contact / name）再按 key 显式替换。
"""

from __future__ import annotations

import re
from typing import Any

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://[^\s，。、；]+", re.IGNORECASE)
_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")          # 18 位身份证
_MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")       # 国内手机号
_LANDLINE_RE = re.compile(r"(?<!\d)\d{3,4}-\d{7,8}(?!\d)") # 座机 区号-号码

# contact 子字段按 key 替换（这些靠正则抓不全：姓名 / 微信号 / QQ / GitHub handle）
_CONTACT_PLACEHOLDER = {
    "email": "<EMAIL>", "phone": "<PHONE>", "mobile": "<PHONE>", "tel": "<PHONE>",
    "wechat": "<WECHAT>", "weixin": "<WECHAT>", "qq": "<QQ>",
    "github": "<URL>", "homepage": "<URL>", "blog": "<URL>", "website": "<URL>",
    "address": "<ADDRESS>", "addr": "<ADDRESS>", "location_detail": "<ADDRESS>",
}
_NAME_KEYS = ("name", "full_name", "姓名")
_ADDRESS_KEYS = ("address", "addr", "home_address", "住址", "现居地址", "详细地址")


def redact_text(text: str) -> str:
    """把一段文本里的邮箱 / URL / 身份证 / 手机 / 座机替换为占位符。其余文本原样保留。"""
    if not isinstance(text, str) or not text:
        return text
    s = _EMAIL_RE.sub("<EMAIL>", text)
    s = _URL_RE.sub("<URL>", s)
    s = _ID_RE.sub("<ID>", s)
    s = _MOBILE_RE.sub("<PHONE>", s)
    s = _LANDLINE_RE.sub("<PHONE>", s)
    return s


def redact_obj(obj: Any) -> Any:
    """递归脱敏任意结构（dict/list/str）：对所有字符串套 redact_text，并对身份字段按 key 替换。

    返回新结构，不就地修改入参。
    """
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(redact_obj(x) for x in obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).strip().lower()
            if lk in _NAME_KEYS:
                out[k] = "<NAME>" if v else v
            elif lk in _CONTACT_PLACEHOLDER:
                out[k] = _CONTACT_PLACEHOLDER[lk] if v else v
            elif lk in _ADDRESS_KEYS:
                out[k] = "<ADDRESS>" if v else v
            else:
                out[k] = redact_obj(v)
        return out
    return obj


def redact_resume_profile(profile: dict) -> dict:
    """脱敏简历画像：移除姓名 / 联系方式 / 精确住址等身份信息，保留技能 / 教育 / 项目 / 偏好。"""
    if not isinstance(profile, dict):
        return profile
    return redact_obj(profile)


def redact_trace(trace: dict) -> dict:
    """脱敏一条 request trace（主要是 user_query 可能含 PII / 粘贴的简历片段；其余多为 job_id+分数）。"""
    if not isinstance(trace, dict):
        return trace
    return redact_obj(trace)
