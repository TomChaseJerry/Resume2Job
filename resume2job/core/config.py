# -*- coding: utf-8 -*-
"""统一模型 / 接口参数配置。

集中管理所有 LLM、Embedding、Rerank 模型与服务端点，方便一处切换。
两种切换方式（环境变量优先级高于此文件默认值）：

  1) 直接改本文件下方的默认字符串（最简单，重启生效）；
  2) 设置环境变量（无需改代码，适合临时/部署切换），例如：
       PowerShell:  $env:RESUME2JOB_CHAT_MODEL = "qwen-plus"
       bash:        export RESUME2JOB_CHAT_MODEL=qwen-plus

各模块统一从这里读取，请勿在业务文件里硬编码模型名 / BASE_URL。
"""

import os


def _env(name: str, default: str) -> str:
    """读取环境变量；为空或未设置时回退到默认值。"""
    val = os.environ.get(name)
    return val if val else default


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量（"0"/"false"/"no" 视为 False，其余非空为 True）。"""
    val = os.environ.get(name)
    if not val:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off")


# ===== 服务端点（阿里云百炼 OpenAI 兼容模式）=====
BASE_URL = _env("RESUME2JOB_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# DashScope 原生 API（rerank 等非 OpenAI 兼容接口）
DASHSCOPE_NATIVE_URL = _env(
    "RESUME2JOB_DASHSCOPE_NATIVE_URL", "https://dashscope.aliyuncs.com/api/v1"
)

# ===== 主对话 / 生成模型 =====
# 用于：简历解析、JD 解析、推荐写作、技能差距、学习计划、面试准备、
#       匹配打分主模型、检索 query 改写等需要较强推理的场景。
CHAT_MODEL = _env("RESUME2JOB_CHAT_MODEL", "deepseek-v4-pro")

# ===== 轻量模型 =====
# 规划（task_type + 检索过滤 + 通勤意图的 Function Calling 结构化输出）：输出短、规则明确。
PLANNER_MODEL = _env("RESUME2JOB_PLANNER_MODEL", "qwen-flash")
# 评分（project/direction）：rubric 明确、输出仅 score + evidence。
SCORE_MODEL = _env("RESUME2JOB_SCORE_MODEL", "qwen-plus")
# 评测 LLM-as-judge：与被评模型区分，避免自评偏置时可单独切换。
JUDGE_MODEL = _env("RESUME2JOB_JUDGE_MODEL", "glm-5")

# ===== Embedding 模型 =====
# 注意：更换 Embedding 模型通常会改变向量维度，需重建 Chroma 向量库后才能检索一致。
EMBEDDING_MODEL = _env("RESUME2JOB_EMBEDDING_MODEL", "text-embedding-v3")

# ===== Rerank 模型（DashScope 原生 text-rerank 接口）=====
RERANK_MODEL = _env("RESUME2JOB_RERANK_MODEL", "qwen3-rerank")

# ===== 检索策略 =====
# 检索模式：vector（纯向量）/ bm25（纯关键词）/ hybrid（RRF 融合，默认）
RETRIEVAL_MODE = _env("RESUME2JOB_RETRIEVAL_MODE", "hybrid")
# 是否在召回后调用 rerank 精排（每次检索多一次 API 调用）
USE_RERANK = _env_bool("RESUME2JOB_USE_RERANK", True)
# RRF 融合常数（业界常用 60）：score = Σ 1 / (RRF_K + rank)
RRF_K = int(_env("RESUME2JOB_RRF_K", "60"))
