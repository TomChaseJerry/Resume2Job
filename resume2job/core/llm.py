# -*- coding: utf-8 -*-
"""统一 LLM / Embedding 客户端与 JSON 输出处理工具（单一事实来源）。

各业务模块（resume_parser / jd_parser / match_scorer /
recommendation / learning_plan / interview ...）统一从这里取用：

    - call_llm(system, user)       —— 文本补全（OpenAI SDK 兼容，阿里云百炼）
    - get_chat_llm(...)            —— LangChain ChatOpenAI 客户端（供 bind_tools /
                                      with_structured_output 等 Function Calling 场景）
    - get_embedding(text)          —— text-embedding 向量
    - clean_llm_json_output(raw)   —— 去 Markdown 围栏、截取 JSON 本体
    - safe_json_parse(raw)         —— 清洗 + 解析，失败返回 None
"""

import os
import json
import re
from typing import Any, List, Optional

from resume2job.core import config


# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    """读取 DASHSCOPE_API_KEY；未配置时抛 RuntimeError（由调用方决定兜底策略）。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置")
    return api_key


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """调用阿里云百炼兼容接口（OpenAI SDK 风格），返回模型纯文本响应。

    默认低温度（抽取 / 打分类任务结构稳定）；生成类任务可自行调高。
    """
    from openai import OpenAI  # 延迟导入，避免无依赖环境 import 期失败

    client = OpenAI(api_key=get_api_key(), base_url=config.BASE_URL)
    completion = client.chat.completions.create(
        model=model or config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return (completion.choices[0].message.content or "").strip()


def get_chat_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    **kwargs: Any,
):
    """构造 LangChain ChatOpenAI 客户端（Function Calling / bind_tools 场景用）。

    enable_thinking=False 关闭 Qwen 混合推理模型的思考模式（结构化输出无需推理过程，
    省时省钱）；旧模型会忽略该参数。
    """
    from langchain_openai import ChatOpenAI  # 延迟导入

    return ChatOpenAI(
        model=model or config.CHAT_MODEL,
        openai_api_key=get_api_key(),
        openai_api_base=config.BASE_URL,
        temperature=temperature,
        extra_body={"enable_thinking": enable_thinking},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list:
    """调用阿里云百炼 Embedding 接口，返回 list[float]。

    错误约定：
      - 空文本 → 抛 ValueError；
      - 未配置 API Key → 抛 RuntimeError；
      - 其余 API 异常由调用方捕获后跳过单条任务。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("get_embedding 输入文本为空")

    from openai import OpenAI  # 延迟导入

    client = OpenAI(api_key=get_api_key(), base_url=config.BASE_URL)
    resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=text)
    return list(resp.data[0].embedding)


# ---------------------------------------------------------------------------
# JSON 输出处理
# ---------------------------------------------------------------------------

def clean_llm_json_output(raw: str) -> str:
    """清洗 LLM 输出：
    1) 去除 ```json / ``` Markdown 包装；
    2) 截取第一个 '{' 到最后一个 '}' 之间的内容；
    3) 返回清洗后的字符串（即使没有 {} 也返回 strip 后的原始内容）。
    """
    if not raw:
        return ""
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return t


def safe_json_parse(raw: str) -> Optional[dict]:
    """清洗 + 解析 LLM 的 JSON 输出。失败时打印错误并返回 None。"""
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


