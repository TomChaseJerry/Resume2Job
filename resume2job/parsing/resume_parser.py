"""
简历解析模块（Resume Parser）

读取 PDF 简历 -> 使用 pdfplumber 提取全文 -> 调用阿里云百炼 (qwen-max) -> 输出标准化 JSON 用户画像

输出 JSON 结构面向 Agentic RAG 岗位推荐场景，覆盖：
教育经历（多段）、技能分组、工作 / 实习经历、项目、科研、论文、获奖、证书、竞赛、求职偏好等。
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Optional, Any

import pdfplumber

# ===== 模型 / LLM 工具（统一走 core 层）=====
from resume2job.core.config import CHAT_MODEL as MODEL_NAME
from resume2job.core.llm import call_llm as _core_call_llm, safe_json_parse


# ===== JSON Schema 示例（用于 Prompt 中向 LLM 展示目标结构）=====
# 该 Schema 面向后续岗位推荐系统：每个字段都对应一个可用于召回 / 排序的画像维度
JSON_SCHEMA_EXAMPLE = {
    "name": "string or null",
    "contact": {
        "email": "string or null",
        "phone": "string or null",
        "github": "string or null",
        "Birthplace": "string or null",
        "other": ["string"]
    },
    "educations": [
        {
            "school": "string",
            "degree": "本科 / 硕士 / 博士 / null",
            "major": "string or null",
            "start_date": "YYYY-MM or null",
            "end_date": "YYYY-MM or 至今 or null",
            "gpa": "string or null",
            "highlights": ["核心课程 / 排名 / 奖学金等"]
        }
    ],
    "highest_degree": "本科 / 硕士 / 博士 / null",
    "skill_groups": [
        {
            "category": "如：编程语言 / 框架 / 大模型 / 数据库 / 工具 / 软技能",
            "items": ["skill1", "skill2"]
        }
    ],
    "skills": ["扁平化技能标签，用于快速召回"],
    "experiences": [
        {
            "company": "string",
            "title": "string",
            "type": "实习 / 全职 / 兼职 / null",
            "start_date": "YYYY-MM or null",
            "end_date": "YYYY-MM or 至今 or null",
            "location": "string or null",
            "tasks": ["职责描述"],
            "achievements": ["可量化成果"],
            "keywords": ["技术 / 业务关键词"]
        }
    ],
    "projects": [
        {
            "name": "string",
            "role": "string or null",
            "description": "string",
            "tech_stack": ["string"],
            "tasks": ["承担的具体工作"],
            "achievements": ["可量化成果"],
            "keywords": ["技术关键词"],
            "evidence": ["来自原文、可证明能力的句子"]
        }
    ],
    "research": [
        {
            "topic": "string",
            "role": "string or null",
            "description": "string",
            "keywords": ["string"]
        }
    ],
    "publications": [
        {
            "title": "string",
            "venue": "期刊 / 会议 or null",
            "year": "string or null",
            "authors": ["string"],
            "role": "一作 / 共一 / 通讯 / 参与 / null"
        }
    ],
    "awards": [
        {"name": "string", "level": "国家级 / 省级 / 校级 / null", "year": "string or null"}
    ],
    "certificates": [
        {"name": "string", "year": "string or null"}
    ],
    "competitions": [
        {"name": "string", "rank": "string or null", "year": "string or null", "keywords": ["string"]}
    ],
    "job_preferences": {
        "intentions": ["大模型算法 / NLP / 后端 / ..."],
        "locations": ["期望城市"],
        "availability": "随时 / 3个月 / YYYY-MM 起 / null",
        "salary_expectation": "string or null",
        "work_mode": "线下 / 远程 / 混合 / null"
    },
    "extraction_meta": {
        "source_file": "string",
        "model": "string",
        "extracted_at": "ISO8601 时间字符串",
        "warnings": ["解析过程中的提示信息"]
    }
}


# ===== System Prompt：约束角色、任务、输出格式 =====
SYSTEM_PROMPT = """你是一名专业的简历信息抽取助手，擅长从中英文简历中抽取结构化信息以供下游岗位推荐系统使用。

【核心原则】
1. 只输出一个合法的 JSON 对象，不要输出 Markdown 代码块标记（如 ```json）、解释、前后缀文字。
2. 不得编造原文未出现的信息；不确定的字段一律置为 null 或空列表 []。
3. 字段缺失时：字符串字段返回 null，列表字段返回 []，对象字段保留键但内部用 null / [] 填充。

【字段抽取要求】
- educations：必须提取**所有**教育经历（本科 + 硕士 + 博士 + 交换 / 联培等），不要只保留最高学历。每段独立成对象。
- degree：必须规范化为「本科 / 硕士 / 博士」之一；学士=本科，硕士研究生=硕士，博士研究生=博士；其他情况（如高中、MBA、非学位项目）置为 null。
- highest_degree：基于 educations 推断（博士 > 硕士 > 本科），若无任何可识别学历则为 null。
- skill_groups：按类别分组（如编程语言 / 框架 / 大模型 / 数据库 / 工具 / 软技能），同时在 skills 中输出扁平化、去重的技能标签便于检索。
- experiences：包含实习、全职、兼职等工作经历，按时间倒序。
- projects.evidence：必须**来自简历原文**，优先保留可量化成果（如「准确率提升 5%」「日活 10w+」「QPS 提升 3 倍」）；当原文确实没有量化数字时，保留能证明能力的原文描述句（如「独立完成模型部署上线」），但不要复述 description。
- achievements 与 evidence 的区别：achievements 是简洁的成果点；evidence 是原文佐证句，可以更长。
- publications.role：一作 / 共一 / 通讯 / 参与，无法判断则 null。
- job_preferences.intentions：抽取明确的求职方向；若简历未写求职意向，则结合最近的实习 / 项目方向合理推断 1~3 个标签，并在 extraction_meta.warnings 中注明「intentions 由经验推断」。
- 日期统一格式 YYYY-MM；正在进行中的写「至今」。

【输出】
只输出 JSON 对象本体，确保可被 json.loads 直接解析。"""


# ===== 兜底默认值（用于 validate_resume_profile）=====
DEFAULT_PROFILE = {
    "name": None,
    "contact": {"email": None, "phone": None, "github": None, "homepage": None, "other": []},
    "current_location": None,
    "educations": [],
    "highest_degree": None,
    "skill_groups": [],
    "skills": [],
    "experiences": [],
    "projects": [],
    "research": [],
    "publications": [],
    "awards": [],
    "certificates": [],
    "competitions": [],
    "job_preferences": {
        "intentions": [],
        "locations": [],
        "availability": None,
        "salary_expectation": None,
        "work_mode": None,
    },
    "extraction_meta": {"source_file": None, "model": None, "extracted_at": None, "warnings": []},
}


# ===== PDF 文本提取 =====
def extract_pdf_text(pdf_path: str) -> str:
    """使用 pdfplumber 逐页提取 PDF 文本并拼接为完整字符串。"""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


# ===== Prompt 构造 =====
def build_user_prompt(resume_text: str) -> str:
    """拼接 User Prompt：简历原文 + JSON Schema 示例 + 明确指令。"""
    schema_str = json.dumps(JSON_SCHEMA_EXAMPLE, ensure_ascii=False, indent=2)
    return (
        "以下是一份简历的原始文本，请按照 System 指令和给定 JSON Schema 抽取信息。\n\n"
        "===== 简历原文 BEGIN =====\n"
        f"{resume_text}\n"
        "===== 简历原文 END =====\n\n"
        "===== 目标 JSON Schema（仅示意字段与类型，值为占位说明）=====\n"
        f"{schema_str}\n\n"
        "===== 指令 =====\n"
        "1. 严格按 Schema 的字段名与层级输出；\n"
        "2. educations 必须包含所有教育经历，不要只保留最高学历；\n"
        "3. highest_degree 根据 educations 推断（博士 > 硕士 > 本科）；\n"
        "4. evidence 必须来自简历原文，优先保留可量化成果；\n"
        "5. 缺失字段：字符串 null，列表 []；\n"
        "6. 只输出 JSON 本体，不要任何额外文字或 Markdown 包装。"
    )


# ===== LLM 调用（统一走 core.llm，抽取任务低温度）=====
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """调用主对话模型，返回纯文本响应。"""
    return _core_call_llm(system_prompt, user_prompt, model=MODEL_NAME)


# ===== 校验与补默认值 =====
def _ensure_type(value: Any, expected_default: Any) -> Any:
    """若 value 类型与默认值不一致，则返回默认值的深拷贝（通过 json 实现）。"""
    if value is None:
        return json.loads(json.dumps(expected_default))
    if isinstance(expected_default, list) and not isinstance(value, list):
        return []
    if isinstance(expected_default, dict) and not isinstance(value, dict):
        return json.loads(json.dumps(expected_default))
    return value


def validate_resume_profile(profile: dict, source_file: Optional[str] = None) -> dict:
    """对 LLM 输出做最小程度的结构校验：
    - 缺失键补默认值；
    - 顶层类型错误时回退默认值；
    - 推断 highest_degree（若 LLM 未给出但 educations 可推断）；
    - 写入 extraction_meta（model / extracted_at / source_file）。
    """
    if not isinstance(profile, dict):
        profile = {}

    # 1) 顶层缺失键补齐 & 顶层类型校验
    result: dict = {}
    for key, default in DEFAULT_PROFILE.items():
        result[key] = _ensure_type(profile.get(key, default), default)

    # 2) 嵌套对象字段补齐（contact / job_preferences / extraction_meta）
    for nested_key in ("contact", "job_preferences", "extraction_meta"):
        nested_default = DEFAULT_PROFILE[nested_key]
        nested_value = result[nested_key] if isinstance(result[nested_key], dict) else {}
        merged = {**nested_default, **nested_value}
        # 内部列表 / 标量类型再做一次保护
        for k, v in nested_default.items():
            merged[k] = _ensure_type(merged.get(k, v), v)
        result[nested_key] = merged

    # 3) highest_degree 兜底推断
    degree_rank = {"本科": 1, "硕士": 2, "博士": 3}
    if not result.get("highest_degree"):
        max_rank = 0
        best = None
        for edu in result.get("educations", []):
            if not isinstance(edu, dict):
                continue
            d = edu.get("degree")
            if d in degree_rank and degree_rank[d] > max_rank:
                max_rank = degree_rank[d]
                best = d
        if best:
            result["highest_degree"] = best
            result["extraction_meta"]["warnings"].append("highest_degree 由 educations 兜底推断")

    # 4) extraction_meta 元数据回填
    result["extraction_meta"]["source_file"] = source_file or result["extraction_meta"].get("source_file")
    result["extraction_meta"]["model"] = MODEL_NAME
    result["extraction_meta"]["extracted_at"] = datetime.now(timezone.utc).isoformat()

    return result


# ===== 主流程 =====
def parse_resume(pdf_path: str) -> dict:
    """主流程：PDF -> 文本 -> LLM -> JSON -> 校验。解析失败返回空字典。"""
    # 步骤 1：从 PDF 提取全文
    resume_text = extract_pdf_text(pdf_path)
    if not resume_text:
        print("[ERROR] PDF 文本为空，无法解析")
        return {}

    # 步骤 2：构造 Prompt
    user_prompt = build_user_prompt(resume_text)

    # 步骤 3：调用 LLM
    try:
        raw_output = call_llm(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        print(f"[ERROR] LLM 调用失败：{e}")
        return {}

    # 步骤 4：解析 JSON
    parsed = safe_json_parse(raw_output)
    if parsed is None:
        return {}

    # 步骤 5：结构校验与补默认值
    return validate_resume_profile(parsed, source_file=os.path.basename(pdf_path))


def main():
    """命令行入口：python resume_parser.py <pdf_path>
    解析结果同时打印到 stdout，并写入当前工作目录下的 `<pdf_basename>.json`。
    """
    parser = argparse.ArgumentParser(description="Resume Parser — PDF 简历转结构化 JSON")
    parser.add_argument("pdf_path", help="待解析的 PDF 简历文件路径")
    parser.add_argument(
        "-o", "--output",
        help="输出 JSON 文件路径；默认保存到当前目录下 <pdf_basename>.json",
        default=None,
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"[ERROR] 文件不存在：{args.pdf_path}")
        sys.exit(1)

    result = parse_resume(args.pdf_path)
    json_str = json.dumps(result, ensure_ascii=False, indent=2)

    # 打印到 stdout
    print(json_str)

    # 写入当前工作目录（输入文件基名 + .json）
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.pdf_path))[0]
        out_path = os.path.join(os.getcwd(), f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"\n[INFO] 结果已保存到：{out_path}")


if __name__ == "__main__":
    main()
