"""
岗位知识库检索模块（Job Retriever）

输入：Resume Profile JSON（用户画像）
输出：Top-K 候选岗位列表，供后续 Match Scorer 精排。

三段式架构（召回 → 融合粗排 → 精排）：
    1. 多 Query 改写（scenario_overview.md §3）：
       - Query1（方向标签）：用户高层能力方向（搜广推 / 后训练 / 大模型应用…）；
       - Query2（实践细节）：项目 / 实习实际任务与技术实践（SFT / DPO / RAG / 多路召回…）；
       Query1/Query2 由 LLM 据画像生成，失败时规则兜底；
       - Query3（方向偏好，**仅在有明确偏好时生成**）：表达用户「想做什么方向」，
         来源优先级＝当前输入显式方向偏好（preferences） > 简历求职意向（intentions），
         两者皆无则不生成。Query3 不承载城市/学历/岗位类型等硬约束，只增强目标方向召回；
       通用基础能力（Python/PyTorch/Linux…）不进任何 Query。
    2. 双通道召回（mode=hybrid，默认）：
       - 向量通道：query embedding 在本地 ChromaDB（data/chroma_db, collection=jobs）语义检索；
       - BM25 通道：jieba 分词 + BM25Okapi 关键词检索（同一份 index_text 语料）；
       mode=vector / bm25 时只走单通道（供评测对比）；
    3. RRF 融合：所有（Query × 通道）的有序命中按 1/(k+rank) 融合去重；
    4. rerank 精排（use_rerank，默认开）：gte-rerank 交叉编码器对头部候选重排；
    5. 截断 Top-K 返回。

硬约束（城市 / 学历 / 岗位类型）在**召回前**做 eligibility 预筛（jobs_store.get_eligible_jobs 按
SQLite 派生列资格筛选），得 allowed_job_ids，三路 Query 共用（向量 Chroma where job_id $in、
BM25 集合过滤）；无 eligible 直接返回空。方向不作硬过滤（仅 Query3 + 评分层 direction_bonus）。

设计取舍：
    - RRF 只依赖名次，BM25（无界分数）与向量相似度（0~1）无需量纲对齐即可融合；
    - 硬约束召回前预筛，故 per_query_k = top_k * 2 即可（命中天然只含 eligible，无需扩池兜底）；
    - Query 中的「编程语言 / 框架 / 数据库 / 工具」必须在画像中显式出现，禁止 LLM 凭常识补充；
    - matched_terms 引入轻量同义词表，提升“为什么召回”的解释力；
    - 任意单个 Query / 单通道失败只跳过该路，绝不让整体崩溃；rerank 失败保持原排序。
"""

import os

# 在导入 chromadb / grpc 之前尽量压低底层 C++ 日志噪音
# （如 "Metric with name 'grpc.resource_quota...' registered more than once"）。
# 仅设置环境变量，不引入额外依赖；设置失败也不影响功能与正常错误日志。
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import re
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Optional

import chromadb

# 统一 Embedding / LLM 工具（core 层单一事实来源，与建库向量空间一致）
from resume2job.core.llm import (
    get_embedding,
    call_llm as _core_call_llm,
    clean_llm_json_output as _clean_llm_json,
)
from resume2job.retrieval.indexer import DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME

# 混合检索三件套：BM25 稀疏通道 / RRF 融合 / rerank 精排
from resume2job.core import config
from resume2job.retrieval.bm25 import get_bm25_corpus
from resume2job.retrieval.fusion import rrf_fuse
from resume2job.retrieval.rerank import rerank_hits
# SQLite 事实源：检索通道只回 job_id + 分数，业务数据在此批量回填（hydration）
from resume2job.storage import jobs_store
# 硬约束归一（城市 / 岗位类型）；学历分级 DEGREE_RANK 供节点把候选人学历映射成 rank
from resume2job.parsing.jd_parser import normalize_city, normalize_job_type, DEGREE_RANK
# 请求级链路追踪（Stage 2）：记录各阶段候选与分数快照；无活跃 trace 时全部 no-op
from resume2job.observability import events


# ===== 请求级硬约束对象（城市 / 学历 / 岗位类型）=====
@dataclass(frozen=True)
class HardConstraints:
    """一次召回请求的硬约束（召回前 eligibility 预筛用）。"""
    city: Optional[str]               # 规范化城市；None=不限城市
    user_degree_rank: Optional[int]   # 候选人学历 rank（本1硕2博3）；None=未知（学历不卡）
    job_type: str                     # 实习 / 校招 / 社招


def resolve_hard_constraints(city, user_degree_rank, user_job_type) -> HardConstraints:
    """把原始入参规整成 HardConstraints（城市去「市」后缀、岗位类型归一三桶、缺省实习）。"""
    return HardConstraints(
        city=normalize_city(city) if city else None,
        user_degree_rank=user_degree_rank,
        job_type=normalize_job_type(user_job_type) if user_job_type else "实习",
    )


# ===== 模型 / 接口常量 =====
from resume2job.core.config import CHAT_MODEL as MODEL_NAME


# ===== 显式技能词表（编程语言 / 框架 / 数据库 / 工具）=====
# 这些词只有在画像中“显式出现”才允许进入 Query；不得由 LLM 凭常识补充。
# 全部小写，匹配时对 query token 做小写精确比较。
EXPLICIT_TECH_VOCAB = {
    # 编程语言
    "python", "java", "c", "c++", "c#", "go", "golang", "rust", "scala",
    "javascript", "typescript", "php", "ruby", "kotlin", "swift", "matlab", "r",
    # 深度学习 / 机器学习框架
    "pytorch", "tensorflow", "tf", "keras", "mindspore", "paddlepaddle", "paddle",
    "jax", "mxnet", "caffe", "sklearn", "scikit-learn", "xgboost", "lightgbm",
    "numpy", "pandas", "scipy", "huggingface", "transformers",
    # 大模型 / 检索工具
    "langchain", "langgraph", "llamaindex", "faiss", "milvus", "chroma", "chromadb",
    "pinecone", "weaviate", "vllm", "deepspeed", "megatron",
    # 数据库
    "mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite", "oracle",
    "elasticsearch", "clickhouse", "hbase", "neo4j", "sql",
    # 工程 / 工具
    "docker", "kubernetes", "k8s", "linux", "git", "fastapi", "flask", "django",
    "spark", "hadoop", "flink", "kafka", "airflow", "grpc", "nginx",
}


# ===== matched_terms 同义词表（轻量、双向）=====
# 用于在“岗位文本中没有出现 query 原词，但出现其同义词”时补充解释。
TERM_SYNONYMS = {
    "GNN": ["图神经网络", "GAT", "GATv2", "GCN"],
    "图神经网络": ["GNN", "GAT", "GATv2", "GCN"],
    "多模态": ["多模态融合", "多模态算法", "跨模态", "跨模态对齐"],
    "多模态融合": ["多模态", "跨模态", "多模态算法"],
    "多模态算法": ["多模态", "多模态融合", "跨模态"],
    "大模型": ["LLM", "大语言模型", "大模型算法", "大模型应用"],
    "大模型算法": ["大模型", "LLM", "大语言模型"],
    "大模型应用": ["RAG", "Agent", "智能体", "工具调用", "大模型"],
    "Agent": ["智能体", "工具调用", "Agent工作流"],
    "智能体": ["Agent", "工具调用", "Agent工作流"],
    "RAG": ["检索增强生成", "向量检索", "召回排序", "大模型应用"],
    "NLP": ["自然语言处理"],
    "自然语言处理": ["NLP"],
    "CV": ["计算机视觉"],
    "计算机视觉": ["CV"],
    "Transformer": ["大模型", "深度学习"],
    "深度学习": ["DL", "neural network", "神经网络"],
    "强化学习": ["RL", "reinforcement learning"],
    "参数高效微调": ["LoRA", "Adapter", "PEFT", "微调"],
    "表征对齐": ["Adapter", "表征空间", "对齐"],
    "时序建模": ["时间序列", "LSTM", "时序演化"],
}


# ===== resume_profile 字段安全提取 =====
def _as_list(value) -> list:
    """非列表返回空列表，过滤掉 None。"""
    if not isinstance(value, list):
        return []
    return [v for v in value if v is not None]


def _flat_strings(value) -> list:
    """把任意值压平成字符串列表，便于拼 query。"""
    out = []
    for v in _as_list(value):
        if isinstance(v, (str, int, float)):
            s = str(v).strip()
            if s:
                out.append(s)
    return out


def _dedup_keep_order(items: list) -> list:
    """字符串列表去重保序（按小写判重）。"""
    seen, out = set(), []
    for x in items:
        if not isinstance(x, str):
            continue
        key = x.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x.strip())
    return out


def _collect_resume_locations(resume_profile: dict) -> list:
    """汇总候选人城市信号：job_preferences.locations + current_location。"""
    job_pref = resume_profile.get("job_preferences") if isinstance(resume_profile.get("job_preferences"), dict) else {}
    locations = _flat_strings(job_pref.get("locations"))
    cur = resume_profile.get("current_location")
    if isinstance(cur, str) and cur.strip():
        locations.append(cur.strip())
    return _dedup_keep_order(locations)


def _resume_intentions(resume_profile: dict) -> list:
    """取简历求职意向（job_preferences.intentions）——Query3 的次优来源。"""
    job_pref = resume_profile.get("job_preferences") if isinstance(resume_profile.get("job_preferences"), dict) else {}
    return _dedup_keep_order(_flat_strings(job_pref.get("intentions")))


def _collect_resume_skills(resume_profile: dict) -> list:
    """从顶层 skills + skill_groups[].items 汇总技能标签。"""
    skills = _flat_strings(resume_profile.get("skills"))
    sg = resume_profile.get("skill_groups")
    if isinstance(sg, list):
        for group in sg:
            if isinstance(group, dict):
                skills.extend(_flat_strings(group.get("items")))
    return _dedup_keep_order(skills)


def _collect_project_terms(resume_profile: dict) -> list:
    """从 projects 汇总项目关键词 / 技术栈 / 名称。"""
    terms = []
    for proj in _as_list(resume_profile.get("projects")):
        if not isinstance(proj, dict):
            continue
        terms.extend(_flat_strings(proj.get("keywords")))
        terms.extend(_flat_strings(proj.get("tech_stack")))
        if isinstance(proj.get("name"), str) and proj["name"].strip():
            terms.append(proj["name"].strip())
    return _dedup_keep_order(terms)


def _build_resume_explicit_blob(resume_profile: dict) -> str:
    """汇总画像中所有“显式出现过”的文本，用于校验技能词是否有据可依。

    覆盖：skills / skill_groups.items / projects.tech_stack / projects.keywords /
          experiences.keywords / research.keywords。统一小写。
    """
    pool = []
    pool.extend(_collect_resume_skills(resume_profile))
    pool.extend(_collect_project_terms(resume_profile))
    for exp in _as_list(resume_profile.get("experiences")):
        if isinstance(exp, dict):
            pool.extend(_flat_strings(exp.get("keywords")))
    for r in _as_list(resume_profile.get("research")):
        if isinstance(r, dict):
            pool.extend(_flat_strings(r.get("keywords")))
    return "\n".join(pool).lower()


# ===== 显式技能过滤：剔除无据可依的语言 / 框架 / 数据库 / 工具 =====
def _filter_ungrounded_tech(query_text: str, resume_blob: str) -> str:
    """从单条 query 中剔除“属于显式技能词表、但未在画像中出现”的 token。

    例：画像里没有 PyTorch 时，LLM 生成的 'PyTorch' 会被移除；
        领域 / 能力词（如 图神经网络 / 多模态 / 深度学习）不在词表中，保留。
    """
    kept = []
    for tok in query_text.split():
        low = tok.strip().lower()
        if not low:
            continue
        if low in EXPLICIT_TECH_VOCAB and low not in resume_blob:
            # 无据可依的显式技能，丢弃
            continue
        kept.append(tok)
    return " ".join(kept)


def _strip_general_base_skills(query_text: str) -> str:
    """无条件剔除通用基础能力词（Python/PyTorch/Linux/...）——区分度低，不进 Query 构造。"""
    kept = [tok for tok in query_text.split()
            if tok.strip().lower() not in GENERAL_BASE_SKILLS]
    return " ".join(kept)


# ===== 通用基础能力（不参与 Query 构造）=====
# 语言 / 框架 / 开发环境等区分度低，许多岗位仅要求其中若干项；若进 Query 会把召回
# 过度拉向泛算法 / 泛开发岗。这些词无条件从 Query 中剔除，仅在召回后做技能校验
# （见 job_matching_and_ranking.md §3「通用基础能力处理」）。全部小写。
GENERAL_BASE_SKILLS = {
    "python", "pytorch", "torch", "linux", "git", "c", "c++", "sql", "docker",
    "tensorflow", "tf", "numpy", "pandas",
}


# ===== Query 生成 Prompt（两路：方向标签 / 实践细节）=====
SYSTEM_PROMPT_QUERY = """你是一个招聘搜索引擎 Query 生成器。请根据候选人画像生成两路不同侧重的中文关键词 Query。

两路 Query 的侧重：
- query_1（方向标签）：表达候选人的高层能力方向，如搜广推、后训练、大模型应用、强化学习、多模态等；
- query_2（实践细节）：表达项目 / 实习中实际完成的任务、方法与技术实践，如 SFT、DPO、RAG、多路召回、自动化评测、轨迹规划等。

要求：
1. 只能输出合法 JSON；
2. 不要输出 Markdown；
3. 不要输出解释文字；
4. 每个 Query 由 6~12 个关键词组成，关键词之间用空格分隔；
5. 允许两路 Query 有少量必要语义重叠，但避免整批关键词重复堆叠；同一术语同时出现在技能与项目中时，
   query_1 保留方向性表达、query_2 保留更细粒度的实践描述；
6. 不要编造候选人完全没有依据的技能；
7. **Python、PyTorch、Linux、Git、C++、SQL、Docker 等语言/框架/开发环境属于通用基础能力，区分度低，禁止写入任何 Query**；
8. 可以做合理泛化，但只能泛化为“领域 / 能力层级词”，例如：
   GATv2 → 图神经网络 / GNN；Transformer-MoE → 深度学习 / 多模态；LoRA → 参数高效微调；多模态融合 → 多模态算法；
9. 生成的 Query 应适合在岗位知识库中进行向量检索。"""


# ===== Step 1：Query 生成 =====
def build_query_user_prompt(resume_profile: dict) -> str:
    """构造 Query 生成的 User Prompt：求职意向 / 技能 / 项目 / 学历 / 城市 + 输出 Schema。"""
    job_pref = resume_profile.get("job_preferences") if isinstance(resume_profile.get("job_preferences"), dict) else {}
    intentions = _flat_strings(job_pref.get("intentions"))
    skills = _collect_resume_skills(resume_profile)
    locations = _collect_resume_locations(resume_profile)
    degree = resume_profile.get("highest_degree") or "未知"

    # 项目摘要：名称 + 关键词 + 技术栈，便于 LLM 做方向泛化
    projects_brief = []
    for proj in _as_list(resume_profile.get("projects"))[:5]:
        if not isinstance(proj, dict):
            continue
        projects_brief.append({
            "name": proj.get("name"),
            "keywords": _flat_strings(proj.get("keywords")),
            "tech_stack": _flat_strings(proj.get("tech_stack")),
        })

    payload = {
        "求职意向": intentions,
        "技能列表": skills,
        "项目经历": projects_brief,
        "最高学历": degree,
        "期望城市或当前位置": locations,
    }
    payload_str = json.dumps(payload, ensure_ascii=False, indent=2)

    return (
        "请根据以下候选人画像，生成 2 路不同侧重的岗位检索 Query。\n\n"
        "===== 候选人画像 =====\n"
        f"{payload_str}\n\n"
        "===== 两路 Query 的侧重 =====\n"
        "query_1（方向标签）：候选人的高层能力方向（搜广推 / 后训练 / 大模型应用 / 强化学习 / 多模态 等）+ 求职方向；\n"
        "query_2（实践细节）：项目 / 实习中实际完成的任务、方法与技术实践（SFT / DPO / RAG / 多路召回 / 自动化评测 / 轨迹规划 等）。\n\n"
        "===== 关键约束 =====\n"
        "Python / PyTorch / Linux / Git / C++ / SQL / Docker 等通用基础能力**禁止写入任何 Query**；\n"
        "项目里的具体技术只能泛化为领域 / 能力词；不得凭常识补充画像中没有的技能。\n\n"
        "===== 输出 JSON Schema =====\n"
        '{\n'
        '  "query_1": "方向标签关键词...",\n'
        '  "query_2": "实践细节关键词..."\n'
        '}\n\n'
        "每个 Query 6~12 个关键词，空格分隔，不要标点，不要解释，不要 Markdown，只输出 JSON 本体。"
    )


def _normalize_query_text(text) -> str:
    """把单条 query 规整为空格分隔、无多余标点的关键词串。"""
    if not isinstance(text, str):
        if isinstance(text, list):
            text = " ".join(str(x) for x in text)
        else:
            text = str(text or "")
    # 把常见标点替换为空格，再压缩空白
    text = re.sub(r"[，,、；;。\.／/|\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rule_based_queries(resume_profile: dict) -> dict:
    """规则兜底：LLM 不可用 / 解析失败时，用画像字段拼出 2 路 Query。

    规则兜底只取自画像显式字段，天然满足“无据不生成显式技能”的约束。
    query_1=方向标签（求职意向 + 技能方向），query_2=实践细节（项目关键词 + 技术栈）。
    """
    skills = _collect_resume_skills(resume_profile)
    project_terms = _collect_project_terms(resume_profile)
    job_pref = resume_profile.get("job_preferences") if isinstance(resume_profile.get("job_preferences"), dict) else {}
    intentions = _flat_strings(job_pref.get("intentions"))

    def _compose(*pools, limit=12) -> str:
        merged, seen = [], set()
        for pool in pools:
            for x in pool:
                low = x.lower()
                if x and low not in seen:
                    seen.add(low)
                    merged.append(x)
                if len(merged) >= limit:
                    break
            if len(merged) >= limit:
                break
        return _normalize_query_text(" ".join(merged))

    q1 = _compose(intentions, skills)            # 方向标签侧重
    q2 = _compose(project_terms, skills)         # 实践细节侧重

    # 两条都空时给一个最泛化的兜底，避免完全无法检索
    if not (q1 or q2):
        fallback = _normalize_query_text(" ".join(skills + intentions)) or "算法 工程师 实习"
        q1 = q2 = fallback

    return {
        "query_1": q1 or q2,
        "query_2": q2 or q1,
    }


def generate_queries(resume_profile: dict, verbose: bool = False) -> dict:
    """Step 1：调用 LLM 生成 2 个检索 Query（query_1 方向标签 / query_2 实践细节）；任何失败都规则兜底。
    Query3（方向偏好召回）不在此生成——在 retrieve_jobs 内据 preferences / 简历意向条件构建。

    无论 LLM 还是规则兜底，最终都会剔除“无据可依的显式技能词”与通用基础能力。
    verbose=True 时打印 LLM 原始输出，便于调试。
    """
    resume_blob = _build_resume_explicit_blob(resume_profile)
    fallback = rule_based_queries(resume_profile)

    def _postprocess(queries: dict) -> dict:
        """统一规整 + 剔除无据显式技能 + 剔除通用基础能力。"""
        out = {}
        for key in ("query_1", "query_2"):
            cleaned = _normalize_query_text(queries.get(key))
            cleaned = _filter_ungrounded_tech(cleaned, resume_blob)
            cleaned = _strip_general_base_skills(cleaned)
            out[key] = cleaned
        return out

    fallback = _postprocess(fallback)

    try:
        raw = _core_call_llm(
            SYSTEM_PROMPT_QUERY,
            build_query_user_prompt(resume_profile),
            model=MODEL_NAME,
            temperature=0.2,  # 略带多样性，但仍保持稳定结构
        )
    except Exception as e:
        print(f"[WARN] LLM Query 调用失败，使用规则兜底 Query。原因：{e}")
        return fallback

    if verbose:
        print(f"[DEBUG] LLM Query 原始输出：\n{raw}")

    try:
        parsed = json.loads(_clean_llm_json(raw))
    except json.JSONDecodeError:
        print("[WARN] LLM Query 解析失败，使用规则兜底 Query。")
        return fallback

    if not isinstance(parsed, dict):
        print("[WARN] LLM Query 解析失败，使用规则兜底 Query。")
        return fallback

    # 逐条规整 + 剔除无据显式技能 + 剔除通用基础能力；缺失或被清空的 query 用规则兜底补齐
    result = {}
    for key in ("query_1", "query_2"):
        cleaned = _normalize_query_text(parsed.get(key))
        cleaned = _filter_ungrounded_tech(cleaned, resume_blob)
        cleaned = _strip_general_base_skills(cleaned)
        result[key] = cleaned if cleaned else fallback.get(key, "")

    if not any(result.values()):
        print("[WARN] LLM Query 解析失败，使用规则兜底 Query。")
        return fallback

    return result


def _format_queries(queries: dict) -> str:
    """把最终 Query 排版成多行，便于终端阅读。"""
    lines = ["[INFO] 最终检索 Query："]
    for key in ("query_1", "query_2"):
        lines.append(f"  {key}: {queries.get(key, '')}")
    return "\n".join(lines)


# ===== 硬约束（城市 / 学历 / 岗位类型）=====
# 不再做召回后 post-filter，改为**召回前** eligibility 预筛（jobs_store.get_eligible_jobs，按 SQLite
# 派生列 cities_json/job_types_json/min_degree_rank/*_status 资格筛选），得 allowed_job_ids，
# 三路 Query 共用；向量走 Chroma where job_id $in、BM25 走集合过滤。方向不作硬过滤（仅 Query3 +
# direction_bonus）；偏好排序统一到评分层 direction_bonus，检索阶段不做 α-blend。


# ===== distance -> similarity =====
def _distance_to_score(distance) -> float:
    """ChromaDB 返回的是 distance（越小越相似），转换为 retrieval_score（越大越相似）。"""
    if not isinstance(distance, (int, float)):
        return 0.0
    return 1.0 / (1.0 + float(distance))


# ===== matched_terms 提取（精确 + 同义词归一化）=====
def _build_haystack(document: str, metadata: dict, jd_profile: dict) -> str:
    """把可用于关键词命中的文本拼成一个小写 haystack。

    覆盖：document + company + title + direction + business_area
          + hard_skills + preferred_skills + domain_keywords + tools_or_frameworks。
    metadata 中的 *_json 字段本身是 JSON 字符串，可直接并入文本做子串匹配；
    preferred_skills 不在扁平 metadata 中，从还原后的 jd_profile 取。
    """
    parts = [document or ""]
    for field in ("company", "title", "direction", "business_area",
                  "hard_skills", "domain_keywords", "tools_or_frameworks"):
        val = metadata.get(field)
        if isinstance(val, str):
            parts.append(val)
    # preferred_skills 来自 jd_profile（document 的“加分项”也含它，这里再补一次更稳）
    if isinstance(jd_profile, dict):
        pref = jd_profile.get("preferred_skills")
        if isinstance(pref, list):
            parts.append(" ".join(str(x) for x in pref))
    return "\n".join(parts).lower()


def _expand_synonyms(token: str) -> list:
    """为一个 query token 生成同义 / 归一候选词（含自身）。

    采用“子串关联”一跳扩展：若同义词表的 key 与 token 互为子串，
    则把该 key 及其同义词都纳入候选。这样 '大模型应用算法' 能关联到
    大模型 / RAG / 智能体 等标准词。
    """
    candidates = [token]
    low = token.lower()
    for key, syns in TERM_SYNONYMS.items():
        klow = key.lower()
        if klow == low or klow in low or low in klow:
            candidates.append(key)
            candidates.extend(syns)
    return candidates


def extract_matched_terms(query_text: str, document: str, metadata: dict,
                          jd_profile: dict, max_terms: int = 8) -> list:
    """从 query 关键词出发，结合同义词，挑出在岗位文本中出现的命中词。

    规则：
      1. query 原词直接出现 → 命中（加原词）；
      2. query 原词未直接出现，但其同义 / 归一词出现在岗位文本 → 命中（加该同义词）；
      3. 去重保序，最多返回 max_terms 个。
    """
    haystack = _build_haystack(document, metadata, jd_profile)
    matched, seen = [], set()

    def _add(term: str):
        key = term.strip().lower()
        if key and key not in seen and len(matched) < max_terms:
            seen.add(key)
            matched.append(term.strip())

    for tok in (query_text or "").split():
        tok = tok.strip()
        if not tok:
            continue
        if len(matched) >= max_terms:
            break
        for cand in _expand_synonyms(tok):
            if cand.lower() in haystack:
                _add(cand)
    return matched[:max_terms]


# ===== 单条 Query 检索 =====
def _query_once(collection, query_text: str, n_results: int, allowed_ids=None) -> list:
    """对单个 Query 执行一次向量检索，返回标准化后的命中列表。

    allowed_ids 非空时限定在 eligibility 预筛出的 job_id 集合内（Chroma where job_id $in）。
    任意异常（embedding / 查询）只打印并返回 []，不向上抛。
    """
    if not query_text or not query_text.strip():
        return []

    # 1) 取向量
    try:
        vector = get_embedding(query_text)
    except Exception as e:
        print(f"[ERROR] Embedding 失败：{e}")
        return []

    # 2) 查询 ChromaDB（硬约束召回前预筛 → 限定 allowed_ids；未传则不加 where）
    try:
        kwargs = {
            "query_embeddings": [vector],
            "n_results": max(1, n_results),
            "include": ["documents", "metadatas", "distances"],
        }
        if allowed_ids:
            kwargs["where"] = {"job_id": {"$in": sorted(allowed_ids)}}
        res = collection.query(**kwargs)
    except Exception as e:
        print(f"[ERROR] ChromaDB 查询失败：{e}")
        return []

    return _normalize_query_result(res, query_text)


def _normalize_query_result(res: dict, query_text: str) -> list:
    """把 ChromaDB query() 返回（每字段是 list[list]）拍平成命中 dict 列表。"""
    if not isinstance(res, dict):
        return []

    def _first(key):
        val = res.get(key)
        if isinstance(val, list) and val:
            inner = val[0]
            return inner if isinstance(inner, list) else []
        return []

    ids = _first("ids")
    documents = _first("documents")
    metadatas = _first("metadatas")
    distances = _first("distances")

    hits = []
    for i, job_id in enumerate(ids):
        if not job_id:
            continue
        document = documents[i] if i < len(documents) else ""
        metadata = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
        distance = distances[i] if i < len(distances) else None
        score = _distance_to_score(distance)  # 无 distance -> 0.0，但不崩溃

        hits.append({
            "job_id": job_id,
            "company": metadata.get("company"),
            "title": metadata.get("title"),
            "direction": metadata.get("direction"),
            "city": metadata.get("city"),
            "retrieval_score": score,
            "vector_score": score,
            "matched_terms": [],   # 回填后由 _finalize_hits 统一填充
            "document": document or "",
            "metadata": metadata,
            "jd_profile": {},      # 由 _hydrate_hits 从 SQLite 事实源回填
        })
    return hits


def _hydrate_hits(hits: list) -> None:
    """检索命中回填（hydration）：从 SQLite 事实源批量取回业务数据。

    Chroma / BM25 通道只携带最小过滤字段；jd_profile 与权威的
    company/title/city/direction 在此一次性 IN 查询补齐。
    事实源缺失某 job_id 时（异常情况）保留通道返回的字段，不报错。
    """
    rows = jobs_store.get_jobs_by_ids([h.get("job_id") for h in hits])
    for hit in hits:
        row = rows.get(hit.get("job_id"))
        if not row:
            continue
        hit["jd_profile"] = row.get("jd_profile") or {}
        for key in ("company", "title", "city", "direction"):
            if row.get(key):
                hit[key] = row[key]


def _finalize_hits(hits: list, query_text: str) -> list:
    """单通道命中的统一收尾：SQLite 回填 + matched_terms 解释。"""
    if not hits:
        return hits
    _hydrate_hits(hits)
    for hit in hits:
        hit["matched_terms"] = extract_matched_terms(
            query_text, hit.get("document") or "", hit.get("metadata") or {},
            hit.get("jd_profile") or {},
        )
    return hits


def _bm25_query_once(corpus, query_text: str, n_results: int, allowed_ids=None) -> list:
    """对单个 Query 执行一次 BM25 检索（回填与 matched_terms 由 _finalize_hits 统一处理）。

    allowed_ids 非空时限定在 eligibility 预筛出的 job_id 集合内（语料打分时过滤）。
    任意异常只打印并返回 []，与向量通道的容错约定一致。
    """
    if not query_text or not query_text.strip():
        return []
    try:
        return corpus.search(query_text, n_results, allowed_ids=allowed_ids)
    except Exception as e:
        print(f"[ERROR] BM25 检索失败：{e}")
        return []


def _build_rerank_query(queries: dict) -> str:
    """把 2 路检索 Query 的关键词去重拼成 rerank 用的单一 query 文本。"""
    tokens, seen = [], set()
    for key in ("query_1", "query_2"):
        for tok in (queries.get(key) or "").split():
            low = tok.lower()
            if low and low not in seen:
                seen.add(low)
                tokens.append(tok)
    return " ".join(tokens)


# ===== 单 Query 检索（评测 / 调试用公共入口）=====
def search_jobs(
    query_text: str,
    top_k: int = 5,
    mode: Optional[str] = None,
    use_rerank: Optional[bool] = None,
) -> list:
    """用给定 query 文本直接检索岗位（不经过 LLM query 改写、不做过滤级联）。

    供 eval 层做检索指标评测（同一 query 在 vector / bm25 / hybrid / +rerank
    四种配置下横向对比），也可用于调试。返回结构与 retrieve_jobs 一致。
    """
    if not isinstance(query_text, str) or not query_text.strip():
        return []

    mode = (mode or config.RETRIEVAL_MODE).lower()
    if mode not in ("vector", "bm25", "hybrid"):
        mode = "hybrid"
    if use_rerank is None:
        use_rerank = config.USE_RERANK
    top_k = max(1, int(top_k))

    try:
        client = chromadb.PersistentClient(path=DEFAULT_DB_PATH)
        collection = client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)
    except Exception as e:
        print(f"[ERROR] 连接 ChromaDB 失败：{e}")
        return []

    ranked_lists = []
    if mode in ("vector", "hybrid"):
        hits = _finalize_hits(_query_once(collection, query_text, top_k * 2), query_text)
        if hits:
            ranked_lists.append(hits)
    if mode in ("bm25", "hybrid"):
        try:
            corpus = get_bm25_corpus(collection)
            hits = _finalize_hits(
                _bm25_query_once(corpus, query_text, top_k * 2), query_text)
            if hits:
                ranked_lists.append(hits)
        except Exception as e:
            print(f"[WARN] BM25 通道失败：{e}")

    merged = rrf_fuse(ranked_lists)
    if not merged:
        return []

    candidates = merged[: max(top_k * 2, top_k)]
    if use_rerank:
        candidates = rerank_hits(query_text, candidates, top_n=len(candidates))
    return candidates[:top_k]


# ===== 主函数 =====
def retrieve_jobs(
    resume_profile: dict,
    top_k: int = 5,
    city_filter: Optional[str] = None,
    user_degree_rank: Optional[int] = None,
    job_type_filter: Optional[str] = None,
    verbose: bool = False,
    mode: Optional[str] = None,
    use_rerank: Optional[bool] = None,
    preferences: Optional[dict] = None,
) -> list:
    """根据用户画像召回 Top-K 候选岗位（硬约束预筛 → 召回 → RRF 融合 → rerank 精排）。

    硬约束（城市 / 学历 / 岗位类型）在**召回前**做 eligibility 预筛（jobs_store.get_eligible_jobs），
    得 allowed_job_ids；Query1/Query2/Query3 三路**共用同一批 allowed_job_ids**（向量走 Chroma
    where job_id $in、BM25 走集合过滤），不每路重查 SQLite。无 eligible 岗位直接返回 []。

    user_degree_rank：候选人学历 rank（本1硕2博3）；None=学历不卡。city_filter=None 不限城市。
    job_type_filter 缺省由调用方给（节点默认实习）。每条命中附 city_match_status / education_match_status
    供报告标注（如「地点待确认」）。

    mode：vector / bm25 / hybrid；use_rerank：是否 gte-rerank 精排。二者默认读 core.config。
    preferences：方向偏好 {tag: weight}，喂 Query3（助召回不改名次，排序统一到评分层 direction_bonus）。
    返回按检索相关性降序的完整候选列表；无结果返回 []。
    """
    if not isinstance(resume_profile, dict):
        print("[ERROR] resume_profile 必须是 dict，返回空列表。")
        return []

    mode = (mode or config.RETRIEVAL_MODE).lower()
    if mode not in ("vector", "bm25", "hybrid"):
        print(f"[WARN] 未知检索模式「{mode}」，回退 hybrid。")
        mode = "hybrid"
    if use_rerank is None:
        use_rerank = config.USE_RERANK
    use_vector = mode in ("vector", "hybrid")
    use_bm25 = mode in ("bm25", "hybrid")

    top_k = max(1, int(top_k) if isinstance(top_k, (int, float)) else 5)
    per_query_k = top_k * 2  # 每路多召回，去重后避免不足（预筛已限定 eligible，无需再扩池）

    # 0) 硬约束 eligibility 预筛（召回**前**）：三路 Query 共用一批 allowed_job_ids
    constraints = resolve_hard_constraints(city_filter, user_degree_rank, job_type_filter)
    eligible = jobs_store.get_eligible_jobs(
        constraints.city, constraints.user_degree_rank, constraints.job_type)
    allowed_job_ids = set(eligible)
    events.record_constraint_filter(
        {"city": constraints.city, "user_degree_rank": constraints.user_degree_rank,
         "job_type": constraints.job_type}, allowed_count=len(allowed_job_ids))
    if not allowed_job_ids:
        print(f"[INFO] 硬约束预筛无 eligible 岗位（城市={constraints.city} / "
              f"学历rank={constraints.user_degree_rank} / 类型={constraints.job_type}），返回空。")
        return []
    print(f"[INFO] 硬约束预筛：{len(allowed_job_ids)} 个 eligible 岗位进入召回")

    # 1) 连接 ChromaDB
    if not os.path.isdir(DEFAULT_DB_PATH):
        print(f"[ERROR] ChromaDB 目录不存在：{DEFAULT_DB_PATH}，请先运行 job_indexer.py 建库。")
        return []
    try:
        client = chromadb.PersistentClient(path=DEFAULT_DB_PATH)
        collection = client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)
    except Exception as e:
        print(f"[ERROR] 连接 ChromaDB 失败：{e}")
        return []

    try:
        total = collection.count()
    except Exception as e:
        print(f"[ERROR] 读取 collection 失败：{e}")
        return []
    if total == 0:
        print(f"[ERROR] collection '{DEFAULT_COLLECTION_NAME}' 为空，没有可检索的岗位。")
        return []

    # 2) 生成 Query 并打印最终用于检索的 Query
    queries = generate_queries(resume_profile, verbose=verbose)
    print(_format_queries(queries))

    # 2.5) BM25 通道：从同一 collection 的 documents 构建（带缓存的）内存索引
    bm25_corpus = None
    if use_bm25:
        try:
            bm25_corpus = get_bm25_corpus(collection)
        except Exception as e:
            print(f"[WARN] BM25 索引构建失败，本次仅用向量通道：{e}")
            use_bm25 = False
            if not use_vector:
                print("[ERROR] bm25 模式下索引构建失败，返回空列表。")
                return []

    # 3) 召回：2 路 Query + Query3（方向偏好）各通道有序命中，**均限定 allowed_job_ids**，RRF 融合去重
    ranked_lists = []
    for qkey in ("query_1", "query_2"):
        query_text = queries.get(qkey, "")
        if not query_text:
            continue
        print(f"[INFO] 执行检索：{qkey} -> {query_text}")
        if use_vector:
            hits = _finalize_hits(_query_once(collection, query_text, per_query_k, allowed_job_ids), query_text)
            print(f"[INFO]   向量通道返回：{len(hits)} 条")
            events.record_channel_hits("dense", hits)
            if hits:
                ranked_lists.append(hits)
        if use_bm25:
            hits = _finalize_hits(_bm25_query_once(bm25_corpus, query_text, per_query_k, allowed_job_ids), query_text)
            print(f"[INFO]   BM25 通道返回：{len(hits)} 条")
            events.record_channel_hits("bm25", hits)
            if hits:
                ranked_lists.append(hits)

    # Query3（方向偏好召回，conditional）：来源优先级 preferences > 简历 intentions；皆无则不生成。
    # 只增强目标方向召回（多一路名次支持而上浮），**同样限定 allowed_job_ids**，剔除通用基础能力。
    q3_terms = list((preferences or {}).keys()) or _resume_intentions(resume_profile)
    query_3 = _strip_general_base_skills(_normalize_query_text(" ".join(q3_terms)))
    events.record_retrieval_queries({**queries, "query_3": query_3})  # 记录三路 Query（含偏好召回）
    if query_3.strip():
        print(f"[INFO] 执行方向偏好召回（Query3）：{query_3}")
        if use_vector:
            h = _finalize_hits(_query_once(collection, query_3, per_query_k, allowed_job_ids), query_3)
            events.record_channel_hits("dense", h)
            if h:
                ranked_lists.append(h)
        if use_bm25:
            h = _finalize_hits(_bm25_query_once(bm25_corpus, query_3, per_query_k, allowed_job_ids), query_3)
            events.record_channel_hits("bm25", h)
            if h:
                ranked_lists.append(h)

    merged = rrf_fuse(ranked_lists)
    if not merged:
        print("[INFO] 所有 Query 均无召回结果，返回空列表。")
        return []
    print(f"[INFO] RRF 融合去重后：{len(merged)} 条（模式：{mode}，硬约束已在召回前预筛）")
    events.record_rrf(merged)

    # 3.5) 把资格状态附到命中（city_match_status / education_match_status，供报告标「地点待确认」等）
    for h in merged:
        st = eligible.get(h.get("job_id"))
        if st:
            h["city_match_status"] = st.get("city_match_status")
            h["education_match_status"] = st.get("education_match_status")

    # 4) rerank 精排（可选）：取融合后的头部候选送交叉编码器重排
    candidates = merged[: max(top_k * 2, top_k)]
    if use_rerank:
        rerank_query = _build_rerank_query(queries)
        candidates = rerank_hits(rerank_query, candidates, top_n=len(candidates))
        if candidates and "rerank_score" in candidates[0]:
            print(f"[INFO] rerank 精排完成：{len(candidates)} 条候选已按相关性重排")
        events.record_rerank(candidates)

    # 排序偏好统一到评分层 direction_bonus，检索阶段不再做 α-blend 加权重排。

    # 4) 截断 Top-K
    result = candidates[:top_k]
    print(f"[INFO] 最终返回 Top-{top_k}：{len(result)} 条")
    return result


# ===== 输出整形 =====
def to_summary(results: list) -> list:
    """把完整结果压缩为终端友好的摘要（去掉 document / metadata / jd_profile）。"""
    summary = []
    for r in results:
        item = {
            "job_id": r.get("job_id"),
            "company": r.get("company"),
            "title": r.get("title"),
            "direction": r.get("direction"),
            "city": r.get("city"),
            "retrieval_score": round(float(r.get("retrieval_score") or 0.0), 4),
            "matched_terms": r.get("matched_terms") or [],
        }
        if "rerank_score" in r:
            item["rerank_score"] = round(float(r["rerank_score"]), 4)
        summary.append(item)
    return summary


# ===== CLI =====
def main():
    """命令行入口：
        python job_retriever.py resume_profile.json --top_k 5 --city 北京 --direction 大模型算法 --education 硕士 --verbose

    默认只输出摘要字段；加 --verbose 输出完整字段（document/metadata/jd_profile）。
    """
    parser = argparse.ArgumentParser(description="Job Retriever — 根据用户画像召回 Top-K 候选岗位")
    parser.add_argument("resume_json", help="Resume Profile JSON 文件路径")
    parser.add_argument("--top_k", type=int, default=5, help="最终返回候选岗位数量（默认 5）")
    parser.add_argument("--city", default=None, help="城市硬约束，例如 北京")
    parser.add_argument("--education", default=None,
                        help="候选人学历（本科/硕士/博士）；资格预筛保留要求≤该学历的岗位。缺省取简历 highest_degree")
    parser.add_argument("--job_type", default=None, help="岗位类型硬约束，例如 实习 / 校招 / 社招")
    parser.add_argument("--mode", default=None, choices=["vector", "bm25", "hybrid"],
                        help="检索模式（默认读 RESUME2JOB_RETRIEVAL_MODE，缺省 hybrid）")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 rerank 精排")
    parser.add_argument("--verbose", action="store_true",
                        help="输出完整字段（document/metadata/jd_profile）及调试日志；默认仅输出摘要")
    args = parser.parse_args()

    if not os.path.isfile(args.resume_json):
        print(f"[ERROR] 文件不存在：{args.resume_json}")
        sys.exit(1)

    try:
        with open(args.resume_json, "r", encoding="utf-8") as f:
            resume_profile = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] resume_profile JSON 解析失败：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 读取文件失败：{e}")
        sys.exit(1)

    if not isinstance(resume_profile, dict):
        print("[ERROR] resume_profile 顶层必须是 JSON 对象")
        sys.exit(1)

    edu_str = str(args.education or resume_profile.get("highest_degree") or "").strip()
    results = retrieve_jobs(
        resume_profile,
        top_k=args.top_k,
        city_filter=args.city,
        user_degree_rank=DEGREE_RANK.get(edu_str),
        job_type_filter=args.job_type,
        verbose=args.verbose,
        mode=args.mode,
        use_rerank=False if args.no_rerank else None,
    )

    # 默认摘要输出；--verbose 输出完整结构
    payload = results if args.verbose else to_summary(results)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
