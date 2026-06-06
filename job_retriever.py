"""
岗位知识库检索模块（Job Retriever）

输入：Resume Profile JSON（用户画像）
输出：Top-K 候选岗位列表（按 retrieval_score 降序），供后续 Match Scorer 精排。

流程：
    1. LLM 根据用户画像生成 3 个不同侧重的检索 Query（技能 / 项目 / 求职意向），失败时规则兜底；
    2. 对每个 Query 取 embedding，在本地 ChromaDB（./job_db, collection=jobs）做向量检索；
    3. 支持 city / direction / education 元数据过滤，过滤后为空时逐级降级重试；
    4. 合并三组结果，按 job_id 去重（保留最高分、合并 matched_terms）；
    5. 按 retrieval_score 降序返回 Top-K。

设计取舍：
    - retrieval_score 由 ChromaDB 的 distance 转换而来（1/(1+distance)），保证“越大越相关”；
    - 召回环节追求覆盖率而非精度，因此 per_query_k = top_k * 2，并对过滤做降级兜底；
    - Query 中的「编程语言 / 框架 / 数据库 / 工具」必须在画像中显式出现，禁止 LLM 凭常识补充；
    - matched_terms 引入轻量同义词表，提升“为什么召回”的解释力；
    - 任意单个 Query 失败（embedding / 查询异常）只跳过该 Query，绝不让整体崩溃。
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
from typing import Optional

import chromadb
from openai import OpenAI

# 复用 indexer 中已实现的 embedding 函数，保持向量空间一致
from job_indexer import get_embedding, DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME


# ===== 模型 / 接口常量 =====
MODEL_NAME = "qwen-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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


# ===== Query 生成 Prompt =====
SYSTEM_PROMPT_QUERY = """你是一个招聘搜索引擎 Query 生成器。请根据候选人画像生成适合岗位检索的中文关键词 Query。

要求：
1. 只能输出合法 JSON；
2. 不要输出 Markdown；
3. 不要输出解释文字；
4. 每个 Query 由 6~12 个关键词组成；
5. 关键词之间用空格分隔；
6. 不要编造候选人完全没有依据的技能；
7. 编程语言、框架、数据库、工具名称必须在候选人画像中显式出现才可使用；不得凭常识补充 Python、PyTorch、TensorFlow、Java、C++、MySQL 等显式技能；
8. 可以做合理泛化，但只能泛化为“领域 / 能力层级词”，例如：
   GATv2 → 图神经网络 / GNN；
   Transformer-MoE → Transformer / 深度学习 / 多模态；
   LoRA → 参数高效微调；
   Adapter → 表征对齐；
   多模态融合 → 多模态算法；
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
        "请根据以下候选人画像，生成 3 个不同侧重的岗位检索 Query。\n\n"
        "===== 候选人画像 =====\n"
        f"{payload_str}\n\n"
        "===== 三个 Query 的侧重 =====\n"
        "query_1（技能侧重）：核心技能、技术栈、框架工具、求职城市；\n"
        "query_2（项目侧重）：项目关键词、项目技术栈、任务方向、适合的岗位方向；\n"
        "query_3（求职意向兜底）：求职意向、目标方向、城市、泛化岗位关键词。\n\n"
        "===== 关键约束 =====\n"
        "编程语言 / 框架 / 数据库 / 工具名称必须在上面画像中显式出现才可写入 Query；\n"
        "不得凭常识补充画像中没有的 Python / PyTorch / TensorFlow 等显式技能；\n"
        "项目里的具体技术只能泛化为领域 / 能力词。\n\n"
        "===== 输出 JSON Schema =====\n"
        '{\n'
        '  "query_1": "关键词1 关键词2 关键词3",\n'
        '  "query_2": "关键词1 关键词2 关键词3",\n'
        '  "query_3": "关键词1 关键词2 关键词3"\n'
        '}\n\n'
        "每个 Query 6~12 个关键词，空格分隔，不要标点，不要解释，不要 Markdown，只输出 JSON 本体。"
    )


def _clean_llm_json(raw: str) -> str:
    """清洗 LLM 输出：去 ```json / ``` 包装，截取首 '{' 到末 '}'。"""
    if not raw:
        return ""
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return t


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
    """规则兜底：LLM 不可用 / 解析失败时，用画像字段拼出 3 个 Query。

    规则兜底只取自画像显式字段，天然满足“无据不生成显式技能”的约束。
    """
    skills = _collect_resume_skills(resume_profile)
    project_terms = _collect_project_terms(resume_profile)
    job_pref = resume_profile.get("job_preferences") if isinstance(resume_profile.get("job_preferences"), dict) else {}
    intentions = _flat_strings(job_pref.get("intentions"))
    locations = _collect_resume_locations(resume_profile)

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

    q1 = _compose(skills, locations, intentions)                 # 技能侧重
    q2 = _compose(project_terms, skills, intentions)             # 项目侧重
    q3 = _compose(intentions, locations, skills, project_terms)  # 意向兜底

    # 三条都空时给一个最泛化的兜底，避免完全无法检索
    if not (q1 or q2 or q3):
        fallback = _normalize_query_text(" ".join(skills + intentions + locations)) or "算法 工程师 实习"
        q1 = q2 = q3 = fallback

    return {
        "query_1": q1 or q3 or q2,
        "query_2": q2 or q1 or q3,
        "query_3": q3 or q1 or q2,
    }


def generate_queries(resume_profile: dict, verbose: bool = False) -> dict:
    """Step 1：调用 LLM 生成 3 个检索 Query；任何失败都规则兜底。

    无论 LLM 还是规则兜底，最终都会剔除“无据可依的显式技能词”。
    verbose=True 时打印 LLM 原始输出，便于调试。
    """
    resume_blob = _build_resume_explicit_blob(resume_profile)
    fallback = rule_based_queries(resume_profile)

    def _postprocess(queries: dict) -> dict:
        """统一规整 + 剔除无据显式技能。"""
        out = {}
        for key in ("query_1", "query_2", "query_3"):
            cleaned = _normalize_query_text(queries.get(key))
            cleaned = _filter_ungrounded_tech(cleaned, resume_blob)
            out[key] = cleaned
        return out

    fallback = _postprocess(fallback)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[WARN] 环境变量 DASHSCOPE_API_KEY 未设置，使用规则兜底 Query。")
        return fallback

    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_QUERY},
                {"role": "user", "content": build_query_user_prompt(resume_profile)},
            ],
            temperature=0.2,  # 略带多样性，但仍保持稳定结构
        )
        raw = (completion.choices[0].message.content or "").strip()
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

    # 逐条规整 + 剔除无据显式技能；缺失或被清空的 query 用规则兜底补齐
    result = {}
    for key in ("query_1", "query_2", "query_3"):
        cleaned = _normalize_query_text(parsed.get(key))
        cleaned = _filter_ungrounded_tech(cleaned, resume_blob)
        result[key] = cleaned if cleaned else fallback.get(key, "")

    if not any(result.values()):
        print("[WARN] LLM Query 解析失败，使用规则兜底 Query。")
        return fallback

    return result


def _format_queries(queries: dict) -> str:
    """把最终 Query 排版成多行，便于终端阅读。"""
    lines = ["[INFO] 最终检索 Query："]
    for key in ("query_1", "query_2", "query_3"):
        lines.append(f"  {key}: {queries.get(key, '')}")
    return "\n".join(lines)


# ===== Step 2：metadata 过滤条件构造 =====
def build_where_filter(city_filter: Optional[str],
                       direction_filter: Optional[str],
                       education_filter: Optional[str]) -> Optional[dict]:
    """构造符合 ChromaDB 语法的 where 过滤条件。

    - 单条件直接 {field: value}；多条件用 {"$and": [...]}；
    - 无任何条件返回 None（表示不传 where）。
    """
    conditions = []
    if city_filter and str(city_filter).strip():
        conditions.append({"city": str(city_filter).strip()})
    if direction_filter and str(direction_filter).strip():
        conditions.append({"direction": str(direction_filter).strip()})
    if education_filter and str(education_filter).strip():
        conditions.append({"education_level": str(education_filter).strip()})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _describe_filter(city: Optional[str], direction: Optional[str], education: Optional[str]) -> str:
    """生成人类可读的过滤级别描述，无任何条件时返回「无过滤」。"""
    parts = []
    if city and str(city).strip():
        parts.append(f"city={str(city).strip()}")
    if direction and str(direction).strip():
        parts.append(f"direction={str(direction).strip()}")
    if education and str(education).strip():
        parts.append(f"education={str(education).strip()}")
    return ", ".join(parts) if parts else "无过滤"


def _build_filter_cascade(city_filter: Optional[str],
                          direction_filter: Optional[str],
                          education_filter: Optional[str]) -> list:
    """构造降级重试的过滤级联（从严到松）：

    1. 全部过滤（city + direction + education）；
    2. 去掉 direction（方向表述差异大，最容易导致空召回）；
    3. 完全不过滤。

    仅保留彼此不同的级别，避免重复检索。每级返回 (desc, where, effective_tuple)。
    """
    levels = [
        (city_filter, direction_filter, education_filter),
        (city_filter, None, education_filter),  # 去方向
        (None, None, None),                      # 不过滤
    ]
    cascade, seen = [], []
    for city, direction, education in levels:
        where = build_where_filter(city, direction, education)
        key = json.dumps(where, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.append(key)
        desc = _describe_filter(city, direction, education)
        cascade.append((desc, where, (city, direction, education)))
    return cascade


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
def _query_once(collection, query_text: str, where: Optional[dict], n_results: int) -> list:
    """对单个 Query 执行一次向量检索，返回标准化后的命中列表。

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

    # 2) 查询 ChromaDB（where=None 时不传，避免空条件报错）
    try:
        kwargs = {
            "query_embeddings": [vector],
            "n_results": max(1, n_results),
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
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

        # 还原 jd_profile（jd_profile_json 解析失败则空字典）
        jd_profile = {}
        raw_profile = metadata.get("jd_profile_json")
        if isinstance(raw_profile, str) and raw_profile.strip():
            try:
                loaded = json.loads(raw_profile)
                if isinstance(loaded, dict):
                    jd_profile = loaded
            except json.JSONDecodeError:
                jd_profile = {}

        hits.append({
            "job_id": job_id,
            "company": metadata.get("company"),
            "title": metadata.get("title"),
            "direction": metadata.get("direction"),
            "city": metadata.get("city"),
            "retrieval_score": score,
            "matched_terms": extract_matched_terms(query_text, document, metadata, jd_profile),
            "document": document or "",
            "metadata": metadata,
            "jd_profile": jd_profile,
        })
    return hits


# ===== Step 3：合并去重 =====
def _merge_hits(all_hits: list, max_terms: int = 8) -> list:
    """同 job_id 仅保留最高分；多 Query 命中时合并 matched_terms（去重保序，限长）。"""
    merged = {}
    for hit in all_hits:
        job_id = hit["job_id"]
        if job_id not in merged:
            merged[job_id] = dict(hit)
            merged[job_id]["matched_terms"] = list(hit.get("matched_terms") or [])[:max_terms]
            continue

        existing = merged[job_id]
        # 合并 matched_terms
        seen = {t.lower() for t in existing["matched_terms"]}
        for t in hit.get("matched_terms") or []:
            if len(existing["matched_terms"]) >= max_terms:
                break
            if t.lower() not in seen:
                seen.add(t.lower())
                existing["matched_terms"].append(t)
        # 保留更高分及其对应的 document / metadata
        if hit["retrieval_score"] > existing["retrieval_score"]:
            existing["retrieval_score"] = hit["retrieval_score"]
            existing["document"] = hit["document"]
            existing["metadata"] = hit["metadata"]
            existing["jd_profile"] = hit["jd_profile"]
            existing["company"] = hit["company"]
            existing["title"] = hit["title"]
            existing["direction"] = hit["direction"]
            existing["city"] = hit["city"]
    return list(merged.values())


# ===== 主函数 =====
def retrieve_jobs(
    resume_profile: dict,
    top_k: int = 5,
    city_filter: Optional[str] = None,
    direction_filter: Optional[str] = None,
    education_filter: Optional[str] = None,
    verbose: bool = False,
) -> list:
    """根据用户画像召回 Top-K 候选岗位。

    返回按 retrieval_score 降序排序的完整候选列表（含 document/metadata/jd_profile）；
    无结果时返回 []。verbose 仅影响日志详细度，不改变返回结构。
    """
    if not isinstance(resume_profile, dict):
        print("[ERROR] resume_profile 必须是 dict，返回空列表。")
        return []

    top_k = max(1, int(top_k) if isinstance(top_k, (int, float)) else 5)
    per_query_k = top_k * 2  # 每个 Query 多召回，去重后避免不足

    # 0) 连接 ChromaDB
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

    # 1) 生成 Query 并打印最终用于检索的 Query
    queries = generate_queries(resume_profile, verbose=verbose)
    print(_format_queries(queries))

    # 2) 过滤级联：从严到松，直到召回到结果
    cascade = _build_filter_cascade(city_filter, direction_filter, education_filter)

    merged, used_desc = [], "无过滤"
    for level_idx, (desc, where, effective) in enumerate(cascade):
        if level_idx > 0:
            # 根据本级实际生效的条件给出更贴切的降级提示
            eff_city, eff_dir, eff_edu = effective
            if eff_dir is None and (eff_city or eff_edu):
                print("[WARN] 方向过滤结果为空，降级为仅城市过滤重试。")
            else:
                print("[WARN] 过滤结果为空，降级为无过滤重试。")

        all_hits = []
        for qkey in ("query_1", "query_2", "query_3"):
            query_text = queries.get(qkey, "")
            if not query_text:
                continue
            print(f"[INFO] 执行检索：{qkey} -> {query_text}")
            hits = _query_once(collection, query_text, where, per_query_k)
            print(f"[INFO] 检索返回：{len(hits)} 条")
            all_hits.extend(hits)

        merged = _merge_hits(all_hits)
        if merged:
            used_desc = desc
            print(f"[INFO] 合并去重后：{len(merged)} 条（过滤级别：{desc}）")
            break

    if not merged:
        print("[INFO] 所有 Query 与过滤级别均无召回结果，返回空列表。")
        return []

    # 3) 排序 + 截断 Top-K
    merged.sort(key=lambda h: h["retrieval_score"], reverse=True)
    result = merged[:top_k]
    print(f"[INFO] 最终返回 Top-{top_k}：{len(result)} 条（过滤级别：{used_desc}）")
    return result


# ===== 输出整形 =====
def to_summary(results: list) -> list:
    """把完整结果压缩为终端友好的摘要（去掉 document / metadata / jd_profile）。"""
    summary = []
    for r in results:
        summary.append({
            "job_id": r.get("job_id"),
            "company": r.get("company"),
            "title": r.get("title"),
            "direction": r.get("direction"),
            "city": r.get("city"),
            "retrieval_score": round(float(r.get("retrieval_score") or 0.0), 4),
            "matched_terms": r.get("matched_terms") or [],
        })
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
    parser.add_argument("--city", default=None, help="城市过滤，例如 北京")
    parser.add_argument("--direction", default=None, help="岗位方向过滤，例如 大模型算法")
    parser.add_argument("--education", default=None, help="学历过滤，例如 本科 / 硕士")
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

    results = retrieve_jobs(
        resume_profile,
        top_k=args.top_k,
        city_filter=args.city,
        direction_filter=args.direction,
        education_filter=args.education,
        verbose=args.verbose,
    )

    # 默认摘要输出；--verbose 输出完整结构
    payload = results if args.verbose else to_summary(results)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
