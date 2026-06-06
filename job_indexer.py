"""
岗位知识库构建模块（Job Indexer）

批量解析 jd_folder 下的 .txt JD 文件，向量化后写入本地 ChromaDB，
供后续岗位语义检索 / 匹配评分使用。

流程：
    扫描 .txt -> 用 job_id 查重 -> 读取原文 -> parse_jd -> 构造 index_text + metadata
        -> 获取 embedding -> 写入 chromadb -> 汇总日志
"""

import os
import sys
import json
import argparse
from typing import Optional

import chromadb
from openai import OpenAI

# 复用已有的 JD 解析器
from jd_parser import parse_jd


# ===== 常量 =====
EMBEDDING_MODEL = "text-embedding-v3"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DB_PATH = "./job_db"
DEFAULT_COLLECTION_NAME = "jobs"
DEFAULT_JOB_PATH = './JDS'


# ===== Embedding =====
def get_embedding(text: str) -> list:
    """调用阿里云百炼 text-embedding-v3，返回 list[float]。

    错误约定：
      - 空文本 → 抛 ValueError；
      - 未配置 API Key → 抛 RuntimeError；
      - 其余 API 异常由调用方捕获后跳过单条任务。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("get_embedding 输入文本为空")

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    # OpenAI SDK 兼容返回结构：resp.data[0].embedding
    return list(resp.data[0].embedding)


# ===== index_text 构建 =====
def _join_list_field(items, sep: str = "、") -> str:
    """把列表字段拼接为可读字符串；非列表返回空。"""
    if not isinstance(items, list):
        return ""
    cleaned = [str(x).strip() for x in items if isinstance(x, (str, int, float)) and str(x).strip()]
    return sep.join(cleaned)


def build_index_text(jd_profile: dict) -> str:
    """从 jd_profile 拼接一段中文可读的检索文本。
    缺失字段直接跳过，避免输出 'None'。
    """
    if not isinstance(jd_profile, dict):
        return ""

    parts = []

    def _add(label: str, value: str):
        if value and str(value).strip():
            parts.append(f"{label}：{value}")

    _add("公司", jd_profile.get("company") or "")
    _add("岗位", jd_profile.get("title") or "")
    _add("岗位类型", jd_profile.get("job_type") or "")
    _add("方向", jd_profile.get("direction") or "")
    _add("业务场景", jd_profile.get("business_area") or "")
    _add("硬技能", _join_list_field(jd_profile.get("hard_skills")))
    _add("工具框架", _join_list_field(jd_profile.get("tools_or_frameworks")))
    _add("领域关键词", _join_list_field(jd_profile.get("domain_keywords")))
    _add("加分项", _join_list_field(jd_profile.get("preferred_skills")))

    # 岗位职责取前 3 条
    resp = jd_profile.get("responsibilities") or []
    if isinstance(resp, list) and resp:
        top3 = [str(x).strip() for x in resp[:3] if isinstance(x, (str, int, float)) and str(x).strip()]
        if top3:
            _add("岗位职责", "；".join(top3))

    _add("学历要求", jd_profile.get("education_requirement") or "")
    _add("经验要求", jd_profile.get("experience_requirement") or "")

    return "\n".join(parts)


# ===== metadata 构建（扁平化 + 列表 -> JSON 字符串）=====
def _meta_str(value) -> Optional[str]:
    """metadata 字符串字段安全转换：非空字符串返回 strip 后值，其余返回 None。"""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _meta_list_json(value) -> str:
    """列表字段统一存为 JSON 字符串，便于回读。"""
    if not isinstance(value, list):
        return "[]"
    cleaned = [v for v in value if v is not None]
    return json.dumps(cleaned, ensure_ascii=False)


def build_metadata(job_id: str, source_file: str, jd_profile: dict) -> dict:
    """构建写入 ChromaDB 的扁平 metadata。
    Chroma 限制值类型只能是 str/int/float/bool/None，因此：
      - location.* 展开为 city / district / office_address；
      - hard_skills / domain_keywords / tools_or_frameworks 转 JSON 字符串；
      - 完整 jd_profile 也以 JSON 字符串形式存在 jd_profile_json。
    """
    if not isinstance(jd_profile, dict):
        jd_profile = {}

    location = jd_profile.get("location") if isinstance(jd_profile.get("location"), dict) else {}

    meta = {
        "job_id": job_id,
        "source_file": source_file,
        "company": _meta_str(jd_profile.get("company")),
        "title": _meta_str(jd_profile.get("title")),
        "job_type": _meta_str(jd_profile.get("job_type")),
        "direction": _meta_str(jd_profile.get("direction")),
        "business_area": _meta_str(jd_profile.get("business_area")),
        "education_requirement": _meta_str(jd_profile.get("education_requirement")),
        "education_level": _meta_str(jd_profile.get("education_level")),
        "experience_requirement": _meta_str(jd_profile.get("experience_requirement")),
        "salary": _meta_str(jd_profile.get("salary")),
        "city": _meta_str(location.get("city")),
        "district": _meta_str(location.get("district")),
        "office_address": _meta_str(location.get("office_address")),
        "hard_skills": _meta_list_json(jd_profile.get("hard_skills")),
        "domain_keywords": _meta_list_json(jd_profile.get("domain_keywords")),
        "tools_or_frameworks": _meta_list_json(jd_profile.get("tools_or_frameworks")),
        "jd_profile_json": json.dumps(jd_profile, ensure_ascii=False),
    }

    # ChromaDB 1.x（Rust 后端）的 metadata 值不接受 None，否则报
    # "Cannot convert Python object to MetadataValue"。
    # 因此剔除所有 None 字段，缺失即不写入该 key。
    return {k: v for k, v in meta.items() if v is not None}


# ===== job_id 查重 =====
def job_exists(collection, job_id: str) -> bool:
    """查询 ChromaDB 中是否已存在指定 job_id。"""
    try:
        result = collection.get(ids=[job_id])
    except Exception as e:
        print(f"[WARN] 查询 job_id 失败（视为不存在）：{e}")
        return False
    ids = result.get("ids") if isinstance(result, dict) else None
    return bool(ids)


# ===== 主流程 =====
def index_jobs(jd_folder: str, db_path: str = DEFAULT_DB_PATH,
               collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
    """遍历 jd_folder 下 .txt 文件，逐条解析、向量化、入库。"""
    # 1) 准备 Chroma 持久化客户端 + collection
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    # 2) 列出所有 .txt 文件（按文件名排序，保证可复现）
    files = sorted(
        f for f in os.listdir(jd_folder)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(jd_folder, f))
    )

    if not files:
        print(f"[WARN] 文件夹中没有 .txt 文件：{jd_folder}")
        print("[DONE] 入库完成：成功 0 条，跳过 0 条，失败 0 条。")
        return

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for fname in files:
        fpath = os.path.join(jd_folder, fname)
        job_id = os.path.splitext(fname)[0]
        print(f"[START] 正在处理：{fname}")

        # 3) 优先用 job_id 查重，避免重复解析浪费 token
        if job_exists(collection, job_id):
            print(f"[SKIP] job_id 已存在：{job_id}")
            skip_count += 1
            continue

        # 4) 读取原文
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                jd_text = f.read()
        except Exception as e:
            print(f"[ERROR] 文件读取失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        if not jd_text or not jd_text.strip():
            print(f"[SKIP] 文件为空：{fname}")
            skip_count += 1
            continue

        # 5) 解析 JD
        try:
            jd_profile = parse_jd(jd_text)
        except Exception as e:
            print(f"[ERROR] 解析失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        if not isinstance(jd_profile, dict) or not jd_profile:
            print(f"[ERROR] 解析失败：{fname}")
            fail_count += 1
            continue

        # 6) 构造 index_text
        index_text = build_index_text(jd_profile)
        if not index_text.strip():
            print(f"[ERROR] index_text 为空：{fname}")
            fail_count += 1
            continue

        # 7) 获取向量
        try:
            vector = get_embedding(index_text)
        except Exception as e:
            print(f"[ERROR] Embedding 失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        # 8) 构造 metadata 并入库
        metadata = build_metadata(job_id, fname, jd_profile)
        try:
            collection.add(
                ids=[job_id],
                embeddings=[vector],
                documents=[index_text],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"[ERROR] 写入 ChromaDB 失败：{fname}，原因：{e}")
            fail_count += 1
            continue

        company = metadata.get("company") or "未知公司"
        title = metadata.get("title") or "未知岗位"
        print(f"[OK] 入库成功：{job_id} - {company} - {title}")
        ok_count += 1

    # 9) 汇总
    print(f"[DONE] 入库完成：成功 {ok_count} 条，跳过 {skip_count} 条，失败 {fail_count} 条。")


# ===== CLI =====
def main():
    """命令行入口：python job_indexer.py <jd_folder>"""
    parser = argparse.ArgumentParser(description="Job Indexer — 批量解析 JD 并入库到本地 ChromaDB")
    parser.add_argument("--jd_folder", default=DEFAULT_JOB_PATH, help="存放多份 JD .txt 文件的目录")
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"ChromaDB 持久化目录（默认 {DEFAULT_DB_PATH}）",
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION_NAME,
        help=f"Collection 名称（默认 {DEFAULT_COLLECTION_NAME}）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.jd_folder):
        print(f"[ERROR] 文件夹不存在：{args.jd_folder}")
        sys.exit(1)

    index_jobs(args.jd_folder, db_path=args.db_path, collection_name=args.collection)

    # ===== 读取测试：回读一条记录，验证入库结果 =====
    print("\n[TEST] 读取测试：从库中回读一条记录")
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection)
    total = collection.count()
    print(f"[TEST] 当前 collection 共 {total} 条记录")
    if total == 0:
        print("[TEST] collection 为空，跳过读取测试")
        return

    jd_test_len = 5
    sample = collection.get(limit=jd_test_len, include=["documents", "metadatas"])
    for i in range(jd_test_len):
        sample_id = sample["ids"][i]
        metadata = sample["metadatas"][i]
        document = sample["documents"][i]
        print(f"[TEST] job_id：{sample_id}")
        print(f"[TEST] 公司/岗位：{metadata.get('company')} - {metadata.get('title')}")
        print(f"[TEST] metadata 字段数：{len(metadata)}")
        print(f"[TEST] index_text 预览：\n{document}")


if __name__ == "__main__":
    main()
