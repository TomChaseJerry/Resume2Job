# -*- coding: utf-8 -*-
"""检索评测集构造（synthetic eval set）。

业界常用做法：从知识库文档**反向生成**检索 query（query 与生成它的文档天然相关），
快速得到 query → relevant_doc 标注，规模可随 JD 库扩展自动重建。

流程：
    1. 遍历 Chroma 岗位库的全部 index_text；
    2. 每条 JD 用 LLM 生成 3 条不同风格的求职检索 query：
         - keyword：技能关键词串（贴近检索 Query 改写的输出形态）；
         - natural：自然语言一句话求职意向；
         - colloquial：口语化表达（含模糊措辞，考验语义召回）；
    3. 写入 eval/data/retrieval_dataset.jsonl，每行：
         {"query_id", "job_id", "style", "query", "relevant_job_ids": [job_id]}

人工抽查：直接编辑该 jsonl —— 改写 query、或在 relevant_job_ids 里增删岗位
（同方向的多条 JD 可互为相关）。retrieval_eval 按文件现状评测。
"""

import os
import json
import argparse

import chromadb

from resume2job.core.config import CHAT_MODEL
from resume2job.core.llm import call_llm, safe_json_parse
from resume2job.storage.paths import CHROMA_DIR, COLLECTION_NAME

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_PATH = os.path.join(DATA_DIR, "retrieval_dataset.jsonl")

_STYLES = ("keyword", "natural", "colloquial")

_SYSTEM_PROMPT = (
    "你是一个检索评测数据构造器。给定一条实习岗位 JD 的结构化摘要，"
    "生成 3 条「正在找实习的学生可能输入的检索请求」，要求该 JD 是这些请求的理想答案。\n"
    "三条风格分别为：\n"
    "1. keyword：6~10 个空格分隔的技能/方向关键词；\n"
    "2. natural：一句完整的自然语言求职意向；\n"
    "3. colloquial：口语化、可含模糊措辞（如「会点」「想搞」）。\n"
    "禁止直接抄写公司名或岗位名（检索者通常不知道具体公司）。\n"
    "只输出 JSON：{\"keyword\": \"...\", \"natural\": \"...\", \"colloquial\": \"...\"}"
)


def _generate_queries_for_job(job_id: str, document: str) -> dict:
    """对单条 JD 生成 3 风格 query。失败返回空 dict（该 JD 跳过）。"""
    try:
        raw = call_llm(
            _SYSTEM_PROMPT,
            f"## JD 摘要\n{document[:1500]}",
            model=CHAT_MODEL,
            temperature=0.5,  # 评测 query 需要多样性
        )
        parsed = safe_json_parse(raw) or {}
    except Exception as e:
        print(f"[WARN] {job_id} query 生成失败：{e}")
        return {}
    return {s: str(parsed.get(s) or "").strip() for s in _STYLES if str(parsed.get(s) or "").strip()}


def build_dataset(output_path: str = DATASET_PATH) -> int:
    """遍历岗位库构造评测集，返回写入的样本条数。"""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    res = collection.get(include=["documents"])
    ids = res.get("ids") or []
    documents = res.get("documents") or []
    if not ids:
        print("[ERROR] 岗位库为空，请先运行 scripts/ingest_jds.py 建库。")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for job_id, document in zip(ids, documents):
            print(f"[build_dataset] 生成 query：{job_id}")
            queries = _generate_queries_for_job(job_id, document or "")
            for style, query in queries.items():
                sample = {
                    "query_id": f"{job_id}__{style}",
                    "job_id": job_id,
                    "style": style,
                    "query": query,
                    "relevant_job_ids": [job_id],
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1

    print(f"[build_dataset] 完成：{count} 条评测样本 → {output_path}")
    print("[build_dataset] 建议人工抽查该文件：改写不自然的 query，"
          "并为同方向岗位互补 relevant_job_ids。")
    return count


def load_dataset(path: str = DATASET_PATH) -> list:
    """读取评测集 jsonl，返回样本 dict 列表。"""
    if not os.path.isfile(path):
        return []
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sample.get("query") and sample.get("relevant_job_ids"):
                samples.append(sample)
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构造检索评测集（LLM 反向生成 query）")
    parser.add_argument("--output", default=DATASET_PATH, help="输出 jsonl 路径")
    args = parser.parse_args()
    build_dataset(args.output)
