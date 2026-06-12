"""
ingest_jds.py

把 JDs/ 文件夹下所有 .txt JD 批量解析并入库到统一向量库
（data/chroma_db，collection=jobs）。

本脚本只是 job_indexer.index_jobs 的一层便捷封装：
    解析（parse_jd）→ 构造 index_text/metadata → 向量化 → 按 job_id 去重 → 写入 Chroma
全部复用 job_indexer，不重复实现入库逻辑，保证与岗位检索读取的是同一个库。

用法：
    python ingest_jds.py                 # 入库默认 JDs/ 目录
    python ingest_jds.py --jd_folder X   # 指定其它目录
"""

import os
import sys
import argparse

# 脚本位于 scripts/ 下，手动把项目根目录加进 sys.path 以便导入 resume2job 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from resume2job.storage.paths import CHROMA_DIR, COLLECTION_NAME, ensure_data_dir
from resume2job.retrieval.indexer import index_jobs

# 默认 JD 目录：锚定到项目根目录，与当前工作目录无关
DEFAULT_JD_FOLDER = os.path.join(_PROJECT_ROOT, "JDs")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量把 JDs/ 下所有 .txt JD 解析并入库到统一向量库（data/chroma_db）"
    )
    parser.add_argument(
        "--jd_folder", default=DEFAULT_JD_FOLDER,
        help=f"存放多份 JD .txt 文件的目录（默认 {DEFAULT_JD_FOLDER}）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.jd_folder):
        print(f"[ERROR] JD 文件夹不存在：{args.jd_folder}")
        return 1

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("[ERROR] 未配置 DASHSCOPE_API_KEY，无法生成 embedding，入库将全部失败。")
        return 1

    ensure_data_dir()
    print(f"[INFO] JD 来源目录：{args.jd_folder}")
    print(f"[INFO] 目标向量库：{CHROMA_DIR}（collection={COLLECTION_NAME}）")
    print("-" * 60)

    # 入库到统一向量库（按 job_id 去重，已存在的会跳过、不浪费 token）
    index_jobs(args.jd_folder, db_path=CHROMA_DIR, collection_name=COLLECTION_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
