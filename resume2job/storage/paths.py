"""
storage/paths.py

全项目统一的数据存储路径（单一事实来源）。

所有持久化都收敛到项目根目录下的 data/：
    - data/resume2job.db   SQLite，存储对话记录 / 用户画像 / JD 元数据；
    - data/chroma_db       Chroma 向量库（collection=jobs），岗位检索与 JD 入库共用同一个库。

路径锚定到项目根目录（resume2job/storage/ 的上两级），与当前工作目录无关。
"""

import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(_BASE_DIR, "data")
SQLITE_PATH = os.path.join(DATA_DIR, "resume2job.db")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "jobs"


def ensure_data_dir() -> None:
    """确保 data/ 目录存在。"""
    os.makedirs(DATA_DIR, exist_ok=True)
