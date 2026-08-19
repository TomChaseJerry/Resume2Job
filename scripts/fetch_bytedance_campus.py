# -*- coding: utf-8 -*-
"""只读验证字节校园正式岗位 Connector；默认不入库、不调用 LLM。"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from resume2job.ingest import ByteDanceCampusConnector

DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "JDs")


def main() -> int:
    parser = argparse.ArgumentParser(description="读取字节校园招聘正式岗位（只读，不入库）")
    parser.add_argument("--keyword", default="", help="岗位关键词；默认不限制")
    parser.add_argument("--page-size", type=int, default=20, help="每页数量，1..100")
    parser.add_argument("--max-pages", type=int, default=1, help="最多请求页数；默认 1 页")
    parser.add_argument("--show", type=int, default=5, help="打印前 N 条岗位摘要")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"JD 保存目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument("--no-save", action="store_true", help="只请求和打印，不保存文件")
    args = parser.parse_args()

    connector = ByteDanceCampusConnector(
        keyword=args.keyword,
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    try:
        payloads = list(connector.fetch())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    saved = []
    if not args.no_save:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for payload in payloads:
            # Connector 生成的 job_id 只包含固定前缀和数字来源 ID，适合作为幂等文件名。
            path = output_dir / f"{payload.job_id}.txt"
            path.write_text(payload.raw_jd.rstrip() + "\n", encoding="utf-8")
            saved.append(path)

    print(f"[OK] 请求成功：取得 {len(payloads)} 条校园正式岗位（本次最多 {args.max_pages} 页）")
    if saved:
        print(f"[OK] 已保存 {len(saved)} 个 JD 文件到：{saved[0].parent}")
    elif args.no_save:
        print("[INFO] --no-save 已启用，本次未写文件")
    for index, payload in enumerate(payloads[:max(0, args.show)], start=1):
        cities = "、".join(payload.extra.get("cities") or []) or "城市未注明"
        print(f"{index}. [{payload.source_job_id}] {payload.title} | {cities}")
        print(f"   {payload.canonical_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
