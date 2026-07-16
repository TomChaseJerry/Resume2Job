# -*- coding: utf-8 -*-
"""
scripts/verify_ingest.py —— Stage 1（resume2job/ingest）接入与生命周期模块的可复盘验收脚本。

设计成「看一眼就懂 Stage 1 在干什么」的体检报告：
    - 默认运行：**全程不调用任何 API、不修改真实库**（生命周期闸门那步用「过期->重新激活」的可逆操作，
      跑完恢复原状）。读真实 data/resume2job.db 做版本/质量/一致性体检 + 一组离线单测。
    - 加 --with-api：额外在**隔离临时库**（临时 SQLite + 临时 Chroma，跑完删除）里跑一次真实端到端接入
      （parse_jd + embedding），演示 created -> unchanged -> updated 的生命周期状态机。真实库不受影响。

用法：
    python scripts/verify_ingest.py                # 离线体检（推荐先跑这个）
    python scripts/verify_ingest.py --with-api     # 额外跑隔离端到端（需联网 + DASHSCOPE_API_KEY）
"""

import os
import sys
import argparse
import collections

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# Windows 控制台默认 GBK，无法编码部分符号；把 stdout 切到 UTF-8（旧控制台可能显示乱码但不会报错；
# Windows Terminal / chcp 65001 下正常）。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

JDS_DIR = os.path.join(_PROJECT_ROOT, "JDs")


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def ok(msg: str) -> None:
    print("  [OK] " + msg)


def info(msg: str) -> None:
    print("  - " + msg)


# ---------------------------------------------------------------------------
def section_imports():
    hr("1. 模块导入 & 计算图自检（无 API）—— 证明无语法错 / 无导入环")
    from resume2job.agent.graph import build_graph
    build_graph()
    from resume2job.ingest import (  # noqa: F401
        RawJobPayload, JobRecord, IngestResult, validate_job, normalize_raw_payload,
        LocalFileConnector, CSVConnector, UserPasteConnector, OfficialCareerConnector,
    )
    from resume2job.ingest.lifecycle import (  # noqa: F401
        ingest_record, ingest_all, mark_expired, reactivate, sweep_stale, backfill_lifecycle_fields,
    )
    ok("ingest 包 + connectors + lifecycle 全部可导入；LangGraph 可编译（indexer/jd_ingest <-> lifecycle 无导入环）")


def section_schema():
    hr("2. jobs 表结构自检 —— 新增 10 个生命周期 / 版本 / 质量列是否已迁移到位")
    import sqlite3
    from resume2job.storage import jobs_store
    c = sqlite3.connect(jobs_store.DB_PATH)
    cols = [r[1] for r in c.execute("PRAGMA table_info(jobs)")]
    c.close()
    new_cols = ["status", "source_job_id", "canonical_url", "content_hash", "collected_at",
                "last_verified_at", "parser_version", "embedding_version", "index_text_version", "quality_score"]
    missing = [x for x in new_cols if x not in cols]
    info(f"jobs 表共 {len(cols)} 列；新增列：{new_cols}")
    if missing:
        print(f"  [FAIL] 缺列：{missing}")
    else:
        ok("10 个新列全部存在（init_db 幂等 ALTER 已生效）")


def section_db_health():
    hr("3. 既有库体检 —— 读真实 data/resume2job.db 的版本 / 质量 / 状态分布 + 一致性")
    from resume2job.storage import jobs_store
    from resume2job.storage.jobs_store import compute_content_hash
    rows = jobs_store.all_rows()
    info(f"岗位总数：{len(rows)}")
    if not rows:
        print("  （库为空，先跑 python scripts/ingest_jds.py 建库）")
        return
    pv = collections.Counter(r.get("parser_version") for r in rows)
    ev = collections.Counter(r.get("embedding_version") for r in rows)
    st = collections.Counter(r.get("status") for r in rows)
    src = collections.Counter(r.get("source") for r in rows)
    qs = [r.get("quality_score") for r in rows if r.get("quality_score") is not None]
    info(f"parser_version    分布：{dict(pv)}")
    info(f"embedding_version 分布：{dict(ev)}")
    info(f"status            分布：{dict(st)}")
    info(f"source            分布：{dict(src)}")
    if qs:
        info(f"quality_score     min/avg/max：{min(qs)} / {round(sum(qs)/len(qs),3)} / {max(qs)}")
    cov_ch = sum(1 for r in rows if r.get("content_hash"))
    cov_lv = sum(1 for r in rows if r.get("last_verified_at"))
    info(f"content_hash 覆盖：{cov_ch}/{len(rows)}；last_verified_at 覆盖：{cov_lv}/{len(rows)}")
    # 一致性不变量：content_hash 必须等于 index_text 的哈希（无绕过写路径的漂移）
    drift = [r["job_id"] for r in rows
             if r.get("index_text") and r.get("content_hash") != compute_content_hash(r["index_text"])]
    if drift:
        print(f"  [FAIL] content_hash 漂移 {len(drift)} 条：{drift[:5]}")
    else:
        ok("content_hash 与 index_text 全库一致（0 漂移）—— 索引一致性不变量成立")


def section_connectors():
    hr("4. Connectors 自检（无 API）—— 各来源都能产出统一的 RawJobPayload")
    from resume2job.ingest import LocalFileConnector, CSVConnector, UserPasteConnector, OfficialCareerConnector
    # LocalFile：真实 JDs/ 目录
    payloads = list(LocalFileConnector(JDS_DIR).fetch()) if os.path.isdir(JDS_DIR) else []
    info(f"LocalFileConnector(JDs/)：产出 {len(payloads)} 条；样例 job_id="
         f"{payloads[0].job_id if payloads else 'N/A'}，正文长度="
         f"{len(payloads[0].raw_jd) if payloads else 0}")
    # CSV：临时写一行演示
    import tempfile
    fd, csv_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("company,title,jd_text\n")
        f.write("测试公司,测试算法岗,负责模型训练，要求熟悉 Python 与 PyTorch。\n")
    csv_payloads = list(CSVConnector(csv_path).fetch())
    os.remove(csv_path)
    info(f"CSVConnector：产出 {len(csv_payloads)} 条；样例 company="
         f"{csv_payloads[0].company if csv_payloads else 'N/A'}")
    # UserPaste
    up = list(UserPasteConnector("这是一段用户粘贴的 JD 原文。").fetch())
    info(f"UserPasteConnector：产出 {len(up)} 条（job_id={up[0].job_id}，无 id -> 全量去重模式）")
    # Official 占位
    try:
        list(OfficialCareerConnector("https://example.com/jobs").fetch())
        print("  [FAIL] OfficialCareerConnector 应抛 NotImplementedError")
    except NotImplementedError:
        ok("OfficialCareerConnector 正确占位（fetch 抛 NotImplementedError，Phase 2 再实现）")


def section_validator():
    hr("5. 质量校验 validator（无 API）—— is_valid 闸门 + quality_score + 具名 warnings")
    from resume2job.ingest import validate_job
    cases = {
        "完整岗位": {"company": "字节跳动", "title": "大模型算法实习生", "job_type": "实习",
                   "hard_skills": ["深度学习", "PyTorch"], "responsibilities": ["训练模型"],
                   "education_level": "硕士", "location": {"cities": ["北京"], "city": "北京"}},
        "多城市无公司(阿里统招式)": {"company": None, "title": "算法工程师", "job_type": "校招",
                   "hard_skills": ["机器学习"], "responsibilities": ["建模"],
                   "education_level": "本科", "location": {"cities": ["北京", "杭州"], "city": "北京"}},
        "残缺岗位(应拦截)": {"company": None, "title": None, "hard_skills": [], "responsibilities": [],
                   "location": {"cities": [], "city": None}},
    }
    long_text = "岗位职责：负责大模型应用研发，包括 RAG 与 Agent。任职要求：熟悉深度学习与 PyTorch。" * 2
    for name, prof in cases.items():
        text = "太短" if "残缺" in name else long_text
        q = validate_job(prof, text)
        info(f"{name}: is_valid={q.is_valid}  quality={q.quality_score}  warnings={q.warnings}")


def section_record_and_normalizer():
    hr("6. JobRecord 往返 + normalizer（无 API）—— 类型化记录装配 + 信封清洗")
    from resume2job.ingest import JobRecord, RawJobPayload, normalize_raw_payload
    from resume2job.retrieval.indexer import build_index_text
    prof = {"company": "字节跳动", "title": "大模型算法实习生", "job_type": "实习",
            "hard_skills": ["深度学习", "PyTorch"], "education_level": "硕士",
            "responsibilities": ["训练模型"], "location": {"cities": ["北京"], "city": "北京"}}
    it = build_index_text(prof)
    rec = JobRecord.from_jd_profile("demo_job", "原文示例", prof, index_text=it, source="user_uploaded", quality_score=1.0)
    d = rec.to_store_dict()
    info(f"JobRecord -> store_dict：版本戳 parser={d['parser_version']} / index_text={d['index_text_version']} / "
         f"embedding={d['embedding_version']}；派生 cities={d['cities_json']} job_types={d['job_types_json']} "
         f"min_degree_rank={d['min_degree_rank']}；content_hash 已生成={bool(d['content_hash'])}")
    rec2 = JobRecord.from_store_row({**d, "jd_profile": prof})
    info(f"store_dict -> JobRecord 往返：job_id={rec2.job_id} status={rec2.status} cities={rec2.cities}（无损）")
    # normalizer 信封清洗（CRLF/BOM/空白）
    p = RawJobPayload(source="x", raw_jd="﻿  正文第一行\r\n\r\n正文第二行  ", company="  字节  ")
    np_ = normalize_raw_payload(p)
    info(f"normalizer：清洗后 raw_jd={np_.raw_jd!r}  company={np_.company!r}（去 BOM/CRLF->LF/strip，保守不动内容指纹）")


def section_lifecycle_gate():
    hr("7. 生命周期闸门（真实库，可逆）—— 过期岗位不再召回 -> 重新激活恢复")
    from resume2job.storage import jobs_store
    from resume2job.ingest.lifecycle import mark_expired, reactivate
    elig = jobs_store.get_eligible_jobs(None, None, "实习")
    if not elig:
        info("当前无「实习」可召回岗位，跳过本节")
        return
    target = next(iter(elig))
    orig = jobs_store.get_job(target)
    n0 = len(elig)
    mark_expired(target)
    n1 = len(jobs_store.get_eligible_jobs(None, None, "实习"))
    reactivate(target)
    n2 = len(jobs_store.get_eligible_jobs(None, None, "实习"))
    info(f"以 {target} 为例：可召回 {n0} -> 标记过期后 {n1}（剔除）-> 重新激活后 {n2}（恢复）")
    if n1 == n0 - 1 and n2 == n0 and jobs_store.get_job(target)["status"] == (orig or {}).get("status", "active"):
        ok("过期岗位被 eligibility 闸门正确排除，且操作可逆（已恢复原状）")
    else:
        print("  [WARN] 闸门行为与预期不符，请检查")


def section_e2e_isolated():
    hr("8. 隔离临时库端到端接入（--with-api：parse_jd + embedding）—— created -> unchanged -> updated")
    import tempfile, shutil
    import chromadb
    from resume2job.storage import jobs_store
    from resume2job.ingest.models import RawJobPayload
    from resume2job.ingest.lifecycle import ingest_record

    # 找一条真实 JD 原文作样本
    sample = None
    if os.path.isdir(JDS_DIR):
        for f in sorted(os.listdir(JDS_DIR)):
            if f.lower().endswith(".txt"):
                sample = open(os.path.join(JDS_DIR, f), encoding="utf-8").read()
                break
    if not sample:
        info("JDs/ 下无 .txt 样本，跳过")
        return

    tmp = tempfile.mkdtemp(prefix="ingest_verify_")
    orig_db = jobs_store.DB_PATH
    try:
        # 切到隔离临时库（真实库完全不受影响）
        jobs_store.DB_PATH = os.path.join(tmp, "test.db")
        jobs_store.init_db()
        col = chromadb.PersistentClient(path=os.path.join(tmp, "chroma")).get_or_create_collection("jobs")

        r1 = ingest_record(RawJobPayload(source="verify", raw_jd=sample, job_id="demo1"), collection=col)
        info(f"首次接入同一 JD（身份模式 job_id=demo1）-> action={r1.action}  quality={r1.quality_score}")
        r2 = ingest_record(RawJobPayload(source="verify", raw_jd=sample, job_id="demo1"), collection=col)
        info(f"再次接入完全相同内容        -> action={r2.action}（哈希未变，0 token 幂等）")
        r3 = ingest_record(RawJobPayload(source="verify", raw_jd=sample + "\n\n附加要求：熟悉 vLLM 推理加速。",
                                         job_id="demo1"), collection=col)
        info(f"接入内容已变更的同一岗位      -> action={r3.action}（重解析 + 删旧向量重嵌入）")
        row = jobs_store.get_job("demo1")
        info(f"临时库最终记录：status={row['status']} parser_ver={row['parser_version']} "
             f"emb_ver={row['embedding_version']} content_hash={row['content_hash'][:12]}...")
        if (r1.action, r2.action, r3.action) == ("created", "unchanged", "updated"):
            ok("生命周期状态机端到端正确：created -> unchanged -> updated")
        else:
            print(f"  [WARN] 动作序列={r1.action, r2.action, r3.action}（预期 created/unchanged/updated）")
    finally:
        jobs_store.DB_PATH = orig_db          # 恢复真实库指向
        shutil.rmtree(tmp, ignore_errors=True)
        info("隔离临时库已删除，真实库未受任何影响")


def main():
    ap = argparse.ArgumentParser(description="Stage 1 ingest 模块验收（默认离线、不改真实库）")
    ap.add_argument("--with-api", action="store_true",
                    help="额外在隔离临时库跑真实端到端接入（需联网 + DASHSCOPE_API_KEY）")
    args = ap.parse_args()

    section_imports()
    section_schema()
    section_db_health()
    section_connectors()
    section_validator()
    section_record_and_normalizer()
    section_lifecycle_gate()
    if args.with_api:
        section_e2e_isolated()
    else:
        hr("8. 端到端接入（已跳过）")
        info("加 --with-api 可在隔离临时库跑 created->unchanged->updated（需联网，真实库不受影响）")

    hr("验收完成")
    print("  持久化结果在哪看：")
    print("    - data/resume2job.db 的 jobs 表（新增 10 列）；或 `python scripts/view_jds.py` 浏览岗位")
    print("    - 真实建库 / 增量入库：`python scripts/ingest_jds.py`")


if __name__ == "__main__":
    main()
