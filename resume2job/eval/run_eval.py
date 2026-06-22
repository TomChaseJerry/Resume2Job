# -*- coding: utf-8 -*-
"""一键评测入口。

用法：
    python -m resume2job.eval.run_eval --build              # 构造/重建检索评测集
    python -m resume2job.eval.run_eval --retrieval          # 检索指标四配置对比
    python -m resume2job.eval.run_eval --judge resume.json jd.txt
                                                            # 端到端生成 + LLM-as-judge
    python -m resume2job.eval.run_eval --build --retrieval  # 组合执行

评测体系两层：
    检索层  Recall@K / MRR / nDCG —— 量化混合检索、rerank 相对纯向量的提升；
    生成层  LLM-as-judge（忠实性 / 有用性 / 证据性）—— 量化推荐报告质量。
"""

import sys
import json
import argparse


def _judge_end_to_end(resume_json: str, jd_file: str) -> None:
    """对一条「简历 × JD」跑完整评分 + 报告生成，再交给 judge 打分。"""
    from resume2job.parsing.jd_parser import parse_jd
    from resume2job.scoring.match_scorer import score_match
    from resume2job.generation.recommendation import generate_report_and_gap
    from resume2job.eval.judge import judge_report

    with open(resume_json, "r", encoding="utf-8") as f:
        resume_profile = json.load(f)
    with open(jd_file, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print("[run_eval] 解析 JD ...")
    jd_profile = parse_jd(jd_text)
    if not jd_profile or jd_profile.get("error"):
        print(f"[run_eval] JD 解析失败：{jd_profile.get('error') if isinstance(jd_profile, dict) else '结果为空'}")
        return
    print("[run_eval] 匹配评分 ...")
    match_score = score_match(resume_profile, jd_profile)
    print("[run_eval] 技能差距 + 推荐报告（合并一次 LLM 调用）...")
    report, skill_gap, _writer = generate_report_and_gap(resume_profile, jd_profile, match_score)

    print("\n----- 推荐报告 -----\n" + report)
    print("\n[run_eval] LLM-as-judge 评审 ...")
    verdict = judge_report(resume_profile, jd_profile, report)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Resume2Job 一键评测")
    parser.add_argument("--build", action="store_true", help="构造/重建检索评测集")
    parser.add_argument("--retrieval", action="store_true", help="跑检索指标四配置对比")
    parser.add_argument("--judge", nargs=2, metavar=("RESUME_JSON", "JD_FILE"),
                        help="端到端生成推荐报告并 LLM-as-judge 打分")
    parser.add_argument("--top_k", type=int, default=5, help="检索评测 top_k")
    args = parser.parse_args(argv)

    if not (args.build or args.retrieval or args.judge):
        parser.print_help()
        return 1

    if args.build:
        from resume2job.eval.build_dataset import build_dataset
        build_dataset()

    if args.retrieval:
        from resume2job.eval.retrieval_eval import run_retrieval_eval
        run_retrieval_eval(top_k=args.top_k)

    if args.judge:
        _judge_end_to_end(args.judge[0], args.judge[1])

    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出
    except Exception:
        pass
    sys.exit(main())
