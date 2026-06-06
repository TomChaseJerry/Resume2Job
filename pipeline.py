# pipeline_v2.py
from resume_parser import parse_resume
from job_retriever import retrieve_jobs
from jd_parser import parse_jd
from match_scorer import score_match
from recommendation_writer import generate_recommendation

# 1. 解析简历
resume = parse_resume("Resumes/resume_test_3.pdf")

# 2. 从岗位库检索候选岗位（取代手动输入JD）
candidates = retrieve_jobs(resume, top_k=5, city_filter="北京")

# 3. 对每个候选岗位评分 + 生成报告
for job in candidates:
    # metadata 里有 jd 原文路径，加载并解析（JD 文件为 UTF-8，需显式指定编码，
    # 否则在中文 Windows 上会按 GBK 解码而报 UnicodeDecodeError）
    with open(f"JDs/{job['job_id']}.txt", "r", encoding="utf-8") as f:
        jd = parse_jd(f.read())
    match = score_match(resume, jd)
    report = generate_recommendation(jd, match)
    print(report)
    print("="*50)