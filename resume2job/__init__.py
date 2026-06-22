"""Resume2Job —— 基于 LangGraph 的 Agentic RAG 实习岗位智能推荐系统。

包结构：
    core/        统一配置（config）与 LLM / Embedding 客户端、JSON 工具（llm）
    parsing/     简历解析（resume_parser）、JD 解析（jd_parser）
    retrieval/   岗位知识库构建（indexer）与混合检索（vector + BM25 + RRF + rerank）
    scoring/     匹配评分（match_scorer：技能 status/skill_score、两层评分、加分）
    generation/  推荐报告 + 技能差距视图（recommendation）、学习计划（learning_plan）、面试题（interview）
    agent/       LangGraph 工作流：State、Function Calling planner、执行节点、Tool Calling 增强节点
    storage/     SQLite / Chroma 存储路径、用户画像缓存、对话记录、JD 自动入库
    eval/        自动化评测：评测集构造、检索指标（Recall@K / MRR / nDCG）、LLM-as-judge
"""
