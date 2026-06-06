# Resume2Job：面向实习求职场景的 Agentic RAG 岗位推荐系统

## 1. 项目目的

本项目面向实习求职场景，构建一个能够理解用户简历、分析岗位 JD、检索候选岗位、评估匹配程度并生成可解释推荐结果的智能岗位推荐系统。

系统输入主要包括：

```text
用户问题
用户上传的简历 PDF
岗位 JD 或岗位知识库
```

系统输出主要包括：

```text
推荐岗位列表
推荐理由
匹配评分
技能差距分析
风险提示
投递建议
```

项目核心能力包括：

```text
简历结构化理解
岗位 JD 结构化理解
岗位知识库检索
Agent 任务规划与工具路由
岗位匹配评分
Skill Gap Analysis
可解释推荐生成
```

---

## 2. 系统总体流程

推荐系统整体流程如下：

```text
用户输入问题 / 上传简历
        ↓
Agent 判断任务类型
        ↓
生成任务计划与工具调用路径
        ↓
简历解析与用户画像构建
        ↓
岗位来源判断：岗位库检索 / 用户提供 JD / 本地导入 JD
        ↓
岗位 JD 结构化分析
        ↓
候选岗位匹配评分
        ↓
Skill Gap Analysis
        ↓
推荐结果生成
```

该流程中，RAG 主要负责岗位候选召回，Agent 负责根据用户问题和中间结果决定是否调用检索、JD 分析、匹配评分、技能差距分析等工具。

---

## 3. Agentic RAG 中 Agent 的体现

本项目中的 Agent 不是简单按固定顺序调用所有模块，而是通过 Planner、Router 和 Checker 实现任务规划、路径选择和结果自检。

### 3.1 Agent 自主性的核心体现

Agent 的自主性主要体现在以下几个方面：

```text
1. 判断用户意图
2. 判断是否需要解析简历
3. 判断是否需要检索岗位库
4. 根据简历画像生成检索 Query
5. 判断检索结果是否足够
6. 判断是否需要读取岗位详情
7. 判断是否需要进行 Skill Gap Analysis
8. 判断是否需要调用通勤工具
9. 根据评分结果决定是否二次分析
10. 生成最终推荐路径
```

也就是说，系统不是每次都固定执行完整链路，而是根据用户问题动态选择执行路径。

---

### 3.2 典型任务路径示例

#### 场景 A：用户希望推荐实习岗位

用户问题：

```text
帮我推荐适合我的北京 AI 算法实习岗位
```

Agent 规划路径：

```text
parse_resume
→ build_user_profile
→ generate_search_queries
→ search_jobs
→ analyze_job_detail
→ score_job_match
→ skill_gap_analysis
→ generate_recommendation
```

---

#### 场景 B：用户粘贴一个 JD，询问是否适合投递

用户问题：

```text
这个岗位我适合投吗？
```

Agent 规划路径：

```text
parse_resume
→ extract_job_requirements
→ score_job_match
→ skill_gap_analysis
→ generate_application_advice
```

该场景下，用户已经提供具体 JD，因此不需要检索岗位库。

---

#### 场景 C：用户只想知道自己距离 Agent 岗位还缺什么

用户问题：

```text
我投 Agent 岗还缺哪些能力？
```

Agent 规划路径：

```text
parse_resume
→ load_target_job_requirement_template
→ skill_gap_analysis
→ generate_improvement_advice
```

该场景下，不需要岗位库检索，也不需要通勤计算。

---

#### 场景 D：用户要求筛选通勤时间

用户问题：

```text
帮我找通勤一小时以内的大模型实习岗位
```

Agent 规划路径：

```text
parse_resume
→ search_jobs
→ extract_office_address
→ calculate_commute
→ filter_jobs
→ score_job_match
→ generate_recommendation
```

该场景下，通勤约束是硬条件，因此需要调用地图或通勤计算工具。

---

## 4. LangGraph 工作流设计

本项目适合使用 LangGraph 构建多节点 Agent 工作流。推荐采用“上层 Agent 决策 + 下层确定性工具”的结构。

```text
START
  ↓
intent_router
  ↓
planner_node
  ↓
resume_router
  ↓
resume_parser_node
  ↓
job_source_router
  ↓
retrieval_node / jd_input_node
  ↓
job_analyzer_node
  ↓
match_scorer_node
  ↓
gap_router
  ↓
skill_gap_node
  ↓
recommendation_node
  ↓
END
```

其中：

```text
intent_router：判断用户任务类型
planner_node：生成任务计划
resume_router：判断是否需要解析或复用简历画像
job_source_router：判断岗位来源
retrieval_node：从岗位知识库中召回候选岗位
jd_input_node：处理用户直接粘贴的 JD
job_analyzer_node：分析岗位核心要求
match_scorer_node：计算匹配分
gap_router：判断是否需要技能差距分析
skill_gap_node：输出技能差距与补强建议
recommendation_node：生成最终推荐报告
```

---

## 5. LangGraph State 设计

建议在 LangGraph 中维护统一 State，用于在各节点之间传递结构化信息。

示例 State：

```json
{
  "user_query": "",
  "task_type": "",
  "plan": {
    "need_resume_parse": true,
    "need_job_search": true,
    "need_jd_analysis": true,
    "need_skill_gap": true,
    "need_commute": false
  },
  "resume_profile": {},
  "search_queries": [],
  "candidate_jobs": [],
  "job_analysis": [],
  "match_scores": [],
  "skill_gaps": [],
  "recommendations": [],
  "errors": []
}
```

State 设计的重点是让每个节点只处理自己负责的字段，并将结果写回 State。这样便于调试、扩展和替换模块。

---

## 6. 模块化设计

### 6.1 Module A：Resume Understanding 简历理解模块

#### 功能目标

读取用户上传的 PDF 简历，并转换为结构化用户画像。

#### 输入

```text
PDF 简历
用户补充问题
```

#### 输出

```json
{
  "education": {
    "school": "",
    "degree": "",
    "major": ""
  },
  "skills": [],
  "projects": [
    {
      "name": "",
      "description": "",
      "keywords": [],
      "evidence": []
    }
  ],
  "job_intention": [],
  "location": "",
  "availability": ""
}
```

#### 核心任务

```text
PDF 文本解析
简历字段抽取
教育背景识别
技能标签抽取
项目经历摘要
求职意向识别
结构化 JSON 输出
```

#### 实现建议

```text
PDF 解析：PyMuPDF / pdfplumber
信息抽取：LLM Structured Output
输出约束：JSON Schema
异常处理：字段缺失时返回 null 或 empty list
```

---

### 6.2 Module B：Job Understanding 岗位理解模块

#### 功能目标

将非结构化岗位 JD 转换为结构化岗位需求。

#### 输入

```text
岗位 JD 原文
```

#### 输出

```json
{
  "company": "",
  "title": "",
  "direction": "",
  "responsibilities": [],
  "required_skills": [],
  "preferred_skills": [],
  "education_requirement": "",
  "internship_duration": "",
  "office_address": "",
  "risk_points": []
}
```

#### 核心任务

```text
岗位职责抽取
任职要求抽取
技能关键词抽取
岗位方向分类
学历要求识别
实习时长识别
办公地址识别
风险点识别
```

#### 设计重点

岗位 JD 需要结构化后再进入评分模块。  
如果只对 JD 原文做向量相似度匹配，后续推荐理由和技能差距分析会缺少明确证据。

---

### 6.3 Module C：Job Knowledge Base 岗位知识库与 RAG 检索模块

#### 功能目标

将岗位信息构建为可检索的知识库，并根据用户画像召回候选岗位。

#### 输入

```text
结构化岗位信息
用户画像
岗位方向偏好
城市偏好
```

#### 输出

```json
[
  {
    "job_id": "",
    "company": "",
    "title": "",
    "retrieval_score": 0.0,
    "matched_terms": []
  }
]
```

#### 核心任务

```text
岗位信息清洗
岗位文本切分
Embedding 向量化
向量数据库存储
岗位候选召回
城市 / 学历 / 方向 / 技能过滤
Top-K 候选岗位返回
```

#### RAG 检索策略

不建议直接使用用户原始问题作为唯一 Query。  
更稳妥的方式是由 Agent 根据用户画像生成多个检索 Query。

示例：

```json
{
  "query_1": "北京 大模型应用算法实习 RAG Agent Python PyTorch",
  "query_2": "北京 多模态算法实习 PyTorch 深度学习",
  "query_3": "北京 机器学习算法实习 GNN 模型训练"
}
```

---

### 6.4 Module D：Planner & Router Agent 规划与路由模块

#### 功能目标

根据用户问题和当前 State，决定系统下一步应执行哪些模块。

#### 输入

```text
用户问题
当前 State
是否已有简历画像
是否已有 JD
是否需要通勤约束
```

#### 输出

```json
{
  "task_type": "job_recommendation",
  "required_steps": [
    "parse_resume",
    "retrieve_jobs",
    "analyze_jobs",
    "score_match",
    "skill_gap_analysis",
    "generate_recommendation"
  ],
  "need_resume_parse": true,
  "need_job_search": true,
  "need_commute": false,
  "need_skill_gap": true
}
```

#### 核心任务

```text
用户意图识别
任务拆解
工具选择
模块路由
条件分支控制
异常情况下的重试或跳过
```

#### 常见路由规则

```text
如果用户上传新简历 → 调用 Resume Parser
如果已有用户画像且未要求更新 → 复用已有画像
如果用户粘贴 JD → 跳过岗位库检索
如果用户要求推荐岗位 → 调用岗位检索
如果用户强调通勤 → 调用通勤工具
如果岗位高分但存在风险 → 调用 Skill Gap Analysis
如果岗位明显不匹配 → 可直接归为不推荐
```

---

### 6.5 Module E：Job Analyzer Agent 岗位分析模块

#### 功能目标

分析候选岗位真正重视的能力要求，为后续评分提供依据。

#### 输入

```text
结构化 JD
岗位原文
```

#### 输出

```json
{
  "core_requirements": {
    "Agent": 0.45,
    "RAG": 0.30,
    "Python": 0.15,
    "工程部署": 0.10
  },
  "job_difficulty": "medium",
  "evidence": [
    "JD 中多次提到 Agent、工具调用和 RAG 系统开发"
  ],
  "risk_points": [
    "需要 Agent 工程经验",
    "需要熟悉 LangChain 或 LangGraph"
  ]
}
```

#### 核心任务

```text
岗位核心能力识别
技能重要性估计
岗位难度判断
岗位风险点总结
岗位证据提取
```

#### 设计重点

很多 JD 会罗列多个技能，但不同技能重要性不同。  
该模块需要判断哪些要求是核心要求，哪些只是加分项。

---

### 6.6 Module F：Match Scorer 岗位匹配评分模块

#### 功能目标

从多个维度评估用户与岗位之间的匹配程度。

#### 输入

```text
用户画像
岗位分析结果
岗位结构化信息
```

#### 输出

```json
{
  "skill_score": {
    "score": 82,
    "evidence": []
  },
  "project_score": {
    "score": 78,
    "evidence": []
  },
  "education_score": {
    "score": 95,
    "evidence": []
  },
  "direction_score": {
    "score": 90,
    "evidence": []
  },
  "final_score": 84,
  "match_level": "recommended"
}
```

#### 推荐评分维度

| 维度 | 说明 |
|---|---|
| 技能匹配度 | 简历技能与 JD 要求是否一致 |
| 项目相关性 | 项目经历是否能支撑岗位要求 |
| 学历适配度 | 是否满足岗位学历要求 |
| 专业适配度 | 专业背景是否相关 |
| 岗位方向匹配度 | 是否符合用户求职意向 |
| 综合推荐分 | 多维度加权后的最终结果 |

#### 评分建议

评分模块不建议完全依赖 LLM 自由打分。  
更稳妥的方式是：

```text
规则分数
+
LLM 证据解释
+
Rubric 约束
```

示例：

```text
技能命中率提供基础分
项目相关性由 LLM 基于证据判断
最终分数通过规则加权生成
```

---

### 6.7 Module G：Skill Gap Analysis 技能差距分析模块

#### 功能目标

分析用户与目标岗位之间的能力差距，并生成补强建议。

#### 输入

```text
用户画像
岗位核心要求
匹配评分结果
```

#### 输出

```json
{
  "matched_skills": [],
  "weak_skills": [],
  "missing_skills": [],
  "risk": "",
  "suggestion": ""
}
```

#### 推荐流程

```text
1. 抽取岗位核心要求
2. 抽取简历中的支撑证据
3. 对每个要求判断：强匹配 / 弱匹配 / 缺失
4. 输出技能差距
5. 输出岗位风险
6. 输出补强建议
```

#### 示例输出

```json
{
  "matched_skills": ["Python", "PyTorch", "多模态建模"],
  "weak_skills": ["RAG", "Agent"],
  "missing_skills": ["LangGraph", "Tool Calling", "RAG Evaluation"],
  "risk": "岗位强调 Agent 工程经验，但简历中缺少完整 Agent 项目。",
  "suggestion": "建议补充一个基于 LangGraph 的 Agentic RAG 项目，并在简历中突出工具调用、状态管理和检索增强生成流程。"
}
```

#### 实现方式

Skill Gap Analysis 可以通过以下方式完成：

```text
Prompt
+
结构化输入
+
评分 Rubric
+
Agent 工作流
```

它不需要训练模型，重点是输入结构化、评价标准明确、输出格式稳定。

---

### 6.8 Module H：Recommendation Writer 推荐生成模块

#### 功能目标

将匹配评分、技能差距和岗位信息转化为面向用户的推荐报告。

#### 输入

```text
Top-K 岗位
匹配评分
Skill Gap Analysis
岗位风险点
```

#### 输出

```text
推荐岗位列表
推荐理由
匹配证据
风险提示
投递建议
```

#### 推荐结果格式

```text
推荐岗位 1：公司名称 - 岗位名称

推荐理由：
该岗位要求 XXX，与用户简历中的 XXX 经历匹配度较高。

匹配证据：
- JD 要求：XXX
- 简历证据：XXX

风险提示：
该岗位强调 XXX，但用户简历中相关经验较弱。

投递建议：
建议在简历中补充 XXX 项目描述，突出 XXX 能力。
```

---

### 6.9 Module I：Commute Tool 通勤计算模块

#### 功能目标

在用户明确提出通勤约束时，计算用户住址到公司办公地址的通勤时间。

#### 输入

```text
用户住址
公司办公地址
交通方式偏好
```

#### 输出

```json
{
  "company": "",
  "office_address": "",
  "commute_time_minutes": 45,
  "commute_method": "地铁 + 步行",
  "within_limit": true
}
```

#### 核心任务

```text
地址地理编码
路线规划
通勤时间计算
通勤约束过滤
将通勤结果写入推荐结果
```

#### 调用规则

通勤工具不需要每次调用。  
只有在以下情况下调用：

```text
用户明确要求通勤时间
用户设置通勤时间上限
岗位排序规则包含通勤因素
```

---

## 7. Prompt 工程设计

本项目包含多个 Prompt 任务，每个 Prompt 都应尽量采用结构化输入和结构化输出。

### 7.1 Prompt 类型

```text
简历结构化抽取 Prompt
JD 结构化抽取 Prompt
用户意图识别 Prompt
任务规划 Prompt
岗位方向分类 Prompt
岗位核心能力分析 Prompt
匹配评分 Prompt
Skill Gap Analysis Prompt
推荐报告生成 Prompt
结果自检 Prompt
```

### 7.2 Prompt 设计原则

```text
明确角色
明确输入字段
明确输出 JSON Schema
提供 Few-shot 示例
使用评分 Rubric
要求输出证据
禁止无依据推断
缺失信息返回 null 或 unknown
```

### 7.3 示例：Skill Gap Prompt 输出要求

```json
{
  "requirement": "LangGraph",
  "status": "missing",
  "resume_evidence": [],
  "reason": "岗位要求 LangGraph，但简历中未出现相关项目或工具链描述。",
  "suggestion": "补充 LangGraph 多节点 Agent 工作流项目经验。"
}
```

---

## 8. 数据结构设计

### 8.1 用户画像表

| 字段 | 说明 |
|---|---|
| user_id | 用户 ID |
| name | 姓名 |
| school | 学校 |
| degree | 学历 |
| major | 专业 |
| skills | 技能标签 |
| projects | 项目经历 |
| job_intention | 求职意向 |
| location | 所在地 |
| availability | 可实习时间 |

---

### 8.2 岗位信息表

| 字段 | 说明 |
|---|---|
| job_id | 岗位 ID |
| company | 公司名称 |
| title | 岗位名称 |
| jd | 岗位原文 |
| responsibilities | 岗位职责 |
| requirements | 任职要求 |
| skills | 技能标签 |
| direction | 岗位方向 |
| education_requirement | 学历要求 |
| internship_duration | 实习时长 |
| office_address | 办公地址 |
| source | 数据来源 |
| publish_time | 发布时间 |

---

### 8.3 推荐结果表

| 字段 | 说明 |
|---|---|
| recommendation_id | 推荐记录 ID |
| user_id | 用户 ID |
| job_id | 岗位 ID |
| skill_score | 技能匹配分 |
| project_score | 项目相关性分 |
| education_score | 学历适配分 |
| direction_score | 岗位方向匹配分 |
| final_score | 综合推荐分 |
| skill_gap | 技能差距 |
| reason | 推荐理由 |
| risk | 风险提示 |
| suggestion | 投递建议 |

---

## 9. 开发顺序

### Stage 1：最小可用闭环

目标是先完成从简历到推荐结果的一条完整链路。

```text
PDF 简历
↓
简历结构化 JSON
↓
手动导入 JD
↓
JD 结构化
↓
匹配评分
↓
推荐报告
```

优先完成：

```text
Resume Parser
JD Parser
Match Scorer
Recommendation Writer
```

---

### Stage 2：加入岗位知识库与 RAG 检索

目标是从岗位库中召回候选岗位。

新增：

```text
岗位文本清洗
岗位结构化入库
Embedding 向量化
向量数据库检索
Top-K 候选岗位召回
城市 / 技能 / 方向过滤
```

---

### Stage 3：加入 Planner 与 Router

目标是让系统能够根据用户问题动态选择执行路径。

新增：

```text
Intent Router
Planner Node
Job Source Router
Gap Router
Commute Router
Result Checker
```

---

### Stage 4：加入 Skill Gap Analysis

目标是增强推荐结果的解释性和求职指导价值。

新增：

```text
岗位核心能力拆解
简历证据匹配
强匹配 / 弱匹配 / 缺失判断
风险提示生成
能力补强建议
```

---

### Stage 5：加入通勤工具与前端展示

目标是提升系统实用性和展示效果。

新增：

```text
地图 API
通勤时间计算
Streamlit / Gradio 前端
推荐结果可视化
```

---

## 10. 项目功能边界

系统重点关注：

```text
简历理解
岗位理解
岗位检索
Agent 工具路由
岗位匹配分析
技能差距分析
可解释推荐
```

系统不包含：

```text
大语言模型训练
LoRA / SFT / RLHF
自动投递简历
自动与 HR 沟通
绕过招聘网站反爬机制
保证岗位实时有效性
采集与岗位推荐无关的敏感信息
```

---

## 11. 后续设计重点

后续可以优先继续明确以下内容：

```text
1. Resume Profile JSON Schema
2. JD Structure JSON Schema
3. LangGraph State 字段设计
4. Planner Node 输入输出格式
5. Router 分支规则
6. Match Scorer 评分 Rubric
7. Skill Gap Analysis 输出格式
8. 推荐结果模板
```

建议下一步从以下两个方向之一继续：

```text
方向一：先设计 Module A 的简历解析 JSON Schema
方向二：先设计 LangGraph State 和节点流转
```
