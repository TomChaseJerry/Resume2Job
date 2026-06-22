# -*- coding: utf-8 -*-
"""岗位「方向标签」词表（粗粒度方向层，**仅供 direction_bonus 使用**）。

与技能匹配层（match_scorer 的 SYNONYM_GROUPS / SKILL_IMPLICATIONS）是两套独立的表：
    - 技能层：细粒度，判断「会不会某个具体技术」，影响 skill_score；
    - 方向层（本文件）：粗粒度，判断「岗位大方向对不对口」，只影响 direction_bonus（≤6）。

匹配方式（match_scorer._map_direction_tags）：把 JD / 用户偏好文本小写后，对每个标签的
关键词做**子串包含**判断，命中即归入该标签集合。compute_direction_bonus 据两侧标签集合
的交集 / 同上层（DIRECTION_PARENTS）给分。

==== 扩词时必须守住的两条边界（否则会大量误命中 / 把弱相关误判为「核心一致」）====
1. **不要把泛词放进标签**：如单独的「人工智能 / 大模型 / 模型训练 / 对齐 / 量化 / 机器学习」。
   它们区分度低，会让无关 JD 命中方向标签。用更具体的措辞（「大模型应用」「安全对齐」「量化交易」）。
2. **同一个关键词不要放进多个标签**：子串匹配下，一词进多标签会让弱相关被判成「核心一致 +6」。
   一个 JD 可以命中多个方向标签（不同关键词），但**单个关键词必须唯一归属一个标签**。
   注意子串包含也算「重复」（如「时间序列」是「金融时间序列」的子串 → 别两边都放）。

变更本文件即可扩方向，无需改 match_scorer 逻辑（它只 import 这两个表）。
"""

DIRECTION_TAG_VOCAB = {
    # ===== LLM / 大模型 =====
    "LLM_APPLICATION": [
        "大模型应用", "大模型工程", "llm 应用", "llm application",
        "llm工程", "ai agent", "agent开发", "agent 开发",
        "agent工程", "agent 工程", "rag", "检索增强",
        "知识库问答", "工具调用", "function calling",
        "工作流编排", "prompt工程", "prompt engineering",
    ],
    "POST_TRAINING": [
        "后训练", "post-training", "post training",
        "监督微调", "指令微调", "sft", "dpo", "orpo", "grpo",
        "rlhf", "奖励模型", "reward model", "偏好优化",
        "偏好学习", "偏好对齐", "模型对齐", "安全对齐",
        "对齐训练", "人类反馈",
    ],
    "FOUNDATION_MODEL_TRAINING": [
        "预训练", "pre-training", "pretraining",
        "持续预训练", "继续预训练", "continued pretraining",
        "语言模型训练", "基础模型训练", "tokenizer",
        "scaling law", "scaling laws", "训练数据配比", "语料配比",
    ],

    # ===== 搜索 / 推荐 / 广告 =====
    "SEARCH_RECOMMENDATION": [
        "搜广推", "搜索推荐", "推荐系统", "推荐算法",
        "召回", "粗排", "精排", "重排", "排序模型",
        "学习排序", "ltr", "点击率预估", "ctr", "cvr",
        "广告算法", "广告推荐", "搜索算法", "检索排序",
        "query理解", "query 理解", "双塔模型", "two-tower",
        "two tower", "协同过滤", "多目标排序",
    ],

    # ===== 强化学习 / 决策控制 =====
    "REINFORCEMENT_LEARNING": [
        "强化学习", "reinforcement learning", "策略优化",
        "策略梯度", "价值函数", "值函数", "q-learning",
        "q learning", "ppo", "sac", "td3", "a2c", "a3c",
        "actor-critic", "actor critic", "离线强化学习",
        "offline rl", "在线强化学习", "多智能体强化学习",
        "模仿学习", "imitation learning",
    ],
    "OPTIMIZATION_OR": [
        "运筹优化", "operations research", "组合优化",
        "combinatorial optimization", "整数规划", "integer programming",
        "线性规划", "linear programming", "调度优化",
        "供应链优化", "资源分配", "启发式算法", "metaheuristic",
    ],

    # ===== 感知 / 视觉 / 语音 =====
    "COMPUTER_VISION": [
        "计算机视觉", "computer vision", "视觉算法",
        "目标检测", "语义分割", "实例分割", "图像分类",
        "关键点检测", "姿态估计", "视觉识别", "图像识别",
        "ocr", "光学字符识别", "视觉跟踪", "多目标跟踪",
        "3d视觉", "3d vision", "点云", "point cloud",
        "nerf",
    ],
    "MULTIMODAL": [
        "多模态", "multimodal", "跨模态", "cross-modal",
        "视觉语言", "vision-language", "vlm", "mllm",
        "图文理解", "图文检索", "图文生成",
        "视频理解", "video understanding", "多模态大模型",
    ],
    "SPEECH_AUDIO": [
        "语音识别", "自动语音识别", "asr",
        "语音合成", "tts", "text-to-speech",
        "说话人识别", "speaker recognition", "声纹",
        "语音唤醒", "关键词唤醒", "语音增强",
        "音频理解", "音频算法", "声音事件检测",
        "音频生成", "降噪",
    ],
    "AIGC_GENERATION": [
        "aigc", "生成式ai", "生成式人工智能",
        "图像生成", "视频生成", "文生图", "文生视频",
        "text-to-image", "text-to-video",
        "扩散模型", "diffusion model", "stable diffusion",
        "controlnet", "数字人", "虚拟人",
        "3d生成", "3d generation", "内容生成",
    ],

    # ===== NLP / 图谱 / 图学习 =====
    "NATURAL_LANGUAGE_PROCESSING": [
        "自然语言处理", "natural language processing", "nlp",
        "文本分类", "命名实体识别", "实体识别",
        "关系抽取", "信息抽取", "文本匹配",
        "机器翻译", "文本生成", "对话系统", "问答系统",
    ],
    "KNOWLEDGE_GRAPH": [
        "知识图谱", "knowledge graph", "图数据库",
        "graph database", "知识推理", "图谱推理",
        "实体链接", "entity linking", "关系推理",
        "图谱问答",
    ],
    "GRAPH_LEARNING": [
        "图神经网络", "graph neural network", "gnn",
        "图表示学习", "graph representation learning",
        "图嵌入", "graph embedding", "图学习",
        "异构图", "heterogeneous graph",
    ],

    # ===== 数据智能 / 金融 =====
    "TIME_SERIES_FORECASTING": [
        "时间序列", "time series", "时序预测",
        "时序建模", "序列预测", "forecasting",
        "需求预测", "销量预测", "流量预测",
        "时序异常", "预测维护",
    ],
    "RISK_CONTROL": [
        "风控", "风险控制", "风险建模", "反欺诈",
        "欺诈检测", "信用评分", "信用风险",
        "信贷风控", "贷前", "贷中", "贷后",
        "反洗钱", "aml", "异常交易", "支付风控",
    ],
    "DATA_MINING": [
        "数据挖掘", "data mining", "数据科学",
        "data science", "用户增长", "增长算法",
        "用户画像", "人群挖掘", "流失预测",
        "留存预测", "因果推断", "causal inference",
        "uplift", "a/b测试", "a/b test", "ab测试",
        "实验设计", "运营算法",
    ],
    "QUANTITATIVE_FINANCE": [
        "量化交易", "量化金融", "量化投资",
        "quantitative finance", "quant trading",
        "alpha因子", "alpha factor", "因子模型",
        "高频交易", "high-frequency", "做市",
        "market making", "portfolio optimization",
    ],

    # ===== 具身 / 自动驾驶 =====
    "ROBOTICS_EMBODIED": [
        "具身智能", "embodied ai", "机器人",
        "robotics", "机械臂", "manipulation",
        "抓取", "grasping", "机器人学习",
        "robot learning", "视觉伺服", "sim2real",
        "仿真训练", "具身导航",
    ],
    "AUTONOMOUS_DRIVING": [
        "自动驾驶", "autonomous driving", "智能驾驶",
        "智驾", "adas", "高级辅助驾驶", "bev",
        "车路协同", "v2x", "端到端驾驶",
        "end-to-end driving", "自动泊车",
        "驾驶感知", "驾驶决策",
    ],

    # ===== AI 系统与基础设施 =====
    "AI_INFRA": [
        "ai infra", "ai infrastructure", "机器学习平台",
        "ml platform", "模型服务", "model serving",
        "推理优化", "inference optimization", "大模型推理",
        "分布式训练", "distributed training", "参数服务器",
        "parameter server", "训练平台", "推理引擎",
        "vllm", "tensorrt", "triton", "cuda",
        "算子优化", "kernel优化", "编译优化",
        "模型并行", "流水线并行", "集群调度",
    ],
}


# 子方向 → 上层方向（compute_direction_bonus 用于「同上层、子方向接近 +4」档）。
# 每个 DIRECTION_TAG_VOCAB 的标签都应在此有父标签。
DIRECTION_PARENTS = {
    # LLM
    "LLM_APPLICATION": "LLM_PRODUCT",
    "POST_TRAINING": "LLM_TRAINING",
    "FOUNDATION_MODEL_TRAINING": "LLM_TRAINING",

    # 搜广推
    "SEARCH_RECOMMENDATION": "SEARCH_RECOMMENDATION",

    # 决策与控制
    "REINFORCEMENT_LEARNING": "DECISION_CONTROL",
    "ROBOTICS_EMBODIED": "DECISION_CONTROL",
    "AUTONOMOUS_DRIVING": "DECISION_CONTROL",
    "OPTIMIZATION_OR": "DECISION_OPTIMIZATION",

    # 感知
    "COMPUTER_VISION": "PERCEPTION",
    "MULTIMODAL": "PERCEPTION",
    "SPEECH_AUDIO": "PERCEPTION",
    "AIGC_GENERATION": "GENERATIVE_MEDIA",

    # 语言、知识与图
    "NATURAL_LANGUAGE_PROCESSING": "LANGUAGE",
    "KNOWLEDGE_GRAPH": "GRAPH_KNOWLEDGE",
    "GRAPH_LEARNING": "GRAPH_KNOWLEDGE",

    # 数据与金融
    "TIME_SERIES_FORECASTING": "DATA_INTELLIGENCE",
    "RISK_CONTROL": "DATA_INTELLIGENCE",
    "DATA_MINING": "DATA_INTELLIGENCE",
    "QUANTITATIVE_FINANCE": "FINANCE",

    # AI 系统
    "AI_INFRA": "ML_SYSTEMS",
}
