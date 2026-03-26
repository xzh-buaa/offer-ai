# -*- coding: utf8 -*-
"""
面试场景动态热词配置
用于提升阿里云 FunASR 语音识别对专业术语的准确率
weight 范围：1-5，数值越高优先级越高
"""

# ========== AI/大模型核心术语 ==========
LLM_TERMS = [
    {"text": "大语言模型", "weight": 5},
    {"text": "大模型", "weight": 4},
    {"text": "LLM", "weight": 4},
    {"text": "Transformer", "weight": 4},
    {"text": "Attention 机制", "weight": 4},
]

# ========== LangChain 生态 ==========
LANGCHAIN_TERMS = [
    {"text": "LangChain", "weight": 5},
    {"text": "LangGraph", "weight": 5},
    {"text": "Agent", "weight": 5},
    {"text": "智能体", "weight": 4},
    {"text": "Tool", "weight": 3},
    {"text": "Chain", "weight": 3},
]

# ========== RAG 与知识图谱 ==========
RAG_KG_TERMS = [
    {"text": "RAG", "weight": 5},
    {"text": "Graph RAG", "weight": 5},
    {"text": "知识图谱", "weight": 5},
    {"text": "Neo4j", "weight": 5},
    {"text": "图数据库", "weight": 4},
    {"text": "向量数据库", "weight": 4},
    {"text": "Embedding", "weight": 4},
    {"text": "检索增强生成", "weight": 4},
]

# ========== ChatBI 相关 ==========
CHATBI_TERMS = [
    {"text": "ChatBI", "weight": 5},
    {"text": "商业智能", "weight": 4},
    {"text": "数据分析", "weight": 3},
    {"text": "可视化", "weight": 3},
]

# ========== 强化学习 ==========
RL_TERMS = [
    {"text": "强化学习", "weight": 4},
    {"text": "PPO", "weight": 4},
    {"text": "DQN", "weight": 3},
    {"text": "策略梯度", "weight": 3},
    {"text": "奖励函数", "weight": 3},
]

# ========== 面试高频通用词 ==========
INTERVIEW_COMMON_TERMS = [
    {"text": "论文", "weight": 5},
    {"text": "项目", "weight": 3},
    {"text": "机制", "weight": 3},
    {"text": "设计", "weight": 3},
    {"text": "优化", "weight": 3},
    {"text": "架构", "weight": 3},
]


def get_interview_vocabulary():
    """
    获取完整的面试场景热词表

    Returns:
        list: 热词列表，格式为 [{"text": "术语", "weight": 权重}, ...]
    """
    vocabulary = (
            LLM_TERMS +
            LANGCHAIN_TERMS +
            RAG_KG_TERMS +
            CHATBI_TERMS +
            RL_TERMS +
            INTERVIEW_COMMON_TERMS
    )

    # 去重（保留最高权重）
    text_to_weight = {}
    for term in vocabulary:
        text = term["text"]
        weight = term["weight"]
        if text not in text_to_weight or weight > text_to_weight[text]:
            text_to_weight[text] = weight

    # 转换回列表格式
    final_vocabulary = [
        {"text": text, "weight": weight}
        for text, weight in text_to_weight.items()
    ]

    # 按权重降序排列
    final_vocabulary.sort(key=lambda x: x["weight"], reverse=True)

    return final_vocabulary


# 测试代码
if __name__ == "__main__":
    vocab = get_interview_vocabulary()
    print(f"✅ 共加载 {len(vocab)} 个热词")
    print("\n前 10 个高优先级热词：")
    for i, term in enumerate(vocab[:10], 1):
        print(f"{i}. {term['text']} (权重：{term['weight']})")
