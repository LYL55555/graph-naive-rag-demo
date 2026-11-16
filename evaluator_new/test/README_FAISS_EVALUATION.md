# FAISS版本Naive RAG Agent评估指南

## 概述

本模块为FAISS版本的Naive RAG Agent提供专门的评估功能，作为原有评估体系的additional module。

## 文件结构

```
evaluator/test/
├── evaluate_naive_faiss_agent.py    # FAISS版本单独评估脚本
├── evaluate_all_agents.py           # 包含naive_faiss的综合评估
└── README_FAISS_EVALUATION.md      # 本说明文档
```

## 快速开始

### 1. 前置条件

确保已构建FAISS索引：
```bash
# 如果还未构建FAISS索引
python examples/build_faiss_example.py
```

### 2. 单独评估FAISS版本Naive Agent

```bash
# 基础评估
python evaluator/test/evaluate_naive_faiss_agent.py \
  --questions_file evaluator/test/questions.json

# 带标准答案的完整评估
python evaluator/test/evaluate_naive_faiss_agent.py \
  --questions_file evaluator/test/questions.json \
  --golden_answers_file evaluator/test/answer.json \
  --verbose

# 仅评估检索性能
python evaluator/test/evaluate_naive_faiss_agent.py \
  --questions_file evaluator/test/questions.json \
  --eval_type retrieval \
  --verbose
```

### 3. 综合对比评估（包含FAISS版本）

```bash
# 评估所有Agent（包括naive_faiss）
python evaluator/test/evaluate_all_agents.py \
  --questions_file evaluator/test/questions.json \
  --agents graph,hybrid,naive,naive_faiss,fusion,deep

# 仅对比naive和naive_faiss
python evaluator/test/evaluate_all_agents.py \
  --questions_file evaluator/test/questions.json \
  --agents naive,naive_faiss \
  --verbose
```

## 评估指标

FAISS版本Naive Agent使用与传统Naive Agent相同的评估指标：

### 答案质量指标
- `em`: 精确匹配
- `f1`: F1分数
- `response_coherence`: 回答连贯性
- `factual_consistency`: 事实一致性
- `answer_comprehensiveness`: 答案完整性
- `llm_evaluation`: LLM评估

### 检索性能指标
- `retrieval_precision`: 检索精确度
- `retrieval_utilization`: 检索利用率
- `retrieval_latency`: 检索延迟
- `chunk_utilization`: 文本块利用率

## 配置参数

### evaluate_naive_faiss_agent.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_dir` | `./evaluation_results/naive_faiss_agent` | 结果保存目录 |
| `--questions_file` | 必需 | 问题文件路径 |
| `--golden_answers_file` | None | 标准答案文件路径 |
| `--verbose` | False | 详细输出模式 |
| `--metrics` | "" | 自定义评估指标 |
| `--eval_type` | "all" | 评估类型 |
| `--faiss_cache_dir` | "data/faiss_indices" | FAISS索引目录 |

## 结果输出

评估结果将保存在指定目录下：

```
evaluation_results/naive_faiss_agent/
├── evaluation.log              # 评估日志
├── evaluation_results.json     # 评估结果JSON
└── answers/                   # Agent回答记录
    ├── question_0.json
    ├── question_1.json
    └── ...
```

## 对比分析

### FAISS vs Neo4j Naive Agent

通过综合评估可以对比：
- **检索速度**: FAISS通常更快
- **检索精度**: 取决于embedding质量和相似度阈值
- **资源占用**: FAISS内存占用更少
- **扩展性**: FAISS更适合大规模文档

### 示例对比命令

```bash
# 生成对比报告
python evaluator/test/evaluate_all_agents.py \
  --questions_file evaluator/test/questions.json \
  --agents naive,naive_faiss \
  --save_dir ./evaluation_results/naive_comparison \
  --verbose
```

## 故障排除

### 常见问题

1. **FAISS索引不存在**
   ```
   错误: FAISS索引目录不存在: data/faiss_indices
   解决: 先运行 python examples/build_faiss_example.py
   ```

2. **导入错误**
   ```
   错误: ModuleNotFoundError: No module named 'langchain_community'
   解决: pip install langchain-community
   ```

3. **评估指标不可用**
   ```
   错误: 某些指标无法计算
   解决: 检查Neo4j连接和LLM配置
   ```

### 调试建议

1. 使用 `--verbose` 参数查看详细日志
2. 检查 `evaluation.log` 文件获取错误信息
3. 验证FAISS索引是否正确加载
4. 确认问题文件格式正确

## 扩展

### 添加自定义指标

在 `evaluator/evaluator_config/agent_evaluation_config.py` 中为 `naive_faiss` 添加新的评估指标：

```python
"naive_faiss": {
    "retrieval_metrics": [
        # 现有指标...
        "custom_metric"  # 新增自定义指标
    ]
}
```

### 自定义FAISS配置

可以通过修改 `NaiveRagAgent` 初始化参数来测试不同的FAISS配置：

```python
# 在 evaluator/utils/eval_utils.py 中
elif agent_type == "naive_faiss":
    from agent.naive_rag_agent import NaiveRagAgent
    return NaiveRagAgent(
        faiss_cache_dir="cache/custom_faiss",  # 自定义路径
        # 其他参数...
    )
``` 