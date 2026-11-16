# FAISS索引构建器使用说明

## 概述

FAISS索引构建器 (`FAISSIndexBuilder`) 是一个用于构建和管理基于文档chunks的向量索引的工具。它复用了项目现有的文档读取和分块逻辑，并使用LangChain的FAISS集成来构建高效的向量搜索索引。

## 主要功能

1. **文档读取**: 从 `files/` 目录读取多种格式的文档（PDF、TXT、DOCX等）
2. **文本分块**: 使用 `ChineseTextChunker` 进行智能中文文本分块
3. **向量索引**: 生成embeddings并构建FAISS向量索引
4. **保存/加载**: 支持索引的本地保存和加载
5. **搜索测试**: 提供搜索功能测试接口

## 快速开始

### 1. 基本使用

```python
from processor.faiss_indexer import FAISSIndexBuilder

# 创建构建器
builder = FAISSIndexBuilder()

# 构建索引
faiss_index = builder.build_index(force_rebuild=True)

# 测试搜索
results = builder.test_search("学生手册", k=3)
for result in results:
    print(f"排名{result['rank']}: {result['content'][:100]}...")
```

### 2. 使用便捷函数

```python
from processor.faiss_indexer import build_faiss_index, load_faiss_index

# 构建索引
faiss_index = build_faiss_index(force_rebuild=True)

# 加载索引
loaded_index = load_faiss_index()

# 直接搜索
docs = loaded_index.similarity_search("查询内容", k=5)
```

## 详细配置

### 初始化参数

```python
builder = FAISSIndexBuilder(
    files_dir="files",              # 文档目录路径
    cache_dir="data/faiss_indices", # 索引存储目录（持久化）
    batch_size=100,                 # 批处理大小
    max_workers=4                   # 并行工作线程数
)
```

### 构建选项

```python
faiss_index = builder.build_index(
    file_extensions=['.txt', '.pdf'],  # 指定文件类型
    force_rebuild=False                # 是否强制重建
)
```

## 目录结构

构建完成后，会在缓存目录下生成以下文件：

```
data/faiss_indices/
├── index.faiss      # FAISS向量索引文件
├── index.pkl        # LangChain元数据文件
└── metadata.json    # 构建统计和元数据
```

## 性能特点

### 优势
- **构建速度快**: 相比Neo4j，FAISS索引构建更快
- **搜索效率高**: 纯向量搜索，延迟低
- **内存友好**: 支持批处理，避免内存溢出
- **避免过滤**: 不依赖LLM提取，避免敏感词过滤

### 适用场景
- 传统RAG应用
- 高并发搜索场景
- 对敏感内容处理要求高的场景
- 需要快速部署的场景

## 测试和验证

### 运行完整测试

```bash
python test/test_faiss_indexer.py
```

### 运行示例

```bash
python examples/build_faiss_example.py
```

### 测试内容

1. **基本功能测试**: 索引构建和加载
2. **搜索功能测试**: 多种查询的搜索准确性
3. **性能测试**: 构建和搜索时间测试
4. **边缘情况测试**: 空查询、长查询等
5. **便捷函数测试**: 快捷接口功能

## 常见问题和解决方案

### 1. 依赖问题

如果遇到FAISS安装问题：

```bash
# 安装CPU版本（推荐）
pip install faiss-cpu

# 或安装GPU版本（需要CUDA）
pip install faiss-gpu
```

### 2. 内存不足

如果构建过程中内存不足：

```python
# 减小批处理大小
builder = FAISSIndexBuilder(batch_size=50)

# 或只处理特定类型文件
faiss_index = builder.build_index(file_extensions=['.txt'])
```

### 3. 中文分词问题

如果遇到HanLP相关错误，确保已安装：

```bash
pip install hanlp==2.1.1
```

### 4. 编码问题

如果遇到文件编码问题，FileReader会自动尝试多种编码方式。

### 5. 搜索结果不理想

- 检查embedding模型是否适合你的数据
- 调整chunk大小和重叠参数
- 增加搜索返回的结果数量(k值)

## 与Neo4j版本的比较

| 特性 | FAISS版本 | Neo4j版本 |
|------|-----------|-----------|
| 构建速度 | 快 | 较慢 |
| 搜索速度 | 很快 | 快 |
| 内存使用 | 低 | 中等 |
| 实时更新 | 需重建 | 支持增量 |
| 图关系 | 不支持 | 支持 |
| 敏感词过滤 | 避免 | 可能遇到 |
| 部署复杂度 | 简单 | 复杂 |

## 扩展和自定义

### 自定义embedding模型

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# 使用自定义embedding
custom_embeddings = HuggingFaceEmbeddings(
    model_name="your-model-name"
)

# 修改get_models.py中的get_embeddings_model函数
```

### 自定义分块策略

```python
# 调整ChineseTextChunker参数
from processor.text_chunker import ChineseTextChunker

chunker = ChineseTextChunker(
    chunk_size=800,    # 更大的chunk
    overlap=100        # 更小的重叠
)
```

### 添加新的文件格式支持

在 `FileReader` 类中添加新的文件读取方法，然后在 `supported_extensions` 字典中注册。

## 最佳实践

1. **首次使用**: 先在小数据集上测试
2. **性能优化**: 根据硬件调整batch_size和max_workers
3. **定期更新**: 文档更新后重建索引
4. **备份索引**: 定期备份构建好的索引文件
5. **监控性能**: 使用内置的统计功能监控性能

## 故障排除

如果遇到问题，可以：

1. 查看详细错误信息和堆栈跟踪
2. 检查 `metadata.json` 中的错误日志
3. 尝试减小批处理大小
4. 清空缓存目录重新构建
5. 检查文档目录权限和内容

## 更新日志

- v1.0.0: 初始版本，支持基本的FAISS索引构建和搜索
- 后续版本将支持增量更新和更多自定义选项 