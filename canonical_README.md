# Canonical Dedup & Reference Folding

## 目的
为同事说明本次 **“构建期消岐（canonical/alias）+ 检索折叠”** 改动：  
- 新增/更新了哪些文件  
- 实现了哪些功能  
- 如何使用  
- 注意事项与后续 TODO  

---

## 变更概览

### 新增
- **构建期消岐流水线**：独立 pipeline，统一做清理/去重/语义分组  
- **测试与调试脚本**：最小化验证、A/B 对比、调试重复原因  
- **检索折叠模块**：postprocess 下独立实现，避免不必要的 Neo4j 连接  

### 更新
- 检索结果合并处最小侵入接入折叠展示  
- 构建脚本支持可配置阈值与分块大小  

---

## 新增文件

### 构建期流水线（核心）
- `processor/canonical_dedup_pipeline.py`  
  - 锅炉化清理（去页眉/页脚/版权等）  
  - 语言检测与标准化  
  - 文档内去重（默认开，可关）  
  - 精确指纹去重（SHA1）  
  - 极端语义相似分组（多语嵌入，阈值可配）  
  - 仅写 canonical 到 FAISS/Neo4j；alias 输出 JSON 映射  

### 构建/测试入口
- `examples/run_canonical_build.py`：构建入口（CLI）  
- `examples/test_faiss_canonical_search.py`：加载 canonical 索引做简单检索  
- `examples/abtest_baseline_vs_canonical.py`：baseline vs canonical A/B 对比  
- `evaluator_new/test/run_chunk_evaluation_example.py`：离线最小化测试（假嵌入）  
- `examples/debug_canonical_dedup.py`：打印指纹/重复组调试  

### 检索折叠后处理
- `postprocess/reference_postprocess.py`  
  - 折叠工具（alias_mapping.json）  
  - 锅炉化过滤  
  - 时间过滤占位  
  - **说明**：此前短暂放在 `search/utils/`，现迁移至 `postprocess/`  

---

## 修改文件
- `search/tool/reasoning/search.py`  
  - 在 `_merge_results` 末尾接入折叠与过滤：  
    ```python
    from postprocess.reference_postprocess import fold_duplicates, filter_boilerplate
    ```
  - 合并完成后执行折叠与锅炉化过滤，统计写入 `result["fold_stats"]`  

- `examples/run_canonical_build.py`  
  - CLI 新增/修复：`--faiss/--no-faiss`、`--neo4j/--no-neo4j`、`--within/--cross`、`--chunk-size/--overlap`  

- `evaluator_new/test/run_chunk_evaluation_example.py`  
  - 内置假嵌入，便于离线测试（无需外部 API）  

---

## 实现功能

### 构建阶段（Build）
- 锅炉化清理、语言检测与标准化  
- 文档内去重（默认开）  
- 精确指纹去重（跨 chunk）  
- 极端语义相似分组（同文档/跨文档）  
- canonical 选择：信息量优先 > 语言偏好(zh) > 来源/时间  
- **写入产物**：  
  - FAISS：仅写 canonical（`data/canonical_index/canonical_faiss/`）  
  - Neo4j：仅建 canonical 的 `__Chunk__`（可选）  
  - alias：`data/canonical_index/alias_mapping.json`  

### 检索阶段（Reference）
- 合并结果后按 `canonical_chunk_id` 折叠  
- 锅炉化过滤  
- 时间过滤（as_of/is_current 钩子，预留）  

---

## 使用方式（常用命令）

### 构建 canonical 索引（单一入口）
```bash
# 通过统一入口执行（默认写入 FAISS 与 Neo4j，产出 alias 与等价组摘要）
python build/main.py canonical

# 可通过环境变量覆盖参数：
#   CANONICAL_FILES_DIR, CANONICAL_OUTPUT_DIR, CANONICAL_WRITE_FAISS(1/0), CANONICAL_WRITE_NEO4J(1/0)
#   CANONICAL_WITHIN, CANONICAL_CROSS, CANONICAL_CHUNK_SIZE, CANONICAL_OVERLAP, CANONICAL_AS_OF
CANONICAL_FILES_DIR=./files CANONICAL_OUTPUT_DIR=./data/canonical_index \
CANONICAL_WITHIN=0.985 CANONICAL_CROSS=0.98 CANONICAL_CHUNK_SIZE=200 CANONICAL_OVERLAP=50 \
python build/main.py canonical
```

### 调整阈值/分块
```bash
python examples/run_canonical_build.py --files ./files --out ./data/canonical_index --faiss --no-neo4j --within 0.985 --cross 0.98 --chunk-size 200 --overlap 50
```

### 快速验证
```bash
python examples/test_faiss_canonical_search.py --index ./data/canonical_index/canonical_faiss --k 3
```

### A/B 对比
```bash
python examples/abtest_baseline_vs_canonical.py --baseline ./data/faiss_indices --canonical ./data/canonical_index/canonical_faiss --k 5
```

### 检索折叠
```bash
python examples/test_reference_folding.py --index ./data/canonical_index/canonical_faiss --k 5
```

### 定位重复（调试）
```bash
python examples/debug_canonical_dedup.py --root ./files --only "txt文件/历史上的今天.txt" --within 0.985 --cross 0.98 --max 200 --chunk-size 200 --overlap 50
```

---

## 注意事项
- **canonical-only 策略**：FAISS/Neo4j 仅写 canonical，alias 不进索引 → 检索端折叠量通常为 0（合理现象）。  
- **阈值/分块敏感**：完全重复靠指纹，近似靠嵌入阈值；分块过大可能漏判。  
- **Neo4j 避免无谓连接**：折叠模块放在 `postprocess/`，避免触发初始化。  
- **文档内去重**：默认开；若重复片段无空行，可扩展分段规则。  
- **嵌入模型一致性**：固定模型版本，保证阈值稳定。  
- **产物位置**：  
  - 索引：`data/canonical_index/canonical_faiss/`  
  - alias 映射：`data/canonical_index/alias_mapping.json`  
  - 等价组摘要：`data/canonical_index/equivalence_summary.json`  

---

## 已知限制
- 文档内去重依赖空行/行分割；连续拼接的重复可能漏判。  
- 检索期语义折叠未默认启用，目前仅 alias 映射折叠。  

---

## 新增能力
- **单一入口整合构建**：`python build/main.py canonical` 一键执行“分块→清理→指纹/语义去重→写入 FAISS/Neo4j→输出 alias 映射/等价组摘要”。  
- **时间语义完善**：构建期为每个 chunk 写入 `valid_from/valid_to/is_current`，支持“最新优先/按 as_of 查询”的时间感知索引与引用。  
- **等价组摘要**：构建期输出 `equivalence_summary.json`，帮助前端展示“还有 N 条近似片段”。  

---

## 已实现内容
- 新增 `processor/canonical_dedup_pipeline.py`：消岐管线  
- 构建入口：`examples/run_canonical_build.py`（阈值/分块可配）  
- 检索端最小侵入折叠：`search/tool/reasoning/search.py` 调用 `postprocess/reference_postprocess.py`  
- A/B 对比、最小化测试与调试脚本（定位重复原因）  

---

## 跨设备零起步部署与运行指南（从零到可用）

本节面向“尚未有任何 canonical 相关代码/环境”的设备，按此步骤可在全新环境完成构建与验证。

### 1. 环境要求
- **Python**: 推荐 3.10 ~ 3.11
- **操作系统**: macOS/Linux/Windows（Linux 推荐）
- **Neo4j**: 已安装并可连接（本地或远程均可）

### 2. 拉取代码
```bash
git clone https://your.repo/graph-rag-agent.git
cd graph-rag-agent-master
```

### 3. 创建虚拟环境并安装依赖
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

如需处理多格式文档，Linux 可能需要系统依赖（可选）：
```bash
sudo apt-get install -y python-dev-is-python3 libxml2-dev libxslt1-dev antiword unrtf poppler-utils
```

### 4. 配置环境变量（.env）
在项目根目录创建 `.env` 文件（示例）：
```env
# Neo4j（如使用 Neo4j 构建/查询）
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
# 可选
NEO4J_DATABASE=neo4j

# 向量嵌入（当前默认 SiliconFlow Embeddings）
SILICONFLOW_MODEL=BAAI/bge-m3
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx

# 如改用 OpenAI，可配置：
# OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large
# OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1

# 其他（可选）
# DUPLICATE_LEVEL=extreme
```

### 5. 准备数据目录
把要构建的原始文档放到 `files/` 目录（已在仓库中），支持 `.txt/.md`（PDF/DOCX 请先抽取为文本）。

目录示例：
```
files/
  ├─ 文档A.txt
  ├─ 文档B.md
  └─ txt文件/
      └─ 历史上的今天.txt
```

### 6. 一键执行 canonical 构建
```bash
python build/main.py canonical
```

默认行为：
- 分块 → 清理 → 指纹/语义去重
- 写入 FAISS（仅 canonical）到 `data/canonical_index/canonical_faiss/`
- 写入 Neo4j 的 `__Chunk__` 节点（仅 canonical，含时间属性）
- 产出 `alias_mapping.json` 与 `equivalence_summary.json`

可通过环境变量覆盖参数（示例）：
```bash
CANONICAL_FILES_DIR=./files \
CANONICAL_OUTPUT_DIR=./data/canonical_index \
CANONICAL_WRITE_FAISS=1 \
CANONICAL_WRITE_NEO4J=1 \
CANONICAL_WITHIN=0.985 \
CANONICAL_CROSS=0.98 \
CANONICAL_CHUNK_SIZE=200 \
CANONICAL_OVERLAP=50 \
CANONICAL_AS_OF=2023-09-01 \
python build/main.py canonical
```

运行完成后，关键产物：
- FAISS：`data/canonical_index/canonical_faiss/`
- Alias 映射：`data/canonical_index/alias_mapping.json`
- 等价组摘要：`data/canonical_index/equivalence_summary.json`
- Neo4j：`__Chunk__` 节点，属性包括 `id,text,fileName,position,length,language,validFrom,validTo,isCurrent`

### 7. 验证与调试
- 简单检索：
```bash
python examples/test_faiss_canonical_search.py --index ./data/canonical_index/canonical_faiss --k 3
```
- 折叠展示：
```bash
python examples/test_reference_folding.py --index ./data/canonical_index/canonical_faiss --k 5
```
- 去重调试：
```bash
python examples/debug_canonical_dedup.py --root ./files --only "txt文件/历史上的今天.txt" --within 0.985 --cross 0.98 --max 200 --chunk-size 200 --overlap 50
```

### 8. 时间语义与命名建议
- 构建期：会为每个 chunk 填充 `valid_from/valid_to/is_current`；
  - `valid_from` 默认尝试从文件名解析（支持 `YYYY[-MM[-DD]]`/`YYYYMM[DD]`），解析失败则使用默认值
  - 传入 `CANONICAL_AS_OF` 时按 as_of 判定 `is_current`；未传则自动将同一“文档基底名”的最新版本标记为当前
- 检索期：
  - 折叠：`postprocess/reference_postprocess.fold_duplicates`
  - 时间过滤：`postprocess/reference_postprocess.time_filter(chunks, as_of="YYYY-MM-DD")`

文件命名建议（便于日期解析）：
```
<标题>_2024-09-01.txt
<标题>-202409.txt
<标题>20230901.md
```

### 9. 需要新增/修改的文件清单（相较旧版本）
- 已修改：
  - `processor/canonical_dedup_pipeline.py`：新增时间语义、等价组摘要输出、代表选择“最新优先”，写入 Neo4j 的时间属性
  - `build/main.py`：新增 `canonical` 子命令，实现单一入口一键构建
  - `postprocess/reference_postprocess.py`：完成 `time_filter` 实装，支持 as_of 与 is_current 策略
  - `canonical_README.md`：本文档
- 若从较旧版本迁移，需要新增（本仓库已包含）：
  - `processor/canonical_dedup_pipeline.py`
  - `postprocess/reference_postprocess.py`
  - `examples/run_canonical_build.py`
  - 其余 canonical 相关文件见上文“新增文件/修改文件”小节

如目标设备没有上述变更文件，请从最新仓库将这些文件拷贝到相同路径覆盖即可。

### 10. 常见问题
- 找不到 `rich`/`python-dotenv`/`faiss-cpu`：请确认已执行 `pip install -r requirements.txt`
- Neo4j 无法连接：检查 `.env` 的 `NEO4J_URI/USERNAME/PASSWORD`，并确认数据库服务已启动
- 无嵌入结果：确认 `SILICONFLOW_API_KEY` 与 `SILICONFLOW_MODEL` 正确；或改用 OpenAI/HuggingFace 并调整 `model/get_models.py`
- PDF/DOCX 无法解析：请先离线抽取为 `.txt/.md` 再放入 `files/`
