## 严格知识库模式（Strict KB Mode）设计与代码改动说明

### 1. 思路与目标（idea）

- 目标：当检索（基于本地/全局索引与图谱）没有拿到足够的上下文时，模型不得使用预训练常识或外部知识作答，必须直接返回固定文案“与文档内容无关”。
- 核心策略：
  - 在“生成”前后各关键节点加入硬性短路判断：若上下文为空或不足阈值，直接返回固定文案，不调用 LLM。
  - 在“Agent 决策”阶段强制要求工具调用（检索）。若 LLM 没有选择任何工具（意味着要凭常识直答），直接返回固定文案。
  - 在严格模式下，绕过所有缓存（包含快速缓存、全局缓存与相似度召回），避免历史答案或相似问题误命中导致“越权作答”。
  - 关闭缓存的向量相似召回，进一步减少“相似问题误命中”。

效果：任何与知识库无关的问题，或检索未取回有效片段的问题，都会稳定返回“与文档内容无关”。

---

### 2. 代码改动明细（逐文件）

#### 2.1 `agent/graph_agent.py`

- 新增常量：
  - `NO_CONTEXT_ANSWER = "与文档内容无关"`

- 构造函数 `__init__`：
  - 在 `super().__init__` 之后，关闭两级缓存的向量相似检索：
    - `self.cache_manager.enable_vector_similarity = False`
    - `self.global_cache_manager.enable_vector_similarity = False`
  - 开启严格模式开关：`self.strict_kb_mode = True`

- 新增覆盖方法：`_agent_node(self, state)`（替代基类默认实现）
  - 继续做关键词增强（沿用原逻辑）。
  - 调用 `self.llm.bind_tools(self.tools).invoke(messages)` 生成工具决策。
  - 严格模式下检查 `response.tool_calls` 或 `response.additional_kwargs["tool_calls"]`：
    - 如果未调用任何工具，直接返回 `NO_CONTEXT_ANSWER`，禁止“无检索直答”。

- 生成节点 `_generate_node(self, state)`：
  - 在调用 LLM 之前新增强校验：当 `docs` 为空或长度 < 80，直接返回 `NO_CONTEXT_ANSWER`。
  - 保持原有缓存读取逻辑不变（严格模式下会被基类总开关跳过，见下文）。

- Reduce 节点 `_reduce_node(self, state)`：
  - 与生成节点一致，若 `docs` 为空或长度 < 80，直接返回 `NO_CONTEXT_ANSWER`。
  - 修复日志打印变量，将 `print(f"当前使用prompt: {prompt}")` 更正为 `print(f"当前使用prompt: {reduce_prompt}")`。

- 流式生成 `_generate_node_stream(self, state)`：
  - 在流式路径开头新增校验：若 `docs` 为空或长度 < 80，直接 `yield NO_CONTEXT_ANSWER` 并 `return`。

- 流式总控 `_stream_process(self, inputs, config)`：
  - 当检索不足（`content` 长度 < 100）时会尝试本地搜索补充；若仍不足或补充失败，新增直接 `yield NO_CONTEXT_ANSWER` 并 `return` 的分支，避免继续走生成导致幻觉。

> 阈值说明：采用了 80/100 字符两个阈值以兼顾不同分支的稳健性；可后续统一为单一常量。

#### 2.2 `agent/base.py`

- 新增严格模式下的“跳过缓存”总开关：当实例属性 `strict_kb_mode` 为 True 时：
  - `_check_all_caches`：直接返回 `None`，不做任何缓存读取。
  - `ask_with_trace`：跳过全局缓存、快速缓存、常规缓存的三段命中逻辑，直接进入标准流程。
  - `ask`：跳过 `_check_all_caches`，直接走标准流程。
  - `ask_stream`：跳过全局/快速/常规缓存三段命中逻辑，直接进入流式标准流程。

> 说明：即便 `graph_agent.py` 局部仍保留“读取缓存”的代码片段，在严格模式下会被基类层面的判断短路掉，确保“严格知识库”原则不被缓存绕过。

---

### 3. 行为变化与调优点

- 行为变化：
  - 未检索到有效片段时，无论是哪条路径（同步生成、reduce、流式），一律返回“与文档内容无关”。
  - Agent 决策阶段若 LLM 不调用任何检索工具，也直接返回“与文档内容无关”。
  - 严格模式下不再命中任何缓存（含向量相似缓存），避免历史答案/相似问题误命中。

- 可调参数：
  - 文本长度阈值（80/100）可按你的数据分布调整。
  - 如需允许缓存，可在 `GraphAgent.__init__` 中将 `self.strict_kb_mode = False`，或在实例化后显式关闭。

- 性能与风险：
  - 跳过缓存会牺牲一定吞吐与延迟优势；如需兼顾性能，可只关闭“相似度缓存”，保留“精确缓存”，但可能引入极少量误命中风险。

---

### 4. 使用与回滚

- 使用：当前 `GraphAgent` 默认启用严格模式，无需额外配置。
- 回滚：
  - 方式一（推荐临时开关）：实例化后设置 `a.strict_kb_mode = False`。
  - 方式二（持久化）：在 `agent/graph_agent.py::__init__` 中将 `self.strict_kb_mode = True` 改为 `False`。

---

### 5. 验证示例

```bash
# 在 conda 环境 graphrag 中验证
python -c "from agent.graph_agent import GraphAgent; a=GraphAgent(); print(a.ask('这是一个明显不在知识库的问题'))"
# 预期输出：与文档内容无关
```

---

### 6. 变更小结（要点列表）

- `agent/graph_agent.py`
  - 新增：`NO_CONTEXT_ANSWER`
  - 新增：`self.strict_kb_mode = True`
  - 新增：覆盖 `_agent_node`，未触发工具调用则直接返回固定文案
  - 生成、reduce、流式：在调用 LLM 前加入“空/不足上下文”短路
  - 修复：`reduce` 节点日志打印变量
  - 关闭：本地/全局缓存的向量相似召回

- `agent/base.py`
  - 新增：严格模式下跳过缓存的逻辑（`_check_all_caches`、`ask_with_trace`、`ask`、`ask_stream`）

如需进一步把“严格模式开关”接入环境变量或后端 API 入参，我可以继续补充实现。


