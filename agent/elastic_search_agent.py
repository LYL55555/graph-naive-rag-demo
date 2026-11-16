from typing import Dict, Any, List
from langchain_core.messages import AIMessage

from agent.base import BaseAgent
from search.tool.elastic_search_tool import ElasticKeywordSearchTool
import json
from config.settings import ES_DEFAULT_PAGE_SIZE


class ElasticSearchAgent(BaseAgent):
    """Elasticsearch 关键词检索 Agent（直连 Tool，不依赖 LLM 工具调用）"""

    def __init__(self):
        self.search_tool = ElasticKeywordSearchTool()
        super().__init__(cache_dir="./cache/elastic_agent")

    def _setup_tools(self) -> List:
        # 不通过 LLM 工具调用，直接在 _agent_node 中调用 Tool
        return []

    def _add_retrieval_edges(self, workflow):
        # 无检索节点，直接由 _agent_node 生成最终消息
        pass

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        return {"low_level": [], "high_level": []}

    def _agent_node(self, state):
        """直接调用 ES Tool 并返回结果，不走 LLM 工具链。"""
        messages = state["messages"]
        user_query = ""
        if messages and hasattr(messages[-1], "content"):
            user_query = messages[-1].content or ""

        # 解析 JSON 查询或直接字符串
        payload: Any = user_query
        if isinstance(user_query, str) and user_query.strip().startswith("{"):
            try:
                payload = json.loads(user_query)
            except Exception:
                payload = user_query

        result = self.search_tool.search(payload)
        return {"messages": [AIMessage(content=result)]}

    def _generate_node(self, state):
        # 不使用，留空
        return {"messages": [AIMessage(content="")]} 

    # --------- 精确缓存：按完整 payload 区分（mode/all/fuzzy 不互通） ---------
    @staticmethod
    def _canonical_payload(query_or_payload: Any) -> Dict[str, Any]:
        if isinstance(query_or_payload, str) and query_or_payload.strip().startswith("{"):
            try:
                p = json.loads(query_or_payload)
                if isinstance(p, dict):
                    return p
            except Exception:
                pass
        if isinstance(query_or_payload, dict):
            return query_or_payload
        # 纯字符串 -> 默认 any
        return {"query": str(query_or_payload), "mode": "any", "page": 1, "size": ES_DEFAULT_PAGE_SIZE}

    @staticmethod
    def _make_cache_key(payload: Dict[str, Any]) -> str:
        return "ES:" + json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def ask(self, query: str, thread_id: str = "default", recursion_limit: int = 5):
        # 不使用任何缓存，每次直连 ES 检索
        payload = self._canonical_payload(query)
        return self.search_tool.search(payload)

    def ask_with_trace(self, query: str, thread_id: str = "default", recursion_limit: int = 5) -> Dict:
        answer = self.ask(query, thread_id, recursion_limit)
        return {"answer": answer, "execution_log": []}

    async def ask_stream(self, query: str, thread_id: str = "default", recursion_limit: int = 5):
        # 直接使用非流式路径
        yield self.ask(query, thread_id, recursion_limit)

    def mark_answer_quality(self, query: str, is_positive: bool, thread_id: str):
        # 关闭缓存：该方法不做任何存取
        return {"status": "success"}

    def clear_cache_for_query(self, query: str, thread_id: str = "default"):
        # 关闭缓存：该方法不做任何存取
        return True

    # 关闭基类的快速缓存路径，避免不同查询被策略归一导致误命中
    def check_fast_cache(self, query: str, thread_id: str = "default"):
        return None


