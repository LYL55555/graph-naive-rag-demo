from typing import Any, Dict, List, Optional
import html
import json
from elasticsearch import Elasticsearch

from config.settings import (
    ELASTICSEARCH_URL,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD,
    ES_INDEX_NAME,
    ES_DEFAULT_PAGE_SIZE,
    ES_HL_PRE,
    ES_HL_POST,
)
from search.tool.base import BaseSearchTool
from langchain_core.tools import BaseTool


class ElasticKeywordSearchTool(BaseSearchTool):
    """
    关键词检索与高亮 Tool（Elasticsearch）
    返回结构：
    {
      "answer": "",
      "data": {"Chunks": [...], "page": 1, "size": 20, "total": 123},
      "diagnostics": {"mode": "all", "es_took_ms": 3, "index": "rag_chunks"}
    }
    """

    def __init__(self, index_name: str = None):
        super().__init__(cache_dir="./cache/elastic_search")
        self.index = index_name or ES_INDEX_NAME
        es_kwargs = {"request_timeout": 10}
        if ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD:
            es_kwargs["basic_auth"] = (ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
        self.es = Elasticsearch(ELASTICSEARCH_URL, **es_kwargs)
        self.query_analyzer: Optional[str] = None
        self.has_text_raw: bool = False
        self._detect_query_analyzer()

    def _setup_chains(self):
        # 非生成式，保留占位即可
        pass

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        # 关键词检索不需要复杂抽取
        return {"low_level": [], "high_level": []}

    def _sanitize_mode(self, mode: str) -> str:
        m = (mode or "any").lower()
        return m if m in ("any", "all") else "any"

    def _sanitize_page_size(self, page: int, size: int) -> (int, int):
        page = max(1, int(page or 1))
        size = int(size or ES_DEFAULT_PAGE_SIZE)
        if size > 50:
            size = 50
        if size <= 0:
            size = 1
        return page, size

    @staticmethod
    def _escape_except_mark(text: str) -> str:
        if text is None:
            return ""
        open_p, close_p = "__M_OPEN__", "__M_CLOSE__"
        text = text.replace("<mark>", open_p).replace("</mark>", close_p)
        text = html.escape(text)
        return text.replace(open_p, "<mark>").replace(close_p, "</mark>")

    def _build_query(self, query: str, mode: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        should = []
        must = []
        must_not = []

        # 根据索引 mapping 选择查询时的 analyzer（IK 或 n-gram）
        analyzer = self.query_analyzer

        if mode == "all":
            if analyzer == "cn_ngram":
                # n-gram 下的短语严格匹配不可靠，退化为 AND 匹配（所有 n-gram 必须出现）
                clause = {"match": {"text": {"query": query, "operator": "and"}}}
                clause["match"]["text"]["analyzer"] = analyzer
                must.append(clause)
            else:
                # IK 等分词器：相邻短语匹配
                clause = {"match_phrase": {"text": {"query": query}}}
                if analyzer:
                    clause["match_phrase"]["text"]["analyzer"] = analyzer
                must.append(clause)
        else:  # any
            clause = {"match": {"text": {"query": query}}}
            if analyzer:
                clause["match"]["text"]["analyzer"] = analyzer
            must.append(clause)

        # 简单过滤示例
        if filters.get("filename"):
            must.append({"term": {"filename": filters["filename"]}})

        return {"bool": {"must": must, "should": should, "must_not": must_not}}

    def search(self, query_input: Any) -> str:
        # 解析参数（支持字符串 JSON）
        if isinstance(query_input, str) and query_input.strip().startswith("{"):
            try:
                parsed = json.loads(query_input)
                query_input = parsed
            except Exception:
                pass

        if isinstance(query_input, dict):
            query = query_input.get("query", "")
            mode = self._sanitize_mode(query_input.get("mode", "any"))
            page = query_input.get("page", 1)
            size = query_input.get("size", ES_DEFAULT_PAGE_SIZE)
            filters = {k: v for k, v in query_input.items() if k in ("filename", "source_type")}
            summary = bool(query_input.get("summary", False))
        else:
            query = str(query_input)
            mode = "any"
            page, size, filters, summary = 1, ES_DEFAULT_PAGE_SIZE, {}, False

        if not query.strip():
            return json.dumps({
                "answer": "",
                "data": {"Chunks": [], "page": 1, "size": 0, "total": 0},
                "diagnostics": {"mode": "any", "index": self.index, "es_took_ms": 0}
            }, ensure_ascii=False)

        # 分页
        page, size = self._sanitize_page_size(page, size)
        frm = (page - 1) * size

        # 构建 DSL
        dsl = {
            "track_total_hits": True,
            "from": frm,
            "size": size,
            "query": self._build_query(query, mode, filters),
            "highlight": {
                "fields": {"text": {}},
                "pre_tags": [ES_HL_PRE],
                "post_tags": [ES_HL_POST],
                "fragment_size": 180,
                "number_of_fragments": 3,
                "require_field_match": False,
            },
        }

        # 执行检索
        took = 0
        hits = []
        total = 0
        try:
            res = self.es.search(index=self.index, body=dsl)
            took = int(res.get("took", 0))
            total = res.get("hits", {}).get("total", {}).get("value", 0)
            for h in res.get("hits", {}).get("hits", []):
                src = h.get("_source", {})
                fragments = h.get("highlight", {}).get("text", [])
                highlighted = "\n".join(fragments) if fragments else src.get("text", "")[:240]
                # 安全处理，仅保留 <mark>
                highlighted = self._escape_except_mark(highlighted)
                hits.append({
                    "id": src.get("chunk_id") or h.get("_id"),
                    "text": src.get("text", "")[:240],
                    "highlighted": highlighted,
                    "filename": src.get("filename", ""),
                    "chunk_index": src.get("chunk_index", 0),
                    "score": float(h.get("_score", 0.0)),
                })
        except Exception as e:
            # 异常时返回空结果并附加诊断
            diag = {
                "mode": mode,
                "es_took_ms": took,
                "index": self.index,
                "error": str(e),
            }
            return json.dumps({"answer": "", "data": {"Chunks": [], "page": page, "size": size, "total": 0}, "diagnostics": diag}, ensure_ascii=False)

        diag = {"mode": mode, "es_took_ms": took, "index": self.index}
        return json.dumps({
            "answer": "" if not summary else "",  # 后续可加总结
            "data": {"Chunks": hits, "page": page, "size": size, "total": total},
            "diagnostics": diag,
        }, ensure_ascii=False)

    def _detect_query_analyzer(self) -> None:
        """检测 text 字段的 analyzer，用于查询侧设置。"""
        try:
            mp = self.es.indices.get_mapping(index=self.index)
            # 映射结构: {index: {mappings: {properties: {text: {analyzer: 'cn' or 'cn_ngram', ...}}}}}
            for _, idx in mp.items():
                props = idx.get("mappings", {}).get("properties", {})
                text_field = props.get("text", {})
                analyzer = text_field.get("analyzer")
                if analyzer in ("cn", "cn_ngram"):
                    self.query_analyzer = analyzer
                # 检测是否存在 keyword 子字段 raw
                fields = text_field.get("fields", {}) if isinstance(text_field, dict) else {}
                if isinstance(fields, dict) and "raw" in fields:
                    self.has_text_raw = True
                return
        except Exception:
            self.query_analyzer = None

    def get_tool(self) -> BaseTool:
        """覆盖父类，实现对任意关键字参数的兼容，避免 _run 收到未知参数报错。"""
        tool_self = self

        class ElasticRetrievalTool(BaseTool):
            name: str = "elastic_keyword_retriever"
            description: str = "Elasticsearch 关键词检索（支持 any/all/phrase/fuzzy，返回高亮片段）"

            def _run(self_tool, query: Any = None, **kwargs) -> str:
                # 将 LLM 传入的 kwargs 直接作为检索参数；若只有 query 则作为简单输入
                if kwargs:
                    payload = dict(kwargs)
                    if query is not None and "query" not in payload:
                        payload["query"] = query
                    return tool_self.search(payload)
                return tool_self.search(query)

            def _arun(self_tool, query: Any = None, **kwargs) -> str:
                raise NotImplementedError("异步执行未实现")

        return ElasticRetrievalTool()


