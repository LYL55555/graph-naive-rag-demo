from typing import List, Dict, Any, Optional
import math
import time
import os
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from config.prompt import NAIVE_PROMPT
from config.settings import response_type, naive_description
from search.tool.base import BaseSearchTool
from processor.faiss_indexer import load_faiss_index


class NaiveSearchTool(BaseSearchTool):
    """简单的Naive RAG搜索工具，使用FAISS进行向量搜索"""
    
    def __init__(
        self,
        faiss_cache_dir: str = None,
        enable_reranker: bool = False,
        reranker_model_name: str = "BAAI/bge-reranker-base",
        weight_temperature: float = 1.0,
        context_max_chars: int = 3500,
    ):
        """
        初始化Naive搜索工具
        
        Args:
            faiss_cache_dir: FAISS索引缓存目录
        """
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/naive_search")
        
        # FAISS相关配置
        self.faiss_cache_dir = faiss_cache_dir
        self.faiss_index: Optional[FAISS] = None
        self.index_loaded = False
        
        # 搜索参数设置
        self.top_k = 3  # 检索的最大文档数量
        self.similarity_threshold = 1.0  # 余弦距离阈值（FAISS返回距离，越小越相似）

        # 重排序与加权配置
        self.enable_reranker = bool(enable_reranker)  # 是否启用跨编码器重排序
        self.reranker_model_name = reranker_model_name  # 适配中文的reranker模型
        self._reranker = None  # 延迟加载
        self.weight_temperature = float(weight_temperature)  # 权重温度，>1 放大差异，<1 平滑差异
        self.context_max_chars = int(context_max_chars)  # 传入LLM的上下文总字符预算，按权重分配
        self.show_diagnostics = True  # 是否在输出中追加检索诊断

        # 诊断用缓存
        self._last_faiss_candidates = []
        self._last_ranked = []
        
        # 设置处理链
        self._setup_chains()
        
        # 尝试加载FAISS索引
        self._load_faiss_index()

    def _ensure_reranker(self):
        """按需加载CrossEncoder重排序模型。"""
        if not self.enable_reranker:
            return None
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model_name)
            return self._reranker
        except Exception as e:
            print(f"加载重排序模型失败，将回退到向量距离加权: {e}")
            self._reranker = None
            self.enable_reranker = False
            return None

    @staticmethod
    def _softmax(values: List[float], temperature: float = 1.0) -> List[float]:
        if not values:
            return []
        if temperature <= 0:
            temperature = 1.0
        scaled = [v / temperature for v in values]
        max_v = max(scaled)
        exp_v = [math.exp(v - max_v) for v in scaled]
        s = sum(exp_v) or 1.0
        return [v / s for v in exp_v]

    def _rerank_and_weight(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对FAISS召回结果进行重排序并计算权重。

        - 若启用CrossEncoder，则用跨编码器评分并softmax归一化为权重。
        - 否则，用FAISS距离转换为相似度(sim≈1-distance)并归一化为权重。
        """
        if not results:
            return []

        # 优先使用CrossEncoder重排序
        reranker = self._ensure_reranker()
        if reranker is not None:
            try:
                pairs = [(query, item.get("text", "")) for item in results]
                scores = reranker.predict(pairs)
                weights = self._softmax(list(map(float, scores)), temperature=self.weight_temperature)
                enriched = []
                for item, score, w in zip(results, scores, weights):
                    new_item = dict(item)
                    new_item["rerank_score"] = float(score)
                    new_item["weight"] = float(w)
                    enriched.append(new_item)
                # 按CrossEncoder分数降序排序
                enriched.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                self._last_ranked = enriched[: self.top_k]
                return self._last_ranked
            except Exception as e:
                print(f"CrossEncoder重排序失败，回退到向量距离加权: {e}")

        # 回退：基于FAISS返回的距离分数进行相似度与权重计算
        sims: List[float] = []
        for item in results:
            dist = float(item.get("score", 1.0))  # 距离，越小越相似
            # 将距离映射为相似度（尝试1 - dist，并裁剪到[0,1]）
            sim = 1.0 - dist
            if sim < 0:
                sim = 0.0
            if sim > 1:
                sim = 1.0
            sims.append(sim)

        weights = self._softmax(sims, temperature=self.weight_temperature)
        enriched = []
        for item, sim, w in zip(results, sims, weights):
            new_item = dict(item)
            new_item["similarity"] = float(sim)
            new_item["weight"] = float(w)
            enriched.append(new_item)

        # 按相似度降序排序
        enriched.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        self._last_ranked = enriched[: self.top_k]
        return self._last_ranked
        
    def _setup_chains(self):
        """设置处理链"""
        # 创建查询处理链
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", NAIVE_PROMPT),
            ("human", """
                ---文档片段--- 
                {context}
                
                问题：
                {query}
                """),
        ])
        
        # 链接到LLM
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()
    
    def _load_faiss_index(self) -> bool:
        """
        加载FAISS索引
        
        Returns:
            bool: 是否成功加载
        """
        # 如果faiss_cache_dir是空字符串，跳过FAISS加载（强制使用Neo4j）
        if self.faiss_cache_dir == "":
            print("跳过FAISS索引加载（使用Neo4j模式）")
            self.faiss_index = None
            self.index_loaded = False
            return False
            
        try:
            print(f"尝试从 {self.faiss_cache_dir} 加载FAISS索引...")
            self.faiss_index = load_faiss_index(self.faiss_cache_dir)
            
            if self.faiss_index:
                self.index_loaded = True
                print("✓ FAISS索引加载成功")
                return True
            else:
                print("✗ FAISS索引加载失败")
                return False
                
        except Exception as e:
            print(f"加载FAISS索引时出错: {str(e)}")
            self.faiss_index = None
            self.index_loaded = False
            return False
    
    def _ensure_index_loaded(self) -> bool:
        """
        确保FAISS索引已加载
        
        Returns:
            bool: 索引是否可用
        """
        if not self.index_loaded or self.faiss_index is None:
            print("FAISS索引未加载，尝试重新加载...")
            return self._load_faiss_index()
        return True
    
    def _search_with_faiss(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        使用FAISS进行向量搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表，包含content和metadata
        """
        if not self._ensure_index_loaded():
            raise RuntimeError("FAISS索引不可用")
        
        k = k or self.top_k
        
        try:
            # 使用FAISS进行相似度搜索
            docs_with_scores = self.faiss_index.similarity_search_with_score(query, k=k*2)  # 获取更多候选
            
            # 过滤低相关性结果并格式化
            results = []
            self._last_faiss_candidates = []
            for doc, score in docs_with_scores:
                # 过滤低相关性结果（注意：FAISS返回的是余弦距离，越小越相似）
                if score > self.similarity_threshold:
                    continue
                # 记录原始候选（诊断用途）
                try:
                    self._last_faiss_candidates.append({
                        "id": doc.metadata.get("chunk_id", "unknown"),
                        "filename": doc.metadata.get("filename", "unknown"),
                        "distance": float(score)
                    })
                except Exception:
                    pass
                
                result = {
                    "id": doc.metadata.get("chunk_id", "unknown"),
                    "text": doc.page_content,
                    "filename": doc.metadata.get("filename", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "score": float(score),
                    "metadata": doc.metadata
                }
                results.append(result)
                
                # 达到所需数量就停止
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            print(f"FAISS搜索失败: {str(e)}")
            raise
    
    def _fallback_to_neo4j(self, query: str) -> List[Dict[str, Any]]:
        """
        回退到Neo4j搜索（保持原有逻辑作为备份）
        
        Args:
            query: 查询文本
            
        Returns:
            List[Dict]: 搜索结果
        """
        print("回退到Neo4j搜索...")
        try:
            # 生成查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 获取带embedding的Chunk节点
            chunks_with_embedding = self.graph.query("""
            MATCH (c:__Chunk__)
            WHERE c.embedding IS NOT NULL
            RETURN c.id AS id, c.text AS text, c.embedding AS embedding
            LIMIT 100  // 获取候选集
            """)
            
            if not chunks_with_embedding:
                return []
            
            # 使用工具类对候选集进行排序
            from search.utils import VectorUtils
            scored_chunks = VectorUtils.rank_by_similarity(
                query_embedding,
                chunks_with_embedding,
                "embedding",
                self.top_k
            )
            
            # 格式化结果
            results = []
            for item in scored_chunks[:self.top_k]:
                result = {
                    "id": item.get("id", "unknown"),
                    "text": item.get("text", ""),
                    "filename": "neo4j_chunk",  # Neo4j中可能没有文件名信息
                    "chunk_index": 0,
                    "score": item.get("score", 0.0),
                    "metadata": {"source": "neo4j"}
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Neo4j回退搜索也失败: {str(e)}")
            return []
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词（naive rag不需要复杂的关键词提取）
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 空的关键词字典
        """
        return {"low_level": [], "high_level": []}
    
    def search(self, query_input: Any) -> str:
        """
        执行Naive RAG搜索 - 使用FAISS向量搜索
        
        参数:
            query_input: 用户查询或包含查询的字典
            
        返回:
            str: 基于检索结果生成的回答
        """
        overall_start = time.time()
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        if not query or not query.strip():
            return "请提供有效的查询内容。\n\n{'data': {'Chunks':[] } }"
        
        # 检查缓存
        cache_key = f"naive_faiss:{query}"  # 修改缓存键以区分FAISS版本
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            print(f"缓存命中: {query[:30]}...")
            return cached_result
        
        try:
            # 使用FAISS进行搜索
            search_start = time.time()
            
            try:
                results = self._search_with_faiss(query, self.top_k)
            except Exception as faiss_error:
                print(f"FAISS搜索失败: {faiss_error}")
                # 回退到Neo4j（如果可用）
                if hasattr(self, 'graph') and self.graph:
                    results = self._fallback_to_neo4j(query)
                else:
                    raise faiss_error
            
            search_time = time.time() - search_start
            self.performance_metrics["query_time"] = search_time
            
            if not results:
                no_result_msg = f"没有找到与'{query}'相关的信息。\n\n{{'data': {{'Chunks':[] }} }}"
                self.cache_manager.set(cache_key, no_result_msg)
                return no_result_msg
            
            # 重排序与加权
            ranked = self._rerank_and_weight(query, results)

            # 按权重降序格式化检索到的文档片段，并按权重分配上下文预算
            chunks_content = []
            chunk_ids = []
            total_weight = sum([float(x.get("weight", 0.0)) for x in ranked]) or 1.0
            for i, item in enumerate(ranked):
                chunk_id = item.get("id", f"chunk_{i}")
                text = item.get("text", "")
                filename = item.get("filename", "unknown")
                weight = item.get("weight", 0.0)
                display_score = item.get("rerank_score", item.get("similarity", None))

                if text:
                    # 按权重分配可用字符数（至少保留一小段以便模型参考）
                    alloc = max(200, int((float(weight) / total_weight) * self.context_max_chars))
                    snippet = text if len(text) <= alloc else (text[:alloc] + "...")
                    chunk_info = f"Chunk ID: {chunk_id}\n"
                    chunk_info += f"来源文件: {filename}\n"
                    if display_score is not None:
                        chunk_info += f"相似度/分数: {float(display_score):.4f}\n"
                    chunk_info += f"权重: {float(weight):.4f}\n"
                    chunk_info += f"内容: {snippet}"

                    chunks_content.append(chunk_info)
                    chunk_ids.append(chunk_id)
            
            context = "\n\n---\n\n".join(chunks_content)
            
            # 生成回答
            llm_start = time.time()
            
            answer = self.query_chain.invoke({
                "query": query,
                "context": context,
                "response_type": response_type
            })
            
            llm_time = time.time() - llm_start
            self.performance_metrics["llm_time"] = llm_time
            
            # 确保回答中包含Chunk ID引用
            if "{'data': {'Chunks':" not in answer and "{'data': {\"Chunks\":" not in answer:
                # 添加引用信息
                chunk_references = ", ".join([f"'{id}'" for id in chunk_ids[:5]])  # 最多引用5个
                answer += f"\n\n{{'data': {{'Chunks':[{chunk_references}] }} }}"
            
            # 追加检索诊断信息（可选）
            if self.show_diagnostics:
                try:
                    diag_lines = []
                    diag_lines.append("\n---\n")
                    diag_lines.append("=== 检索诊断 ===")
                    diag_lines.append(f"Reranker启用: {self.enable_reranker}")
                    if self.enable_reranker:
                        diag_lines.append(f"Reranker模型: {self.reranker_model_name}")
                    diag_lines.append(f"权重温度: {self.weight_temperature}")
                    diag_lines.append(f"上下文预算(字符): {self.context_max_chars}")

                    if self._last_faiss_candidates:
                        diag_lines.append("\n- FAISS原始候选(前k):")
                        for c in self._last_faiss_candidates[: self.top_k]:
                            fid = c.get('id','?')
                            dist = float(c.get('distance', 0.0)) if 'distance' in c else 0.0
                            fn = c.get('filename','unknown')
                            diag_lines.append(f"  • {fid} | 距离: {dist:.4f} | 文件: {fn}")

                    if self._last_ranked:
                        diag_lines.append("\n- 重排序后(前k):")
                        for r in self._last_ranked[: self.top_k]:
                            rid = r.get('id','?')
                            wt = float(r.get('weight', 0.0))
                            rr = r.get('rerank_score', None)
                            sim = r.get('similarity', None)
                            if rr is not None:
                                diag_lines.append(f"  • {rid} | 重排分: {float(rr):.4f} | 权重: {wt:.4f}")
                            else:
                                diag_lines.append(f"  • {rid} | 相似度: {float(sim or 0.0):.4f} | 权重: {wt:.4f}")

                    answer += "\n" + "\n".join(diag_lines)
                except Exception:
                    pass
            
            # 缓存结果
            self.cache_manager.set(cache_key, answer)
            
            # 记录总耗时
            total_time = time.time() - overall_start
            self.performance_metrics["total_time"] = total_time
            
            # 输出性能统计
            print(f"搜索完成 - 查询时间: {search_time:.4f}s, LLM时间: {llm_time:.4f}s, 总时间: {total_time:.4f}s")
            print(f"返回 {len(results)} 个相关结果")
            
            return answer
            
        except Exception as e:
            error_msg = f"搜索过程中出现错误: {str(e)}"
            print(error_msg)
            # 不缓存错误结果
            return f"搜索过程中出错: {str(e)}\n\n{{'data': {{'Chunks':[] }} }}"
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        获取搜索统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = {
            "faiss_index_loaded": self.index_loaded,
            "faiss_cache_dir": self.faiss_cache_dir,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "performance_metrics": self.performance_metrics.copy()
        }
        
        # 如果索引已加载，添加索引信息
        if self.faiss_index:
            try:
                # 尝试获取索引中的文档数量
                test_docs = self.faiss_index.similarity_search("test", k=1)
                stats["index_document_count"] = "available"
            except:
                stats["index_document_count"] = "unknown"
        
        return stats
    
    def reload_index(self) -> bool:
        """
        重新加载FAISS索引
        
        Returns:
            bool: 是否成功重新加载
        """
        print("强制重新加载FAISS索引...")
        self.faiss_index = None
        self.index_loaded = False
        return self._load_faiss_index()
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具
        
        返回:
            BaseTool: 搜索工具实例
        """
        class NaiveRetrievalTool(BaseTool):
            name: str = "naive_retriever"
            description: str = naive_description
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return NaiveRetrievalTool()
    
    def close(self):
        """关闭资源"""
        if self.faiss_index:
            # FAISS索引不需要显式关闭，但可以清理引用
            self.faiss_index = None
        self.index_loaded = False
        super().close()