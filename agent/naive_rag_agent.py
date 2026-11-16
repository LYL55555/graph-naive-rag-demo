from typing import List, Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import asyncio

from config.prompt import NAIVE_PROMPT
from config.settings import response_type
from search.tool.naive_search_tool import NaiveSearchTool
from agent.base import BaseAgent


class NaiveRagAgent(BaseAgent):
    """使用FAISS向量检索的Naive RAG Agent实现"""
    
    def __init__(self, faiss_cache_dir: str = None):
        """
        初始化Naive RAG Agent
        
        Args:
            faiss_cache_dir: FAISS索引缓存目录，确保与构建时路径一致
        """
        # 保存FAISS缓存路径
        self.faiss_cache_dir = faiss_cache_dir
        
        # 初始化FAISS-based Naive搜索工具
        self.search_tool = NaiveSearchTool(faiss_cache_dir=faiss_cache_dir)
        
        # 设置Agent缓存目录（与FAISS索引分开）
        self.cache_dir = "./cache/naive_agent"
        
        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)
        
        # 验证FAISS索引是否正确加载
        self._verify_faiss_setup()
    
    def _verify_faiss_setup(self):
        """验证FAISS索引设置是否正确"""
        try:
            stats = self.search_tool.get_search_stats()
            if stats['faiss_index_loaded']:
                print(f"✅ NaiveRagAgent: FAISS索引加载成功")
                print(f"   索引路径: {stats['faiss_cache_dir']}")
                print(f"   检索参数: top_k={stats['top_k']}, threshold={stats['similarity_threshold']}")
            else:
                print(f"⚠️ NaiveRagAgent: FAISS索引加载失败")
                print(f"   索引路径: {stats['faiss_cache_dir']}")
                print(f"   建议: 请先运行 'python -c \"from processor.faiss_indexer import build_faiss_index; build_faiss_index()\"'")
        except Exception as e:
            print(f"❌ NaiveRagAgent: FAISS验证失败 - {str(e)}")
    
    def get_agent_info(self) -> Dict:
        """获取Agent信息"""
        base_info = {
            "agent_type": "NaiveRagAgent",
            "agent_description": "使用FAISS向量检索的Naive RAG Agent",
            "faiss_cache_dir": self.faiss_cache_dir,
            "agent_cache_dir": self.cache_dir
        }
        
        # 添加搜索工具状态
        try:
            search_stats = self.search_tool.get_search_stats()
            base_info.update({
                "faiss_index_loaded": search_stats['faiss_index_loaded'],
                "search_params": {
                    "top_k": search_stats['top_k'],
                    "similarity_threshold": search_stats['similarity_threshold']
                }
            })
        except Exception as e:
            base_info["search_tool_error"] = str(e)
        
        return base_info
    
    def invoke(self, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        LangGraph标准的invoke方法
        
        Args:
            inputs: 输入字典，应包含 'messages' 键
            config: 可选配置字典
            
        Returns:
            Dict[str, Any]: 包含 'messages' 键的响应字典
        """
        try:
            # 提取消息
            messages = inputs.get("messages", [])
            if not messages:
                raise ValueError("输入必须包含 'messages' 键且不为空")
            
            # 获取最后一条消息的内容作为查询
            last_message = messages[-1]
            query = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # 设置默认配置
            if config is None:
                config = {}
            
            # 构建LangGraph配置
            graph_config = {
                "configurable": {
                    "thread_id": config.get("configurable", {}).get("thread_id", "default"),
                    "recursion_limit": config.get("configurable", {}).get("recursion_limit", 5)
                }
            }
            
            # 使用LangGraph执行
            final_state = None
            for output in self.graph.stream(inputs, config=graph_config):
                final_state = output
                
            # 从memory中获取最终状态
            if hasattr(self, 'memory') and self.memory:
                try:
                    chat_history = self.memory.get(graph_config)["channel_values"]["messages"]
                    return {"messages": chat_history}
                except Exception:
                    pass
            
            # 如果无法从memory获取，尝试从最后的输出获取
            if final_state:
                # 查找包含 'messages' 的输出
                for node_name, node_output in final_state.items():
                    if isinstance(node_output, dict) and "messages" in node_output:
                        return {"messages": messages + node_output["messages"]}
            
            # 最后的备用方案：使用ask方法
            result = self.ask(query, thread_id=graph_config["configurable"]["thread_id"])
            
            from langchain_core.messages import AIMessage
            return {"messages": messages + [AIMessage(content=result)]}
            
        except Exception as e:
            # 错误处理
            from langchain_core.messages import AIMessage
            error_msg = f"抱歉，处理您的请求时出现错误: {str(e)}"
            return {"messages": inputs.get("messages", []) + [AIMessage(content=error_msg)]}

    def _setup_tools(self) -> List:
        """设置工具"""
        return [
            self.search_tool.get_tool(),
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边"""
        # 简单的从检索直接到生成，无需复杂路由
        workflow.add_edge("retrieve", "generate")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        提取查询关键词 - 简化版本，不做实际的关键词提取
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含低级和高级关键词（空列表）
        """
        # Naive实现不需要关键词提取
        return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """生成回答节点逻辑"""
        messages = state["messages"]
        
        # 安全地获取问题和检索结果
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
            
        try:
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            docs = "无法获取检索结果"

        # 首先尝试全局缓存
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "全局缓存命中")
            return {"messages": [AIMessage(content=global_result)]}

        # 获取当前会话ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 然后检查会话缓存
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "会话缓存命中")
            # 将命中内容同步到全局缓存
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
        ("system", NAIVE_PROMPT),
        ("human", """
            ---检索结果--- 
            {context}
            
            问题：
            {question}
            """),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()
        try:
            response = rag_chain.invoke({
                "context": docs, 
                "question": question, 
                "response_type": response_type
            })
            
            # 缓存结果 - 同时更新会话缓存和全局缓存
            if response and len(response) > 10:
                # 更新会话缓存
                self.cache_manager.set(question, response, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(question, response)
            
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            response)
            
            return {"messages": [AIMessage(content=response)]}
        except Exception as e:
            error_msg = f"生成回答时出错: {str(e)}"
            self._log_execution("generate_error", 
                            {"question": question, "docs_length": len(docs)}, 
                            error_msg)
            return {"messages": [AIMessage(content=f"抱歉，我无法回答这个问题。技术原因: {str(e)}")]}
    
    async def _stream_process(self, inputs, config):
        """实现流式处理过程"""
        # 获取会话信息
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        query = inputs["messages"][-1].content
        
        # 开始处理提示
        yield "**开始处理查询**...\n\n"
        
        try:
            # 执行Naive搜索
            search_result = self.search_tool.search(query)
            
            # 分块返回结果
            if search_result:
                yield "**已找到相关信息，正在生成回答**...\n\n"
                
                # 分块返回
                sentences = re.split(r'([.!?。！？]\s*)', search_result)
                buffer = ""
                
                for i in range(0, len(sentences)):
                    buffer += sentences[i]
                    
                    # 当缓冲区包含完整句子或达到合理大小时输出
                    if (i % 2 == 1) or len(buffer) >= 40:
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.01)
                
                # 输出任何剩余内容
                if buffer:
                    yield buffer
            else:
                yield "未找到与您问题相关的信息。请尝试更换关键词或提供更多细节。"
        
        except Exception as e:
            yield f"**处理查询时出错**: {str(e)}"
    
    def close(self):
        """关闭资源"""
        # 先关闭父类资源
        super().close()
        
        # 再关闭搜索工具资源
        if self.search_tool:
            self.search_tool.close()