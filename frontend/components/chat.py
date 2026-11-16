import streamlit as st
import uuid
import re
import json
import ast
import traceback
from utils.api import send_message, send_feedback, get_source_content, get_knowledge_graph_from_message, get_source_file_info_batch, clear_chat, send_message_stream
from utils.helpers import extract_source_ids

def reset_processing_lock():
    """Reset processing lock status"""
    st.session_state.processing_lock = False

def display_chat_interface():
    """Display main chat interface"""
    st.title("GraphRAG Chat System")
    
    # 确保锁变量存在并设初始值，如果不存在的话
    if "processing_lock" not in st.session_state:
        st.session_state.processing_lock = False
    
    # 设置栏（两行：第一行 三列；第二行（elastic_agent）横向一排 ES 参数控件）
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # 添加重置锁的功能到 selectbox 的 on_change 参数
            previous_agent = st.session_state.agent_type
            agent_type = st.selectbox(
                "Select Agent Type",
                options=["graph_agent", "hybrid_agent", "naive_rag_agent", "deep_research_agent", "fusion_agent", "elastic_agent"],
                key="header_agent_type",
                help="Select different agents to experience different retrieval strategies",
                index=(
                    ["graph_agent", "hybrid_agent", "naive_rag_agent", "deep_research_agent", "fusion_agent", "elastic_agent"].index(st.session_state.agent_type)
                    if st.session_state.agent_type in ["graph_agent", "hybrid_agent", "naive_rag_agent", "deep_research_agent", "fusion_agent", "elastic_agent"]
                    else 0
                ),
                on_change=reset_processing_lock
            )
            
            # 检查是否切换了agent类型
            if previous_agent != agent_type:
                # 切换agent类型时重置锁
                st.session_state.processing_lock = False
                
            st.session_state.agent_type = agent_type
            
            # 添加思考过程切换 - 仅当选择 deep_research_agent 时显示
            if agent_type == "deep_research_agent":
                show_thinking = st.checkbox("Show Reasoning Process", 
                          value=st.session_state.get("show_thinking", False),
                          key="header_show_thinking",
                          help="Show AI's thinking process",
                          on_change=reset_processing_lock)
                st.session_state.show_thinking = show_thinking

                use_deeper = st.checkbox("Use Enhanced Research Tool", 
                                        value=st.session_state.get("use_deeper_tool", True),
                                        key="header_use_deeper",
                                        help="Enable community-aware and knowledge graph enhancement",
                                        on_change=reset_processing_lock)
                st.session_state.use_deeper_tool = use_deeper
    
        with col2:
            # 添加流式响应选项 - 仅当调试模式未启用时显示
            if not st.session_state.debug_mode:
                use_stream = st.checkbox("Use Streaming Response", 
                                        value=st.session_state.get("use_stream", True),
                                        key="header_use_stream",
                                        help="Enable streaming response to display results in real-time",
                                        on_change=reset_processing_lock)
                st.session_state.use_stream = use_stream
            else:
                # 在调试模式下自动禁用流式响应
                st.session_state.use_stream = False
                st.info("Streaming response is disabled in debug mode")
            # elastic_agent doesn't need streaming response, force close for block rendering
            if agent_type == "elastic_agent":
                st.session_state.use_stream = False
                st.caption("elastic_agent selected: streaming disabled to show highlighted results")
            
        with col3:
            # Modified clear chat button with lock reset functionality
            st.button("🗑️ Clear Chat", key="header_clear_chat", on_click=clear_chat_with_lock_reset)

        # 第二行：仅 elastic_agent 时显示，横向一排 ES 参数控件
        if agent_type == "elastic_agent":
            st.markdown("")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                es_mode = st.radio(
                    "Mode",
                    ["any", "all"],
                    index=(
                        ["any", "all"].index(st.session_state.get("es_mode", "any"))
                        if st.session_state.get("es_mode") in ["any", "all"] else 0
                    ),
                    key="header_es_mode",
                    help="any: OR; all: exact phrase/adjacent phrase matching (depends on whether text.raw exists)",
                    horizontal=True,
                )
                st.session_state.es_mode = es_mode
            with c2:
                es_page = st.number_input(
                    "Page",
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get("es_page", 1) or 1),
                    key="header_es_page",
                )
                st.session_state.es_page = int(es_page)
            with c3:
                es_size = st.number_input(
                    "Per Page (≤50)",
                    min_value=1,
                    max_value=50,
                    step=1,
                    value=min(int(st.session_state.get("es_size", 20) or 20), 50),
                    key="header_es_size",
                )
                st.session_state.es_size = int(es_size)
    
    # 分隔线
    st.markdown("---")
    
    # If there's a request being processed, show warning
    if st.session_state.processing_lock:
        st.warning("Please wait for the current operation to complete...")
        # Add button to force reset lock
        if st.button("Force Reset Processing Status", key="force_reset_lock"):
            st.session_state.processing_lock = False
            st.rerun()
    
    # 聊天区域
    chat_container = st.container()
    with chat_container:
        # 显示现有消息
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                # 获取要显示的内容
                content = msg["content"]
                
                # 处理deep_research_agent的思考过程
                if msg["role"] == "assistant":
                    # elastic_agent：优先渲染 ES 高亮结果
                    es_rendered = False
                    if st.session_state.get("agent_type") == "elastic_agent":
                        try:
                            parsed = None
                            try:
                                parsed = json.loads(content)
                            except Exception:
                                parsed = ast.literal_eval(content) if isinstance(content, str) else None
                            if isinstance(parsed, dict) and isinstance(parsed.get("data", {}).get("Chunks", []), list):
                                es_rendered = True
                                chunks = parsed["data"]["Chunks"]
                                diags = parsed.get("diagnostics", {})
                                st.markdown(f"Search Results: {len(chunks)} items | Mode: {diags.get('mode','')} | Index: {diags.get('index','')} | took: {diags.get('es_took_ms','-')} ms")
                                max_show = min(len(chunks), 20)
                                for j in range(max_show):
                                    ch = chunks[j]
                                    st.markdown(ch.get("highlighted", ""), unsafe_allow_html=True)
                                    st.caption(f"{ch.get('filename','')}  # {ch.get('chunk_index',0)} | score={ch.get('score',0)} | id={ch.get('id','')}")
                                if len(chunks) > max_show:
                                    st.info(f"Showing first {max_show} items, adjust page/size for pagination.")
                        except Exception:
                            es_rendered = False
                    if not es_rendered:
                        # 判断是否需要显示思考过程
                        show_thinking = (st.session_state.agent_type == "deep_research_agent" and 
                                        st.session_state.get("show_thinking", False))
                        
                        # 优先使用raw_thinking字段
                        if "raw_thinking" in msg and show_thinking:
                            # 提取思考过程
                            thinking_process = msg["raw_thinking"]
                            answer_content = msg.get("processed_content", content)
                            
                            # 格式化思考过程，使用引用格式
                            thinking_lines = thinking_process.split('\n')
                            quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                            
                            # 显示思考过程
                            st.markdown(quoted_thinking)
                            
                            # 添加两行空行间隔
                            st.markdown("\n\n")
                            
                            # 显示答案
                            st.markdown(answer_content)
                        # 检查是否有<think>标签
                        elif "<think>" in content and "</think>" in content:
                            # 提取<think>标签中的内容
                            thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                            
                            if thinking_match:
                                thinking_process = thinking_match.group(1)
                                # 移除思考过程，保留答案
                                answer_content = content.replace(f"<think>{thinking_process}</think>", "").strip()
                                
                                if show_thinking:
                                    # 显示思考过程（仅当show_thinking为True时）
                                    # 格式化思考过程，使用引用格式
                                    thinking_lines = thinking_process.split('\n')
                                    quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                                    
                                    # 显示思考过程
                                    st.markdown(quoted_thinking)
                                    
                                    # 添加两行空行间隔
                                    st.markdown("\n\n")
                                    
                                    # 显示答案
                                    st.markdown(answer_content)
                                else:
                                    # 只显示答案部分（不显示思考过程）
                                    st.markdown(answer_content)
                            else:
                                # 如果提取失败，显示完整内容但移除可能的<think>标签
                                cleaned_content = re.sub(r'<think>|</think>', '', content)
                                st.markdown(cleaned_content)
                        else:
                            # 普通回答，无思考过程
                            st.markdown(content)
                else:
                    # 普通消息直接显示
                    st.markdown(content)
                
                # 为AI回答添加反馈按钮和源内容引用
                if msg["role"] == "assistant":
                    # 生成一个唯一的消息ID (如果之前没有)
                    if "message_id" not in msg:
                        msg["message_id"] = str(uuid.uuid4())
                        
                    # 查找对应的用户问题
                    user_query = ""
                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                        user_query = st.session_state.messages[i-1]["content"]
                        
                    # 检查是否已经提供过反馈
                    feedback_key = f"{msg['message_id']}"
                    feedback_type_key = f"feedback_type_{feedback_key}"
                    
                    # 创建一个容器用于显示反馈结果
                    feedback_container = st.empty()
                    
                    if feedback_key not in st.session_state.feedback_given:
                        # 添加反馈按钮
                        col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
                        
                        with col1:
                            thumbs_up_key = f"thumbs_up_{msg['message_id']}_{i}"
                            if st.button("👍", key=thumbs_up_key):
                                # 检查是否有正在处理的请求
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False
                                
                                if st.session_state.feedback_in_progress:
                                    with feedback_container:
                                        st.warning("Please wait for the current operation to complete...")
                                else:
                                    st.session_state.feedback_in_progress = True
                                    try:
                                        with feedback_container:
                                            with st.spinner("Submitting feedback..."):
                                                response = send_feedback(
                                                    msg["message_id"], 
                                                    user_query, 
                                                    True, 
                                                    st.session_state.session_id,
                                                    st.session_state.agent_type
                                                )
                                        
                                        st.session_state.feedback_given.add(feedback_key)
                                        st.session_state[feedback_type_key] = "positive"
                                        
                                        # Display different messages based on response
                                        with feedback_container:
                                            if response and "action" in response:
                                                if "high quality" in response["action"] or "高质量" in response["action"]:
                                                    st.success("Thank you for your feedback! This answer has been marked as high quality.", icon="🙂")
                                                else:
                                                    st.success("Thank you for your feedback!", icon="👍")
                                            else:
                                                st.info("Feedback received.", icon="ℹ️")
                                    except Exception as e:
                                        st.error(f"Error submitting feedback: {str(e)}")
                                    finally:                                           
                                        st.session_state.feedback_in_progress = False
                                    
                        with col2:
                            thumbs_down_key = f"thumbs_down_{msg['message_id']}_{i}"
                            if st.button("👎", key=thumbs_down_key):
                                # 检查是否有正在处理的请求
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False
                                
                                if st.session_state.feedback_in_progress:
                                    with feedback_container:
                                        st.warning("Please wait for the current operation to complete...")
                                else:
                                    st.session_state.feedback_in_progress = True
                                    try:
                                        with feedback_container:
                                            with st.spinner("Submitting feedback..."):
                                                response = send_feedback(
                                                    msg["message_id"], 
                                                    user_query, 
                                                    False, 
                                                    st.session_state.session_id,
                                                    st.session_state.agent_type
                                                )
                                        
                                        st.session_state.feedback_given.add(feedback_key)
                                        st.session_state[feedback_type_key] = "negative"
                                        
                                        # Display different messages based on response
                                        with feedback_container:
                                            if response and "action" in response:
                                                if "clear" in response["action"] or "清除" in response["action"]:
                                                    st.error("Feedback received. This answer will no longer be used.", icon="🔄")
                                                else:
                                                    st.error("Feedback received. We will improve.", icon="👎")
                                            else:
                                                st.info("Feedback received.", icon="ℹ️")
                                    except Exception as e:
                                        st.error(f"Error submitting feedback: {str(e)}")
                                    finally:
                                        st.session_state.feedback_in_progress = False
                    else:
                        # Display feedback type already provided
                        feedback_type = st.session_state.get(feedback_type_key, None)
                        with feedback_container:
                            if feedback_type == "positive":
                                st.success("You have given positive feedback for this answer!", icon="👍")
                            elif feedback_type == "negative":
                                st.error("You have provided improvement suggestions for this answer.", icon="👎")
                            else:
                                st.info("Feedback received.", icon="ℹ️")
                
                    # If it's an AI answer with source content references, show view source content button
                    if st.session_state.debug_mode and st.session_state.agent_type != "deep_research_agent":
                        source_ids = extract_source_ids(msg["content"])
                        if source_ids:
                            with st.expander("View Source Text", expanded=False):
                                # Get all source file information
                                source_infos = get_source_file_info_batch(source_ids)
                                
                                for s_idx, source_id in enumerate(source_ids):
                                    # Use cached information
                                    display_name = source_infos.get(source_id, {}).get("file_name", f"Source {source_id}")
                                    source_btn_key = f"src_{source_id}_{i}_{s_idx}"
                                    
                                    if st.button(f"Load {display_name}", key=source_btn_key):
                                        with st.spinner(f"Loading source text {display_name}..."):
                                            source_data = get_source_content(source_id)
                                            if source_data and "content" in source_data:
                                                st.session_state.source_content = source_data["content"]
                                                st.session_state.current_tab = "Source Content"
                                                st.rerun()
                        
                        # If it's the last AI message, add automatic knowledge graph extraction button - disabled for deep_research_agent
                        if st.session_state.agent_type != "deep_research_agent":
                            extract_kg_key = f"extract_kg_{i}"
                            if st.button("Extract Knowledge Graph", key=extract_kg_key):
                                with st.spinner("Extracting knowledge graph data..."):
                                    # Get corresponding user query
                                    user_query = ""
                                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                                        user_query = st.session_state.messages[i-1]["content"]
                                        
                                    # Use user query to filter knowledge graph
                                    kg_data = get_knowledge_graph_from_message(msg["content"], user_query)
                                    if kg_data and len(kg_data.get("nodes", [])) > 0:
                                        # Ensure current message has correct kg_data
                                        st.session_state.messages[i]["kg_data"] = kg_data
                                        # Update current graph message index to current processing message index
                                        st.session_state.current_kg_message = i
                                        st.session_state.current_tab = "Knowledge Graph"  # Auto switch to knowledge graph tab
                                        st.rerun()
        
        # Handle new messages
        if prompt := st.chat_input("Please enter your question...", key="chat_input"):
            # Check if there's a request being processed
            if "processing_lock" not in st.session_state:
                st.session_state.processing_lock = False
                
            if st.session_state.processing_lock:
                st.warning("Please wait for the current operation to complete...")
                return
                
            st.session_state.processing_lock = True
            
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                try:
                    # 初始化流式响应的占位符
                    message_placeholder = st.empty()
                    full_response = ""
                    thinking_content = ""
                    
                    # 检查流式响应是否启用 (当调试模式禁用时)
                    use_stream = st.session_state.get("use_stream", True) and not st.session_state.debug_mode
                    
                    if use_stream:
                        # 定义令牌处理器
                        def handle_token(token, is_thinking=False):
                            nonlocal full_response, thinking_content
                            try:
                                # 检查token是否是JSON字符串
                                if isinstance(token, str) and token.startswith("{") and token.endswith("}"):
                                    try:
                                        import json
                                        # 尝试解析JSON
                                        json_data = json.loads(token)
                                        if "content" in json_data:
                                            token = json_data["content"]
                                        elif "status" in json_data:
                                            # 跳过状态消息
                                            return
                                    except json.JSONDecodeError as json_error:
                                        # 不是有效的JSON，保持原样
                                        print(f"JSON解析错误: {str(json_error)}")
                                        pass
                                
                                if is_thinking:
                                    # 添加到思考内容
                                    thinking_content += token
                                    # 将思考内容格式化为引用文本
                                    thinking_lines = thinking_content.split('\n')
                                    quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                                    # 在占位符中显示
                                    message_placeholder.markdown(quoted_thinking)
                                else:
                                    # 添加到完整响应
                                    full_response += token
                                    # 在占位符中显示，添加光标模拟打字效果
                                    message_placeholder.markdown(full_response + "▌")
                            except Exception as e:
                                print(f"处理令牌出错: {str(e)}")
                        
                        # 使用流式 API
                        with st.spinner("思考中..."):
                            try:
                                raw_thinking = send_message_stream(prompt, handle_token)
                                # 检查是否有响应
                                if not full_response or full_response.startswith("{") and full_response.endswith("}"):
                                    print("流式响应格式不正确，使用非流式API")
                                    response = send_message(prompt)
                                    if response:
                                        full_response = response.get("answer", "")
                                        message_placeholder.markdown(full_response)
                            except Exception as e:
                                print(f"流式API失败: {str(e)}")
                                response = send_message(prompt)
                                if response:
                                    full_response = response.get("answer", "")
                                    message_placeholder.markdown(full_response)
                        
                        # 最后一次更新，移除光标
                        message_placeholder.markdown(full_response)
                        
                        # 创建消息对象
                        message_obj = {
                            "role": "assistant",
                            "content": full_response,
                            "message_id": str(uuid.uuid4())
                        }
                        
                        # 如果有思考内容，添加到消息中
                        if thinking_content:
                            message_obj["raw_thinking"] = thinking_content
                            message_obj["processed_content"] = full_response
                    else:
                        # 使用非流式 API
                        with st.spinner("思考中..."):
                            # elastic_agent 时，将侧边栏的 es_mode/page/size 透传为 JSON
                            if st.session_state.agent_type == "elastic_agent":
                                es_mode = st.session_state.get("es_mode", "any")
                                es_page = int(st.session_state.get("es_page", 1) or 1)
                                es_size = int(st.session_state.get("es_size", 20) or 20)
                                payload = json.dumps({
                                    "query": prompt,
                                    "mode": es_mode,
                                    "page": es_page,
                                    "size": es_size
                                }, ensure_ascii=False)
                                response = send_message(payload)
                            else:
                                response = send_message(prompt)
                        
                        if response:
                            answer = response.get("answer", "抱歉，我无法处理您的请求。")
                            
                            # 在占位符中显示内容
                            message_placeholder.markdown(answer)
                            
                            # 创建消息对象
                            message_obj = {
                                "role": "assistant", 
                                "content": answer,
                                "message_id": str(uuid.uuid4())
                            }
                            
                            # 如果有思考内容，添加到消息中
                            if "raw_thinking" in response:
                                message_obj["raw_thinking"] = response["raw_thinking"]
                                message_obj["processed_content"] = answer
                                
                            # 添加执行轨迹到消息对象
                            if "execution_log" in response and st.session_state.debug_mode:
                                st.session_state.execution_log = response["execution_log"]
                        else:
                            # 处理响应为空的情况
                            error_message = "抱歉，服务器没有返回有效响应。"
                            message_placeholder.markdown(error_message)
                            message_obj = {
                                "role": "assistant", 
                                "content": error_message,
                                "message_id": str(uuid.uuid4())
                            }
                    
                    # 添加到会话状态
                    st.session_state.messages.append(message_obj)
                        
                    # 从回答中提取知识图谱数据，deep_research_agent禁用此功能
                    if st.session_state.debug_mode and st.session_state.agent_type != "deep_research_agent":
                        with st.spinner("提取知识图谱数据..."):
                            # 获取当前新消息的索引，即最后一条消息
                            current_msg_index = len(st.session_state.messages) - 1
                            
                            # 优先使用后端返回的kg_data
                            kg_data = response.get("kg_data") if not use_stream else None
                            
                            # 如果后端没有返回kg_data，尝试从回答中提取，并传递用户查询
                            if not kg_data or len(kg_data.get("nodes", [])) == 0:
                                answer_content = message_obj["content"]
                                kg_data = get_knowledge_graph_from_message(answer_content, prompt)
                            
                            if kg_data and len(kg_data.get("nodes", [])) > 0:
                                # 更新该消息的kg_data
                                st.session_state.messages[current_msg_index]["kg_data"] = kg_data
                                
                                # 更新当前处理的图谱消息索引为最新消息的索引
                                st.session_state.current_kg_message = current_msg_index
                                
                                # 自动切换到知识图谱标签
                                st.session_state.current_tab = "知识图谱"
                                st.rerun()
                            else:
                                if st.session_state.agent_type != "deep_research_agent":
                                    st.warning("无法提取知识图谱数据")
                except Exception as e:
                    st.error(f"处理消息时出错: {str(e)}")
                    traceback.print_exc()
                finally:
                    # 确保请求处理完成后释放锁
                    st.session_state.processing_lock = False
                    
            st.rerun()

def clear_chat_with_lock_reset():
    """清除聊天并重置处理锁"""
    # 重置处理锁
    st.session_state.processing_lock = False
    # 调用原始清除函数
    clear_chat()