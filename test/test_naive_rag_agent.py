#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
更新后的NaiveRagAgent测试脚本
验证FAISS集成后的Agent是否正常工作
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.naive_rag_agent import NaiveRagAgent
from langchain_core.messages import HumanMessage


def test_agent_initialization():
    """测试Agent初始化"""
    print("=== 测试1: Agent初始化 ===")
    
    try:
        # 创建Agent实例
        agent = NaiveRagAgent()
        
        # 获取Agent信息
        info = agent.get_agent_info()
        
        print("Agent信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 检查FAISS索引状态
        if info.get('faiss_index_loaded', False):
            print("✅ Agent初始化成功，FAISS索引已加载")
            return agent, True
        else:
            print("⚠️ Agent初始化成功，但FAISS索引未加载")
            return agent, False
            
    except Exception as e:
        print(f"❌ Agent初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False


def test_agent_invoke():
    """测试Agent.invoke()方法"""
    print("\n=== 测试2: Agent.invoke()方法 ===")
    
    agent, index_loaded = test_agent_initialization()
    
    if not agent:
        print("❌ Agent初始化失败，跳过invoke测试")
        return False
    
    if not index_loaded:
        print("⚠️ FAISS索引未加载，invoke可能失败")
    
    # 测试查询列表
    test_queries = [
        "学生手册有什么内容？",
        "如何申请奖学金？",
        "论文答辩的流程是什么？"
    ]
    
    results = {}
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试查询 {i}: {query} ---")
        
        try:
            # 构建输入消息
            messages = [HumanMessage(content=query)]
            input_data = {"messages": messages}
            
            # 记录开始时间
            start_time = time.time()
            
            # 调用Agent
            response = agent.invoke(input_data)
            
            # 记录结束时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 分析响应
            if response and "messages" in response:
                last_message = response["messages"][-1]
                content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                print(f"✅ 查询成功 (耗时: {elapsed_time:.2f}s)")
                print(f"响应长度: {len(content)} 字符")
                
                # 显示响应预览
                preview_lines = content.split('\n')[:3]
                for line in preview_lines:
                    if line.strip():
                        preview = line[:80] + "..." if len(line) > 80 else line
                        print(f"  {preview}")
                
                # 检查是否包含引用信息
                if "{'data': {'Chunks':" in content or '{"data": {"Chunks":' in content:
                    print("  ✅ 包含chunk引用信息")
                else:
                    print("  ⚠️ 缺少chunk引用信息")
                
                results[query] = {
                    "status": "success",
                    "time": elapsed_time,
                    "length": len(content),
                    "has_references": "{'data': {'Chunks':" in content or '{"data": {"Chunks":' in content
                }
                
            else:
                print("❌ 响应格式异常")
                results[query] = {"status": "error", "error": "Invalid response format"}
                
        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
            results[query] = {"status": "error", "error": str(e)}
    
    # 统计结果
    successful = sum(1 for r in results.values() if r.get("status") == "success")
    total = len(test_queries)
    
    print(f"\n=== 测试结果统计 ===")
    print(f"成功查询: {successful}/{total}")
    
    if successful > 0:
        avg_time = sum(r.get("time", 0) for r in results.values() if r.get("status") == "success") / successful
        print(f"平均响应时间: {avg_time:.2f}s")
        
        with_refs = sum(1 for r in results.values() if r.get("has_references", False))
        print(f"包含引用: {with_refs}/{successful}")
    
    # 清理资源
    try:
        agent.close()
        print("✅ Agent资源清理完成")
    except Exception as e:
        print(f"⚠️ Agent清理时出现警告: {str(e)}")
    
    return successful == total


def test_specific_invoke_format():
    """测试特定的invoke格式（按照用户要求）"""
    print("\n=== 测试3: 特定invoke格式 ===")
    
    try:
        agent = NaiveRagAgent()
        
        # 按照用户要求的格式
        test_message = HumanMessage("test query")
        result = agent.invoke({"messages": [test_message]})
        
        print("✅ Agent.invoke({\"messages\": [HumanMessage(\"test query\")]}) 成功")
        print(f"响应类型: {type(result)}")
        
        if isinstance(result, dict) and "messages" in result:
            print(f"响应消息数量: {len(result['messages'])}")
            last_msg = result["messages"][-1]
            print(f"最后消息类型: {type(last_msg)}")
            print(f"最后消息内容长度: {len(last_msg.content)}")
            
            # 显示部分内容
            preview = last_msg.content[:150] + "..." if len(last_msg.content) > 150 else last_msg.content
            print(f"内容预览: {preview}")
            
            return True
        else:
            print("❌ 响应格式不符合预期")
            return False
            
    except Exception as e:
        print(f"❌ 特定格式测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_compatibility():
    """测试LangGraph工作流兼容性"""
    print("\n=== 测试4: LangGraph工作流兼容性 ===")
    
    try:
        agent = NaiveRagAgent()
        
        # 检查工作流图是否正确构建
        if hasattr(agent, 'graph') and agent.graph:
            print("✅ LangGraph工作流图构建成功")
            
            # 检查节点
            nodes = getattr(agent.graph, 'nodes', {})
            print(f"工作流节点: {list(nodes.keys()) if hasattr(nodes, 'keys') else 'Unknown'}")
            
            # 检查工具
            tools = agent._setup_tools()
            print(f"配置的工具数量: {len(tools)}")
            
            if tools:
                tool = tools[0]
                print(f"工具名称: {tool.name}")
                print(f"工具描述: {tool.description[:50]}...")
            
            return True
        else:
            print("❌ LangGraph工作流图未正确构建")
            return False
            
    except Exception as e:
        print(f"❌ 工作流兼容性测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("开始更新后的NaiveRagAgent测试")
    print("=" * 60)
    
    test_results = []
    
    try:
        # 测试1: Agent初始化
        _, init_success = test_agent_initialization()
        test_results.append(("Agent初始化", init_success))
        
        # 测试2: Agent invoke方法
        invoke_success = test_agent_invoke()
        test_results.append(("Agent invoke", invoke_success))
        
        # 测试3: 特定invoke格式
        specific_success = test_specific_invoke_format()
        test_results.append(("特定invoke格式", specific_success))
        
        # 测试4: 工作流兼容性
        workflow_success = test_workflow_compatibility()
        test_results.append(("LangGraph工作流", workflow_success))
        
    except Exception as e:
        print(f"测试过程中出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 输出测试结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！NaiveRagAgent FAISS集成完全成功！")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关问题。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 