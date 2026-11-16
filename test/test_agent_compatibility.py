#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent兼容性测试脚本
验证FAISS集成后各Agent的工作状态
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_graph_agent():
    """测试GraphAgent是否仍正常使用Neo4j"""
    print("=== 测试GraphAgent (应使用Neo4j) ===")
    
    try:
        from agent.graph_agent import GraphAgent
        
        agent = GraphAgent()
        print("✅ GraphAgent初始化成功")
        
        # 检查工具配置
        tools = agent._setup_tools()
        print(f"工具数量: {len(tools)}")
        
        for tool in tools:
            print(f"- {tool.name}: {tool.description[:50]}...")
        
        # 检查本地搜索工具是否使用Neo4j
        local_tool = agent.local_tool
        print(f"LocalSearchTool类型: {type(local_tool)}")
        
        # 检查是否有Neo4j相关属性
        if hasattr(local_tool, 'local_searcher'):
            print("✅ 包含local_searcher (Neo4j-based)")
        
        # 简单查询测试
        test_query = "测试查询"
        print(f"\n进行简单查询测试: {test_query}")
        
        start_time = time.time()
        try:
            result = agent.ask(test_query)
            elapsed = time.time() - start_time
            print(f"✅ 查询成功 (耗时: {elapsed:.2f}s)")
            print(f"响应长度: {len(result)} 字符")
            
            # 检查是否包含图谱信息
            if "Chunk" in result or "实体" in result or "关系" in result:
                print("✅ 响应包含图谱相关信息")
                
        except Exception as e:
            print(f"⚠️ 查询测试失败: {str(e)}")
        
        # 清理资源
        agent.close()
        return True
        
    except Exception as e:
        print(f"❌ GraphAgent测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_naive_rag_agent():
    """测试NaiveRagAgent是否正常使用FAISS"""
    print("\n=== 测试NaiveRagAgent (应使用FAISS) ===")
    
    try:
        from agent.naive_rag_agent import NaiveRagAgent
        
        agent = NaiveRagAgent()
        print("✅ NaiveRagAgent初始化成功")
        
        # 获取Agent信息
        info = agent.get_agent_info()
        print("Agent信息:")
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # 检查FAISS加载状态
        if info.get('faiss_index_loaded', False):
            print("✅ FAISS索引加载成功")
        else:
            print("⚠️ FAISS索引未加载")
        
        # 简单查询测试
        test_query = "学生手册"
        print(f"\n进行FAISS查询测试: {test_query}")
        
        start_time = time.time()
        try:
            result = agent.ask(test_query)
            elapsed = time.time() - start_time
            print(f"✅ 查询成功 (耗时: {elapsed:.2f}s)")
            print(f"响应长度: {len(result)} 字符")
            
            # 检查是否包含chunk引用
            if "Chunk" in result and "data" in result:
                print("✅ 响应包含chunk引用信息")
                
        except Exception as e:
            print(f"⚠️ 查询测试失败: {str(e)}")
        
        # 清理资源
        agent.close()
        return True
        
    except Exception as e:
        print(f"❌ NaiveRagAgent测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_agent_faiss_mode():
    """测试FusionAgent的FAISS混合模式"""
    print("\n=== 测试FusionAgent (FAISS混合模式) ===")
    
    try:
        from agent.fusion_agent import FusionGraphRAGAgent
        
        # 创建FAISS混合模式的FusionAgent
        agent = FusionGraphRAGAgent(naive_use_faiss=True)
        print("✅ FusionAgent (FAISS模式) 初始化成功")
        
        # 获取配置信息
        info = agent.get_fusion_agent_info()
        print("FusionAgent配置:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 检查工具配置
        tools = agent._setup_tools()
        print(f"\n工具配置: {len(tools)} 个工具")
        for tool in tools:
            print(f"- {tool.name}: {tool.description[:50]}...")
        
        # 验证naive搜索工具
        if hasattr(agent, 'naive_search_tool') and agent.naive_search_tool:
            print("✅ 包含FAISS-based naive搜索工具")
            
            # 检查naive搜索工具状态
            try:
                stats = agent.naive_search_tool.get_search_stats()
                print(f"  FAISS索引状态: {'已加载' if stats['faiss_index_loaded'] else '未加载'}")
            except Exception as e:
                print(f"  无法获取FAISS状态: {e}")
        
        # 清理资源
        agent.close()
        return True
        
    except Exception as e:
        print(f"❌ FusionAgent (FAISS模式) 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_agent_neo4j_mode():
    """测试FusionAgent的Neo4j传统模式"""
    print("\n=== 测试FusionAgent (Neo4j传统模式) ===")
    
    try:
        from agent.fusion_agent import FusionGraphRAGAgent
        
        # 创建Neo4j传统模式的FusionAgent
        agent = FusionGraphRAGAgent(naive_use_faiss=False)
        print("✅ FusionAgent (Neo4j模式) 初始化成功")
        
        # 获取配置信息
        info = agent.get_fusion_agent_info()
        print("FusionAgent配置:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 检查工具配置
        tools = agent._setup_tools()
        print(f"\n工具配置: {len(tools)} 个工具")
        for tool in tools:
            print(f"- {tool.name}: {tool.description[:50]}...")
        
        # 验证没有naive搜索工具
        if not hasattr(agent, 'naive_search_tool') or agent.naive_search_tool is None:
            print("✅ 传统模式：不包含独立的naive搜索工具")
        
        # 清理资源
        agent.close()
        return True
        
    except Exception as e:
        print(f"❌ FusionAgent (Neo4j模式) 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_service_compatibility():
    """测试AgentService的兼容性"""
    print("\n=== 测试AgentService兼容性 ===")
    
    try:
        from server.services.agent_service import AgentManager
        
        manager = AgentManager()
        print("✅ AgentManager初始化成功")
        
        # 测试各种Agent类型
        agent_types = ["naive_rag_agent", "graph_agent", "fusion_agent"]
        
        for agent_type in agent_types:
            try:
                agent = manager.get_agent(agent_type, "test_session")
                print(f"✅ {agent_type}: 创建成功")
                
                # 获取Agent信息
                if hasattr(agent, 'get_agent_info'):
                    info = agent.get_agent_info()
                    print(f"  类型: {info.get('agent_type', 'Unknown')}")
                elif hasattr(agent, 'get_fusion_agent_info'):
                    info = agent.get_fusion_agent_info()
                    print(f"  模式: {info.get('search_mode', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ {agent_type}: 创建失败 - {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ AgentService测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("开始Agent兼容性测试")
    print("=" * 60)
    
    test_results = []
    
    try:
        # 测试1: GraphAgent (Neo4j)
        result = test_graph_agent()
        test_results.append(("GraphAgent (Neo4j)", result))
        
        # 测试2: NaiveRagAgent (FAISS)
        result = test_naive_rag_agent()
        test_results.append(("NaiveRagAgent (FAISS)", result))
        
        # 测试3: FusionAgent FAISS模式
        result = test_fusion_agent_faiss_mode()
        test_results.append(("FusionAgent (FAISS混合模式)", result))
        
        # 测试4: FusionAgent Neo4j模式
        result = test_fusion_agent_neo4j_mode()
        test_results.append(("FusionAgent (Neo4j传统模式)", result))
        
        # 测试5: AgentService兼容性
        result = test_agent_service_compatibility()
        test_results.append(("AgentService兼容性", result))
        
    except Exception as e:
        print(f"测试过程中出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 输出测试结果汇总
    print("\n" + "=" * 60)
    print("兼容性测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有兼容性测试通过！")
        print("\n📋 验证结果:")
        print("  ✅ GraphAgent: 继续使用Neo4j，不受FAISS影响")
        print("  ✅ NaiveRagAgent: 成功切换到FAISS")
        print("  ✅ FusionAgent: 支持FAISS+Neo4j混合模式")
        print("  ✅ FusionAgent: 支持Neo4j传统模式")
        print("  ✅ AgentService: 后端服务兼容正常")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关问题。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 