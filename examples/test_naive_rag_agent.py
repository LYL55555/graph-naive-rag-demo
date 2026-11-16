#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NaiveRagAgent使用示例
展示如何使用FAISS集成版本的NaiveRagAgent
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.naive_rag_agent import NaiveRagAgent
from langchain_core.messages import HumanMessage


def main():
    """主函数"""
    print("=== NaiveRagAgent FAISS版本使用示例 ===\n")
    
    # 1. 创建Agent实例
    print("1. 创建NaiveRagAgent...")
    try:
        agent = NaiveRagAgent()
        print("✅ Agent创建成功")
    except Exception as e:
        print(f"❌ Agent创建失败: {str(e)}")
        return False
    
    # 2. 查看Agent信息
    print("\n2. Agent信息:")
    try:
        info = agent.get_agent_info()
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"⚠️ 获取Agent信息失败: {str(e)}")
    
    # 3. 测试查询
    print("\n3. 测试查询:")
    
    test_queries = [
        "学生手册包含哪些内容？",
        "奖学金申请的条件是什么？",
        "论文答辩需要准备什么？",
        "勤工助学的相关规定有哪些？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 查询 {i}: {query} ---")
        
        try:
            # 按照标准格式调用Agent
            messages = [HumanMessage(content=query)]
            response = agent.invoke({"messages": messages})
            
            if response and "messages" in response:
                answer = response["messages"][-1].content
                
                print("✅ 查询成功")
                print(f"响应长度: {len(answer)} 字符")
                
                # 显示主要内容（去除引用部分）
                main_content = answer.split("{'data':")[0].strip()
                lines = main_content.split('\n')
                
                print("主要内容:")
                for line in lines[:4]:  # 显示前4行
                    if line.strip():
                        preview = line[:100] + "..." if len(line) > 100 else line
                        print(f"  {preview}")
                
                # 检查引用信息
                if "{'data': {'Chunks':" in answer:
                    import re
                    chunks_match = re.search(r"{'data': {'Chunks':\[(.*?)\]", answer)
                    if chunks_match:
                        chunk_count = len(chunks_match.group(1).split(','))
                        print(f"📚 引用了 {chunk_count} 个文档片段")
                
            else:
                print("❌ 响应格式异常")
                
        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
    
    # 4. 演示特定的invoke格式（按照用户要求）
    print("\n4. 特定invoke格式演示:")
    print("运行: Agent.invoke({\"messages\": [HumanMessage(\"test query\")]})")
    
    try:
        test_message = HumanMessage("test query")
        result = agent.invoke({"messages": [test_message]})
        
        print("✅ 特定格式调用成功")
        print(f"返回类型: {type(result)}")
        print(f"包含keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict) and "messages" in result:
            print(f"消息数量: {len(result['messages'])}")
            last_msg = result["messages"][-1]
            print(f"最后消息类型: {type(last_msg).__name__}")
            content_preview = last_msg.content[:100] + "..." if len(last_msg.content) > 100 else last_msg.content
            print(f"内容预览: {content_preview}")
        
    except Exception as e:
        print(f"❌ 特定格式调用失败: {str(e)}")
    
    # 5. 性能统计
    print("\n5. 性能统计:")
    try:
        # 进行一次计时测试
        import time
        
        test_query = "学生手册相关规定"
        messages = [HumanMessage(content=test_query)]
        
        start_time = time.time()
        response = agent.invoke({"messages": messages})
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"单次查询耗时: {elapsed:.2f} 秒")
        
        if response and "messages" in response:
            content_length = len(response["messages"][-1].content)
            print(f"响应长度: {content_length} 字符")
            print(f"生成速度: {content_length/elapsed:.1f} 字符/秒")
        
    except Exception as e:
        print(f"⚠️ 性能测试失败: {str(e)}")
    
    # 6. 清理资源
    print("\n6. 清理资源...")
    try:
        agent.close()
        print("✅ 资源清理完成")
    except Exception as e:
        print(f"⚠️ 清理时出现警告: {str(e)}")
    
    print("\n=== 示例完成 ===")
    print("🎉 NaiveRagAgent FAISS版本运行正常")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✨ 示例运行成功！")
        else:
            print("\n❌ 示例运行失败！")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行过程中出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)