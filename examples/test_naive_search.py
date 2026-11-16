#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NaiveSearchTool使用示例
展示如何使用修改后的FAISS集成版本
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from search.tool.naive_search_tool import NaiveSearchTool
from processor.faiss_indexer import build_faiss_index


def main():
    """主函数"""
    print("=== NaiveSearchTool FAISS版本使用示例 ===\n")
    
    # 1. 确保FAISS索引存在
    print("1. 检查FAISS索引...")
    faiss_index = build_faiss_index(force_rebuild=False)
    
    if not faiss_index:
        print("   FAISS索引不存在，开始构建...")
        faiss_index = build_faiss_index(force_rebuild=True)
        
        if not faiss_index:
            print("   ❌ 无法构建FAISS索引")
            return False
    
    print("   ✓ FAISS索引准备完成")
    
    # 2. 创建搜索工具
    print("\n2. 创建NaiveSearchTool...")
    search_tool = NaiveSearchTool()
    
    # 检查状态
    stats = search_tool.get_search_stats()
    print(f"   索引加载状态: {stats['faiss_index_loaded']}")
    print(f"   检索参数: top_k={stats['top_k']}, threshold={stats['similarity_threshold']}")
    
    if not stats['faiss_index_loaded']:
        print("   ❌ FAISS索引未正确加载")
        return False
    
    # 3. 进行搜索测试
    print("\n3. 搜索测试:")
    
    test_queries = [
        "学生手册有什么内容？",
        "如何申请奖学金？",
        "英语课程是怎么安排的？",
        "论文答辩的具体流程是什么？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   查询 {i}: {query}")
        print("   " + "-" * 40)
        
        try:
            result = search_tool.search(query)
            
            # 分析结果
            if "没有找到" in result:
                print("   结果: 未找到相关信息")
            elif "出错" in result:
                print("   结果: 搜索出现错误")
            else:
                # 显示结果摘要
                lines = result.split('\n')
                content_lines = [line for line in lines if line.strip() and not line.startswith('{')]
                
                if content_lines:
                    print("   结果摘要:")
                    # 显示前几行内容
                    for line in content_lines[:3]:
                        if len(line.strip()) > 10:
                            print(f"     {line.strip()[:80]}...")
                
                # 检查引用信息
                if "{'data': {'Chunks':" in result:
                    import re
                    chunks_match = re.search(r"\{'data': \{'Chunks':\[(.*?)\]", result)
                    if chunks_match:
                        chunk_refs = chunks_match.group(1)
                        print(f"   引用chunks: {chunk_refs}")
                
                print(f"   完整回答长度: {len(result)} 字符")
        
        except Exception as e:
            print(f"   ❌ 查询失败: {str(e)}")
    
    # 4. 性能统计
    print("\n4. 性能统计:")
    final_stats = search_tool.get_search_stats()
    if 'performance_metrics' in final_stats:
        metrics = final_stats['performance_metrics']
        print(f"   最后查询时间: {metrics.get('query_time', 0):.4f}s")
        print(f"   最后LLM时间: {metrics.get('llm_time', 0):.4f}s")
        print(f"   最后总时间: {metrics.get('total_time', 0):.4f}s")
    
    # 5. 测试工具接口
    print("\n5. 测试工具接口:")
    tool = search_tool.get_tool()
    print(f"   工具名称: {tool.name}")
    print(f"   工具描述: {tool.description[:60]}...")
    
    # 通过工具接口进行一次查询
    tool_query = "学生手册"
    print(f"   通过工具查询: '{tool_query}'")
    
    try:
        tool_result = tool._run(tool_query)
        if tool_result and len(tool_result) > 10:
            print("   ✓ 工具接口正常工作")
            # 显示简短摘要
            summary = tool_result[:100].replace('\n', ' ')
            print(f"   结果预览: {summary}...")
        else:
            print("   ❌ 工具接口返回异常")
    except Exception as e:
        print(f"   ❌ 工具接口调用失败: {str(e)}")
    
    print("\n=== 示例完成 ===")
    print("✓ NaiveSearchTool FAISS版本运行正常")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 示例运行成功！")
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