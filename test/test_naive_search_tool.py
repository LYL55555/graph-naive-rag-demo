#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修改后的NaiveSearchTool测试脚本
测试FAISS集成后的搜索功能
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from search.tool.naive_search_tool import NaiveSearchTool
from processor.faiss_indexer import build_faiss_index


def setup_test_environment():
    """设置测试环境"""
    print("=== 设置测试环境 ===")
    
    # 确保FAISS索引存在
    print("检查FAISS索引...")
    faiss_index = build_faiss_index(cache_dir="data/faiss_indices", force_rebuild=False)
    
    if not faiss_index:
        print("FAISS索引不存在，开始构建...")
        faiss_index = build_faiss_index(cache_dir="data/faiss_indices", force_rebuild=True)
        
        if not faiss_index:
            print("❌ FAISS索引构建失败，无法继续测试")
            return False
    
    print("✓ FAISS索引准备完成")
    return True


def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试1: 基本功能测试 ===")
    
    # 创建搜索工具
    search_tool = NaiveSearchTool()
    
    # 检查索引加载状态
    stats = search_tool.get_search_stats()
    print(f"索引加载状态: {stats['faiss_index_loaded']}")
    print(f"FAISS缓存目录: {stats['faiss_cache_dir']}")
    print(f"检索数量: {stats['top_k']}")
    
    if not stats['faiss_index_loaded']:
        print("❌ FAISS索引未加载")
        return False
    
    print("✓ 基本功能正常")
    return True


def test_search_functionality():
    """测试搜索功能"""
    print("\n=== 测试2: 搜索功能测试 ===")
    
    search_tool = NaiveSearchTool()
    
    # 测试查询列表
    test_queries = [
        "学生手册",
        "奖学金申请",
        "课程教学",
        "论文答辩",
        "勤工助学管理办法",
        "英语课程实施方案",
        "请假申请表"
    ]
    
    results_summary = {}
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        
        start_time = time.time()
        result = search_tool.search(query)
        search_time = time.time() - start_time
        
        # 分析结果
        if "没有找到" in result or "出错" in result:
            print("  ❌ 搜索失败或无结果")
            results_summary[query] = "failed"
        else:
            print(f"  ✓ 搜索成功 (耗时: {search_time:.4f}s)")
            print(f"  结果长度: {len(result)} 字符")
            
            # 检查是否包含引用信息
            if "{'data': {'Chunks':" in result or '{"data": {"Chunks":' in result:
                print("  ✓ 包含引用信息")
            else:
                print("  ⚠️ 缺少引用信息")
            
            # 显示结果片段
            preview = result[:200].replace('\n', ' ')
            print(f"  预览: {preview}...")
            
            results_summary[query] = "success"
    
    # 统计结果
    successful = sum(1 for status in results_summary.values() if status == "success")
    total = len(test_queries)
    
    print(f"\n搜索测试统计:")
    print(f"  成功: {successful}/{total} ({successful/total*100:.1f}%)")
    
    return successful > 0


def test_performance():
    """测试性能"""
    print("\n=== 测试3: 性能测试 ===")
    
    search_tool = NaiveSearchTool()
    
    test_query = "学生手册相关内容"
    times = []
    
    print(f"对查询 '{test_query}' 进行5次性能测试...")
    
    for i in range(5):
        start_time = time.time()
        result = search_tool.search(test_query)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        print(f"  测试 {i+1}: {elapsed:.4f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n性能统计:")
    print(f"  平均耗时: {avg_time:.4f}s")
    print(f"  最短耗时: {min_time:.4f}s")
    print(f"  最长耗时: {max_time:.4f}s")
    
    # 获取详细性能指标
    stats = search_tool.get_search_stats()
    if 'performance_metrics' in stats:
        metrics = stats['performance_metrics']
        print(f"  查询时间: {metrics.get('query_time', 0):.4f}s")
        print(f"  LLM时间: {metrics.get('llm_time', 0):.4f}s")
        print(f"  总时间: {metrics.get('total_time', 0):.4f}s")
    
    return avg_time < 10.0  # 期望平均响应时间小于10秒


def test_caching():
    """测试缓存功能"""
    print("\n=== 测试4: 缓存功能测试 ===")
    
    search_tool = NaiveSearchTool()
    
    test_query = "测试缓存查询"
    
    # 第一次查询（应该不命中缓存）
    print("第一次查询...")
    start_time = time.time()
    result1 = search_tool.search(test_query)
    time1 = time.time() - start_time
    
    # 第二次查询（应该命中缓存）
    print("第二次查询（测试缓存）...")
    start_time = time.time()
    result2 = search_tool.search(test_query)
    time2 = time.time() - start_time
    
    print(f"第一次查询耗时: {time1:.4f}s")
    print(f"第二次查询耗时: {time2:.4f}s")
    
    # 验证结果一致性
    results_match = result1 == result2
    cache_faster = time2 < time1 * 0.5  # 缓存应该快很多
    
    print(f"结果一致性: {'✓' if results_match else '❌'}")
    print(f"缓存加速: {'✓' if cache_faster else '❌'}")
    
    return results_match and cache_faster


def test_edge_cases():
    """测试边缘情况"""
    print("\n=== 测试5: 边缘情况测试 ===")
    
    search_tool = NaiveSearchTool()
    
    edge_cases = [
        ("", "空查询"),
        ("   ", "空白查询"),
        ("a", "单字符查询"),
        ("这是一个非常长的查询" * 50, "超长查询"),
        ("!@#$%^&*()", "特殊字符查询"),
        ("1234567890", "纯数字查询"),
        ("English query", "英文查询"),
        ("混合query测试123", "混合语言查询")
    ]
    
    results = {}
    
    for query, description in edge_cases:
        print(f"\n测试 {description}: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        try:
            result = search_tool.search(query)
            
            if "请提供有效的查询内容" in result:
                print("  ✓ 正确处理无效查询")
                results[description] = "handled"
            elif "没有找到" in result:
                print("  ✓ 正确返回无结果")
                results[description] = "no_result"
            elif "出错" in result:
                print("  ❌ 查询出错")
                results[description] = "error"
            else:
                print("  ✓ 正常返回结果")
                results[description] = "success"
                
        except Exception as e:
            print(f"  ❌ 异常: {str(e)}")
            results[description] = "exception"
    
    # 统计结果
    handled_count = sum(1 for status in results.values() 
                       if status in ["handled", "no_result", "success"])
    total_count = len(edge_cases)
    
    print(f"\n边缘情况处理统计:")
    print(f"  正常处理: {handled_count}/{total_count}")
    
    return handled_count == total_count


def test_tool_interface():
    """测试工具接口"""
    print("\n=== 测试6: 工具接口测试 ===")
    
    search_tool = NaiveSearchTool()
    
    # 获取工具实例
    tool = search_tool.get_tool()
    
    print(f"工具名称: {tool.name}")
    print(f"工具描述: {tool.description[:100]}...")
    
    # 测试工具调用
    test_query = "学生手册"
    print(f"\n通过工具接口查询: '{test_query}'")
    
    try:
        result = tool._run(test_query)
        
        if result and len(result) > 10:
            print("✓ 工具接口正常工作")
            print(f"结果长度: {len(result)} 字符")
            return True
        else:
            print("❌ 工具接口返回异常结果")
            return False
            
    except Exception as e:
        print(f"❌ 工具接口调用失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("开始修改后的NaiveSearchTool测试")
    print("=" * 50)
    
    # 设置测试环境
    if not setup_test_environment():
        print("❌ 测试环境设置失败")
        return False
    
    # 运行测试
    test_results = []
    
    try:
        # 基本功能测试
        result1 = test_basic_functionality()
        test_results.append(("基本功能", result1))
        
        # 搜索功能测试
        result2 = test_search_functionality()
        test_results.append(("搜索功能", result2))
        
        # 性能测试
        result3 = test_performance()
        test_results.append(("性能测试", result3))
        
        # 缓存功能测试
        result4 = test_caching()
        test_results.append(("缓存功能", result4))
        
        # 边缘情况测试
        result5 = test_edge_cases()
        test_results.append(("边缘情况", result5))
        
        # 工具接口测试
        result6 = test_tool_interface()
        test_results.append(("工具接口", result6))
        
    except Exception as e:
        print(f"测试过程中出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！NaiveSearchTool FAISS集成成功。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关问题。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 