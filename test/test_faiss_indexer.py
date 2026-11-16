#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISS索引构建器测试脚本
测试文档读取、分块、索引构建和搜索功能
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processor.faiss_indexer import FAISSIndexBuilder, build_faiss_index, load_faiss_index


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试1: 基本功能测试 ===")
    
    # 创建构建器实例
    builder = FAISSIndexBuilder()
    
    # 获取索引信息
    info = builder.get_index_info()
    print(f"索引信息: {info}")
    
    # 构建索引（强制重建）
    print("\n开始构建FAISS索引...")
    faiss_index = builder.build_index(force_rebuild=True)
    
    if faiss_index:
        print("✓ 索引构建成功")
        
        # 测试加载
        print("\n测试加载索引...")
        loaded_index = builder.load_index()
        if loaded_index:
            print("✓ 索引加载成功")
        else:
            print("✗ 索引加载失败")
            
        return True
    else:
        print("✗ 索引构建失败")
        return False


def test_search_functionality():
    """测试搜索功能"""
    print("\n=== 测试2: 搜索功能测试 ===")
    
    builder = FAISSIndexBuilder()
    
    # 测试查询列表
    test_queries = [
        "学生手册",
        "奖学金评定",
        "课程教学",
        "申请表格",
        "论文答辩",
        "勤工助学",
        "英语课程"
    ]
    
    search_results = {}
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        results = builder.test_search(query, k=3)
        
        if results:
            search_results[query] = len(results)
            for result in results:
                print(f"  排名{result['rank']}: {result['content'][:100]}...")
                print(f"    来源: {result['metadata'].get('filename', 'Unknown')}")
                print(f"    长度: {result['content_length']} 字符")
        else:
            print("  未找到相关结果")
            search_results[query] = 0
    
    # 统计搜索结果
    total_queries = len(test_queries)
    successful_queries = sum(1 for count in search_results.values() if count > 0)
    
    print(f"\n搜索测试统计:")
    print(f"  总查询数: {total_queries}")
    print(f"  成功查询数: {successful_queries}")
    print(f"  成功率: {successful_queries/total_queries*100:.1f}%")
    
    return successful_queries > 0


def test_performance():
    """测试性能"""
    print("\n=== 测试3: 性能测试 ===")
    
    builder = FAISSIndexBuilder()
    
    # 测试构建时间
    start_time = time.time()
    faiss_index = builder.build_index(force_rebuild=True)
    build_time = time.time() - start_time
    
    if faiss_index:
        print(f"✓ 索引构建耗时: {build_time:.2f} 秒")
        
        # 测试搜索时间
        test_query = "学生手册"
        search_times = []
        
        for i in range(5):
            start_time = time.time()
            results = builder.test_search(test_query, k=5)
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"✓ 平均搜索耗时: {avg_search_time:.4f} 秒")
        
        # 获取索引统计信息
        info = builder.get_index_info()
        if 'chunks_count' in info:
            print(f"✓ 索引chunks数量: {info['chunks_count']}")
            print(f"✓ 平均每个chunk搜索时间: {avg_search_time/info['chunks_count']*1000:.4f} 毫秒")
        
        return True
    else:
        print("✗ 性能测试失败，无法构建索引")
        return False


def test_edge_cases():
    """测试边缘情况"""
    print("\n=== 测试4: 边缘情况测试 ===")
    
    builder = FAISSIndexBuilder()
    
    # 测试空查询
    print("测试空查询...")
    empty_results = builder.test_search("", k=3)
    print(f"空查询结果数: {len(empty_results)}")
    
    # 测试很长的查询
    print("测试长查询...")
    long_query = "这是一个非常长的查询文本" * 20
    long_results = builder.test_search(long_query, k=3)
    print(f"长查询结果数: {len(long_results)}")
    
    # 测试特殊字符查询
    print("测试特殊字符查询...")
    special_query = "!@#$%^&*()_+"
    special_results = builder.test_search(special_query, k=3)
    print(f"特殊字符查询结果数: {len(special_results)}")
    
    # 测试数字查询
    print("测试数字查询...")
    number_query = "2023"
    number_results = builder.test_search(number_query, k=3)
    print(f"数字查询结果数: {len(number_results)}")
    
    return True


def test_convenience_functions():
    """测试便捷函数"""
    print("\n=== 测试5: 便捷函数测试 ===")
    
    # 测试便捷构建函数
    print("测试便捷构建函数...")
    faiss_index = build_faiss_index(force_rebuild=True)
    
    if faiss_index:
        print("✓ 便捷构建函数成功")
        
        # 测试便捷加载函数
        print("测试便捷加载函数...")
        loaded_index = load_faiss_index()
        
        if loaded_index:
            print("✓ 便捷加载函数成功")
            return True
        else:
            print("✗ 便捷加载函数失败")
            return False
    else:
        print("✗ 便捷构建函数失败")
        return False


def main():
    """主测试函数"""
    print("开始FAISS索引构建器测试")
    print("=" * 50)
    
    # 检查必要的目录和文件
    project_root = Path(__file__).parent.parent
    files_dir = project_root / "files"
    
    if not files_dir.exists():
        print(f"错误: 文档目录不存在: {files_dir}")
        return False
    
    print(f"文档目录: {files_dir}")
    print(f"文档数量: {len(list(files_dir.rglob('*')))} 个文件")
    
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
        
        # 边缘情况测试
        result4 = test_edge_cases()
        test_results.append(("边缘情况", result4))
        
        # 便捷函数测试
        result5 = test_convenience_functions()
        test_results.append(("便捷函数", result5))
        
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
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！FAISS索引构建器工作正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 