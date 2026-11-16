#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISS索引构建和使用示例
展示如何构建FAISS索引并进行搜索
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processor.faiss_indexer import FAISSIndexBuilder


def main():
    """主函数"""
    print("=== FAISS索引构建和使用示例 ===\n")
    
    # 1. 创建FAISS索引构建器
    print("1. 创建FAISS索引构建器...")
    builder = FAISSIndexBuilder(
        cache_dir="data/faiss_indices",  # 使用持久化目录
        batch_size=50  # 较小的批处理大小用于示例
    )
    
    # 2. 检查当前状态
    print("2. 检查索引状态...")
    info = builder.get_index_info()
    print(f"   索引存在: {info['index_exists']}")
    print(f"   缓存目录: {info['cache_dir']}")
    print(f"   文档目录: {info['files_dir']}")
    
    # 3. 构建索引
    print("\n3. 构建FAISS索引...")
    faiss_index = builder.build_index(
        file_extensions=['.txt', '.pdf'],  # 只处理txt和pdf文件
        force_rebuild=True  # 强制重建以确保获得最新数据
    )
    
    if not faiss_index:
        print("   ✗ 索引构建失败")
        return False
    
    print("   ✓ 索引构建成功")
    
    # 4. 显示构建统计信息
    print("\n4. 构建统计信息:")
    updated_info = builder.get_index_info()
    if 'build_stats' in updated_info:
        stats = updated_info['build_stats']
        print(f"   处理文档: {stats.get('files_processed', 0)} 个")
        print(f"   生成chunks: {stats.get('chunks_created', 0)} 个")
        print(f"   构建时间: {stats.get('build_time', 0):.2f} 秒")
    
    # 5. 进行搜索测试
    print("\n5. 搜索测试:")
    test_queries = [
        "学生手册相关内容",
        "奖学金申请条件",
        "课程教学安排",
        "论文答辩流程"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   查询 {i}: '{query}'")
        results = builder.test_search(query, k=2)  # 返回top-2结果
        
        if results:
            for result in results:
                print(f"     排名{result['rank']}: {result['content'][:80]}...")
                print(f"       来源: {result['metadata'].get('filename', 'Unknown')}")
                print(f"       chunk索引: {result['metadata'].get('chunk_index', 'N/A')}")
        else:
            print("     未找到相关结果")
    
    # 6. 测试索引加载
    print("\n6. 测试索引加载...")
    loaded_index = builder.load_index()
    if loaded_index:
        print("   ✓ 索引加载成功")
        
        # 用加载的索引进行一次搜索
        test_docs = loaded_index.similarity_search("学生", k=1)
        if test_docs:
            print(f"   测试搜索结果: {test_docs[0].page_content[:50]}...")
    else:
        print("   ✗ 索引加载失败")
    
    print("\n=== 示例完成 ===")
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