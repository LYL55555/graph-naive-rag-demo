#!/usr/bin/env python3
"""
Evaluator缓存清理工具

专门用于清理evaluator测试过程中产生的缓存文件，
不影响其他模块的缓存。
"""

import os
import shutil
from pathlib import Path
import argparse


def clean_evaluator_cache(dry_run: bool = False) -> None:
    """
    清理evaluator相关的缓存
    
    Args:
        dry_run: 是否只显示要删除的内容而不实际删除
    """
    print("🧹 Evaluator缓存清理工具")
    print("=" * 50)
    
    # 获取项目根目录
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    # 定义要清理的缓存目录
    cache_dirs_to_clean = [
        # evaluator测试缓存
        project_root / "evaluator" / "test" / "cache",
        
        # 各个agent的缓存（只清理evaluator相关的）
        project_root / "cache" / "naive_agent",
        project_root / "cache" / "graph_agent", 
        project_root / "cache" / "hybrid_agent",
        project_root / "cache" / "fusion_agent",
        project_root / "cache" / "deep_research_agent",
        
        # 搜索工具缓存
        project_root / "cache" / "naive_search",
        project_root / "cache" / "local_search",
        project_root / "cache" / "global_search", 
        project_root / "cache" / "hybrid_search",
        
        # 注意：FAISS索引现在存储在 data/faiss_indices 中，属于持久化数据，不应被清理
        
        # 评估结果缓存
        project_root / "evaluator" / "test" / "evaluation_results",
    ]
    
    # 定义要保留的文件
    files_to_keep = {
        ".gitkeep", 
        "readme.md", 
        "README.md",
        "__init__.py"
    }
    
    total_deleted = 0
    total_size_freed = 0
    
    for cache_dir in cache_dirs_to_clean:
        if not cache_dir.exists():
            print(f"⏩ 跳过不存在的目录: {cache_dir.relative_to(project_root)}")
            continue
            
        print(f"\n📁 检查目录: {cache_dir.relative_to(project_root)}")
        
        # 计算目录大小
        dir_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = Path(root) / file
                if file not in files_to_keep:
                    try:
                        file_size = file_path.stat().st_size
                        dir_size += file_size
                        file_count += 1
                        
                        if dry_run:
                            print(f"  🗑️  将删除: {file_path.relative_to(cache_dir)} ({file_size} bytes)")
                        else:
                            file_path.unlink()
                            print(f"  ✅ 已删除: {file_path.relative_to(cache_dir)}")
                            
                    except Exception as e:
                        print(f"  ❌ 删除失败: {file_path.relative_to(cache_dir)} - {e}")
        
        # 删除空的子目录
        if not dry_run:
            for root, dirs, files in os.walk(cache_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        if not any(dir_path.iterdir()):  # 目录为空
                            dir_path.rmdir()
                            print(f"  🗂️  删除空目录: {dir_path.relative_to(cache_dir)}")
                    except Exception as e:
                        pass  # 忽略非空目录或权限错误
        
        if file_count > 0:
            print(f"  📊 该目录: {file_count} 个文件, {dir_size / 1024:.1f} KB")
            total_deleted += file_count
            total_size_freed += dir_size
        else:
            print(f"  ✨ 该目录已经很干净了")
    
    print("\n" + "=" * 50)
    if dry_run:
        print(f"🔍 预览模式: 将删除 {total_deleted} 个文件，释放 {total_size_freed / 1024:.1f} KB")
        print("💡 使用 --execute 参数来实际执行删除操作")
    else:
        print(f"✅ 清理完成: 删除了 {total_deleted} 个文件，释放了 {total_size_freed / 1024:.1f} KB")
        print("🎉 Evaluator缓存已清理完毕！")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="清理evaluator测试相关的缓存文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python cache_cleaner.py                 # 预览要删除的文件
  python cache_cleaner.py --execute       # 实际执行删除操作
  python cache_cleaner.py --dry-run       # 同预览模式
        """
    )
    
    parser.add_argument(
        "--execute", 
        action="store_true", 
        help="实际执行删除操作（默认只预览）"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="只预览要删除的文件，不实际删除"
    )
    
    args = parser.parse_args()
    
    # 默认是dry_run模式，除非明确指定--execute
    dry_run = not args.execute or args.dry_run
    
    if dry_run:
        print("🔍 预览模式：将显示要删除的文件但不实际删除")
        print("💡 使用 --execute 参数来实际执行删除操作\n")
    else:
        print("⚠️  警告：即将删除evaluator相关的缓存文件！")
        response = input("确认继续？(y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ 操作已取消")
            return
        print()
    
    clean_evaluator_cache(dry_run=dry_run)


if __name__ == "__main__":
    main() 