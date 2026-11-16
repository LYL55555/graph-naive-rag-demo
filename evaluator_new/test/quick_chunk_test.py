import os
import sys
import time
import json
from typing import List, Dict, Any
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from processor.faiss_indexer import FAISSIndexBuilder
from search.tool.naive_search_tool import NaiveSearchTool
from config.settings import FILES_DIR

class QuickChunkTester:
    """快速Chunk长度测试器"""
    
    def __init__(self):
        self.test_questions = [
            "旷课多少学时会被退学？",
            "国家奖学金和国家励志奖学金互斥吗？",
            "优秀学生要怎么申请？",
            "那上海市奖学金呢？",
            "学生申诉的程序是什么？",
            "学生违纪处分有哪些类型？"
        ]
        
        # 测试配置
        self.chunk_configs = [
            {"name": "small", "chunk_size": 200, "overlap": 40},
            {"name": "medium", "chunk_size": 500, "overlap": 100},
            {"name": "large", "chunk_size": 800, "overlap": 150},
            {"name": "xlarge", "chunk_size": 1200, "overlap": 200},
        ]
    
    def modify_config(self, chunk_size: int, overlap: int):
        """临时修改配置"""
        settings_path = Path("config/settings.py")
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式替换
        import re
        content = re.sub(r'CHUNK_SIZE\s*=\s*\d+', f'CHUNK_SIZE = {chunk_size}', content)
        content = re.sub(r'OVERLAP\s*=\s*\d+', f'OVERLAP = {overlap}', content)
        
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def test_single_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """测试单个配置"""
        config_name = config["name"]
        chunk_size = config["chunk_size"]
        overlap = config["overlap"]
        
        print(f"\n{'='*50}")
        print(f"测试配置: {config_name}")
        print(f"Chunk大小: {chunk_size}, 重叠: {overlap}")
        print(f"{'='*50}")
        
        try:
            # 1. 修改配置
            self.modify_config(chunk_size, overlap)
            
            # 2. 重建FAISS索引
            print("重建FAISS索引...")
            start_time = time.time()
            
            index_builder = FAISSIndexBuilder()
            cache_dir = index_builder.cache_dir
            
            # 清理旧索引
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
            
            # 重建索引
            faiss_index = index_builder.build_index(force_rebuild=True)
            build_time = time.time() - start_time
            
            if not faiss_index:
                return {"error": "FAISS索引重建失败"}
            
            print(f"索引重建完成，耗时: {build_time:.2f}秒")
            
            # 3. 测试检索性能
            print("测试检索性能...")
            search_tool = NaiveSearchTool()
            
            results = {
                "config": config,
                "build_time": build_time,
                "search_results": [],
                "avg_search_time": 0,
                "total_chunks": 0
            }
            
            search_times = []
            
            for i, question in enumerate(self.test_questions):
                print(f"  测试问题 {i+1}: {question[:30]}...")
                
                search_start = time.time()
                search_result = search_tool.search(question, top_k=5)
                search_time = time.time() - search_start
                search_times.append(search_time)
                
                # 统计chunk数量
                chunk_count = len(search_result.get("chunks", []))
                results["total_chunks"] += chunk_count
                
                results["search_results"].append({
                    "question": question,
                    "search_time": search_time,
                    "chunk_count": chunk_count,
                    "result_preview": str(search_result)[:100] + "..." if len(str(search_result)) > 100 else str(search_result)
                })
                
                print(f"    检索时间: {search_time:.3f}秒, 找到chunks: {chunk_count}")
            
            results["avg_search_time"] = sum(search_times) / len(search_times)
            print(f"平均检索时间: {results['avg_search_time']:.3f}秒")
            print(f"总chunk数量: {results['total_chunks']}")
            
            return results
            
        except Exception as e:
            print(f"测试配置 {config_name} 时出错: {e}")
            return {"error": str(e)}
    
    def run_quick_test(self):
        """运行快速测试"""
        print("开始快速Chunk长度测试")
        print(f"测试问题数量: {len(self.test_questions)}")
        
        all_results = []
        
        for config in self.chunk_configs:
            result = self.test_single_config(config)
            all_results.append(result)
        
        # 生成简单报告
        self.generate_quick_report(all_results)
        
        print("\n快速测试完成！")
        return all_results
    
    def generate_quick_report(self, results: List[Dict[str, Any]]):
        """生成快速测试报告"""
        report_file = Path("quick_chunk_test_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 快速Chunk长度测试报告\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 配置对比
            f.write("## 配置对比\n\n")
            f.write("| 配置名称 | Chunk大小 | 重叠大小 | 重叠比例 |\n")
            f.write("|---------|-----------|----------|----------|\n")
            
            for result in results:
                if "error" not in result:
                    config = result["config"]
                    overlap_ratio = config["overlap"] / config["chunk_size"] * 100
                    f.write(f"| {config['name']} | {config['chunk_size']} | {config['overlap']} | {overlap_ratio:.1f}% |\n")
            
            # 性能对比
            f.write("\n## 性能对比\n\n")
            f.write("| 配置名称 | 索引构建时间(秒) | 平均检索时间(秒) | 总Chunk数量 |\n")
            f.write("|---------|------------------|------------------|------------|\n")
            
            for result in results:
                if "error" not in result:
                    config_name = result["config"]["name"]
                    build_time = result.get("build_time", 0)
                    avg_search_time = result.get("avg_search_time", 0)
                    total_chunks = result.get("total_chunks", 0)
                    
                    f.write(f"| {config_name} | {build_time:.2f} | {avg_search_time:.3f} | {total_chunks} |\n")
            
            # 详细结果
            f.write("\n## 详细测试结果\n\n")
            
            for result in results:
                if "error" not in result:
                    config_name = result["config"]["name"]
                    f.write(f"### {config_name} 配置\n\n")
                    f.write(f"- Chunk大小: {result['config']['chunk_size']}\n")
                    f.write(f"- 重叠大小: {result['config']['overlap']}\n")
                    f.write(f"- 索引构建时间: {result['build_time']:.2f}秒\n")
                    f.write(f"- 平均检索时间: {result['avg_search_time']:.3f}秒\n")
                    f.write(f"- 总Chunk数量: {result['total_chunks']}\n\n")
                    
                    f.write("**测试问题结果:**\n\n")
                    for i, search_result in enumerate(result["search_results"]):
                        f.write(f"{i+1}. **问题**: {search_result['question']}\n")
                        f.write(f"   - 检索时间: {search_result['search_time']:.3f}秒\n")
                        f.write(f"   - 找到Chunks: {search_result['chunk_count']}\n")
                        f.write(f"   - 结果预览: {search_result['result_preview']}\n\n")
                else:
                    f.write(f"### {result['config']['name']} 配置 - 错误\n\n")
                    f.write(f"错误信息: {result['error']}\n\n")
            
            # 推荐
            valid_results = [r for r in results if "error" not in r]
            if valid_results:
                # 按平均检索时间排序
                valid_results.sort(key=lambda x: x["avg_search_time"])
                fastest = valid_results[0]
                
                f.write("## 推荐配置\n\n")
                f.write(f"**最快检索配置**: {fastest['config']['name']}\n")
                f.write(f"- Chunk大小: {fastest['config']['chunk_size']}\n")
                f.write(f"- 平均检索时间: {fastest['avg_search_time']:.3f}秒\n")
                f.write(f"- 索引构建时间: {fastest['build_time']:.2f}秒\n\n")
        
        print(f"测试报告已生成: {report_file}")

def main():
    tester = QuickChunkTester()
    results = tester.run_quick_test()
    
    # 打印简要结果
    print("\n简要结果:")
    for result in results:
        if "error" not in result:
            config_name = result["config"]["name"]
            avg_time = result["avg_search_time"]
            build_time = result["build_time"]
            print(f"  {config_name}: 检索时间={avg_time:.3f}s, 构建时间={build_time:.2f}s")
        else:
            print(f"  {result['config']['name']}: 错误 - {result['error']}")

if __name__ == "__main__":
    main() 