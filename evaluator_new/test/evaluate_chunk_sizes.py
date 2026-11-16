import os
import sys
import json
import argparse
import time
import shutil
from typing import List, Dict, Any, Tuple
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluator import set_debug_mode
from evaluator.utils.logging_utils import setup_logger
from evaluator.evaluator_config.agent_evaluation_config import get_agent_metrics
from evaluator.utils.eval_utils import evaluate_agent, load_questions_and_answers
from processor.faiss_indexer import FAISSIndexBuilder
from config.settings import FILES_DIR

class ChunkSizeEvaluator:
    """Chunk长度评估器"""
    
    def __init__(self, base_dir: str = "./evaluation_results/chunk_size_evaluation"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("chunk_size_evaluation", self.base_dir / "evaluation.log")
        
        # 定义要测试的chunk大小配置
        self.chunk_configs = [
            {"name": "small", "chunk_size": 200, "overlap": 40},
            {"name": "medium", "chunk_size": 500, "overlap": 100},  # 当前配置
            {"name": "large", "chunk_size": 800, "overlap": 150},
            {"name": "xlarge", "chunk_size": 1200, "overlap": 200},
        ]
        
        # 评估指标
        self.metrics = [
            "RetrievalPrecision",
            "RetrievalUtilization", 
            "RetrievalLatency",
            "ChunkUtilization",
            "AnswerRelevance",
            "AnswerCompleteness",
            "AnswerAccuracy"
        ]
    
    def backup_current_config(self):
        """备份当前配置"""
        config_backup = self.base_dir / "config_backup"
        config_backup.mkdir(exist_ok=True)
        
        # 备份当前settings.py中的chunk配置
        settings_path = Path("config/settings.py")
        if settings_path.exists():
            shutil.copy2(settings_path, config_backup / "settings_backup.py")
            self.logger.info("已备份当前配置")
    
    def modify_chunk_config(self, chunk_size: int, overlap: int):
        """临时修改chunk配置"""
        settings_path = Path("config/settings.py")
        
        if not settings_path.exists():
            self.logger.error("找不到config/settings.py文件")
            return False
        
        # 读取当前配置
        with open(settings_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换配置
        content = content.replace(
            f"CHUNK_SIZE = {self._extract_current_chunk_size(content)}",
            f"CHUNK_SIZE = {chunk_size}"
        )
        content = content.replace(
            f"OVERLAP = {self._extract_current_overlap(content)}",
            f"OVERLAP = {overlap}"
        )
        
        # 写回文件
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"已修改配置: CHUNK_SIZE={chunk_size}, OVERLAP={overlap}")
        return True
    
    def _extract_current_chunk_size(self, content: str) -> int:
        """提取当前CHUNK_SIZE值"""
        import re
        match = re.search(r'CHUNK_SIZE\s*=\s*(\d+)', content)
        return int(match.group(1)) if match else 500
    
    def _extract_current_overlap(self, content: str) -> int:
        """提取当前OVERLAP值"""
        import re
        match = re.search(r'OVERLAP\s*=\s*(\d+)', content)
        return int(match.group(1)) if match else 100
    
    def rebuild_faiss_index(self, config_name: str) -> bool:
        """重建FAISS索引"""
        try:
            self.logger.info(f"开始重建FAISS索引: {config_name}")
            
            # 清理旧的索引
            index_builder = FAISSIndexBuilder()
            cache_dir = index_builder.cache_dir
            
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                self.logger.info(f"已清理旧索引: {cache_dir}")
            
            # 重建索引
            start_time = time.time()
            faiss_index = index_builder.build_index(force_rebuild=True)
            build_time = time.time() - start_time
            
            if faiss_index:
                self.logger.info(f"FAISS索引重建成功，耗时: {build_time:.2f}秒")
                return True
            else:
                self.logger.error("FAISS索引重建失败")
                return False
                
        except Exception as e:
            self.logger.error(f"重建FAISS索引时出错: {e}")
            return False
    
    def evaluate_config(self, config: Dict[str, Any], questions: List[str], 
                       golden_answers: List[Dict] = None) -> Dict[str, Any]:
        """评估单个配置"""
        config_name = config["name"]
        chunk_size = config["chunk_size"]
        overlap = config["overlap"]
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"开始评估配置: {config_name}")
        self.logger.info(f"Chunk大小: {chunk_size}, 重叠: {overlap}")
        self.logger.info(f"{'='*50}")
        
        # 创建配置专用目录
        config_dir = self.base_dir / config_name
        config_dir.mkdir(exist_ok=True)
        
        try:
            # 1. 修改配置
            if not self.modify_chunk_config(chunk_size, overlap):
                return {"error": "配置修改失败"}
            
            # 2. 重建FAISS索引
            if not self.rebuild_faiss_index(config_name):
                return {"error": "FAISS索引重建失败"}
            
            # 3. 执行评估
            start_time = time.time()
            evaluate_agent(
                agent_type="naive_faiss",
                questions=questions,
                golden_answers=golden_answers,
                save_dir=str(config_dir),
                metrics=self.metrics,
                verbose=False
            )
            eval_time = time.time() - start_time
            
            # 4. 收集评估结果
            results = self._collect_evaluation_results(config_dir)
            results["config"] = config
            results["evaluation_time"] = eval_time
            
            self.logger.info(f"配置 {config_name} 评估完成，耗时: {eval_time:.2f}秒")
            return results
            
        except Exception as e:
            self.logger.error(f"评估配置 {config_name} 时出错: {e}")
            return {"error": str(e)}
    
    def _collect_evaluation_results(self, config_dir: Path) -> Dict[str, Any]:
        """收集评估结果"""
        results = {}
        
        # 读取评估结果文件
        metric_score_file = config_dir / "metric_score.txt"
        if metric_score_file.exists():
            with open(metric_score_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        metric, score = line.strip().split(':', 1)
                        try:
                            results[metric.strip()] = float(score.strip())
                        except ValueError:
                            continue
        
        # 读取详细评估数据
        evaluation_data_file = config_dir / "evaluation_data.json"
        if evaluation_data_file.exists():
            with open(evaluation_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results["detailed_data"] = data
        
        return results
    
    def generate_comparison_report(self, all_results: List[Dict[str, Any]]):
        """生成对比报告"""
        report_file = self.base_dir / "chunk_size_comparison_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Chunk长度评估对比报告\n\n")
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 配置对比表
            f.write("## 配置对比\n\n")
            f.write("| 配置名称 | Chunk大小 | 重叠大小 | 重叠比例 |\n")
            f.write("|---------|-----------|----------|----------|\n")
            
            for result in all_results:
                if "error" not in result:
                    config = result["config"]
                    overlap_ratio = config["overlap"] / config["chunk_size"] * 100
                    f.write(f"| {config['name']} | {config['chunk_size']} | {config['overlap']} | {overlap_ratio:.1f}% |\n")
            
            # 性能对比表
            f.write("\n## 性能对比\n\n")
            f.write("| 配置名称 | 检索精确率 | 检索利用率 | 检索延迟(秒) | 块利用率 | 答案相关性 | 答案完整性 | 答案准确性 |\n")
            f.write("|---------|------------|------------|--------------|----------|------------|------------|------------|\n")
            
            for result in all_results:
                if "error" not in result:
                    config_name = result["config"]["name"]
                    metrics = [
                        result.get("RetrievalPrecision", 0),
                        result.get("RetrievalUtilization", 0),
                        result.get("RetrievalLatency", 0),
                        result.get("ChunkUtilization", 0),
                        result.get("AnswerRelevance", 0),
                        result.get("AnswerCompleteness", 0),
                        result.get("AnswerAccuracy", 0)
                    ]
                    
                    f.write(f"| {config_name} | " + " | ".join(f"{m:.4f}" for m in metrics) + " |\n")
            
            # 综合分析
            f.write("\n## 综合分析\n\n")
            
            # 找出最佳配置
            best_config = self._find_best_config(all_results)
            if best_config:
                f.write(f"**推荐配置**: {best_config['config']['name']}\n")
                f.write(f"- Chunk大小: {best_config['config']['chunk_size']}\n")
                f.write(f"- 重叠大小: {best_config['config']['overlap']}\n")
                f.write(f"- 综合得分: {best_config.get('composite_score', 0):.4f}\n\n")
            
            # 详细分析
            f.write("### 详细分析\n\n")
            f.write("1. **检索性能**: 较小的chunk通常提供更精确的检索，但可能丢失上下文\n")
            f.write("2. **答案质量**: 较大的chunk提供更丰富的上下文，但可能包含无关信息\n")
            f.write("3. **计算效率**: 较小的chunk计算更快，但块数量更多\n")
            f.write("4. **存储开销**: 较小的chunk存储开销更大，但检索更灵活\n\n")
            
            # 错误信息
            errors = [r for r in all_results if "error" in r]
            if errors:
                f.write("### 错误信息\n\n")
                for error in errors:
                    f.write(f"- {error['config']['name']}: {error['error']}\n")
        
        self.logger.info(f"对比报告已生成: {report_file}")
    
    def _find_best_config(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """找出最佳配置"""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return None
        
        # 计算综合得分（可以调整权重）
        for result in valid_results:
            scores = [
                result.get("RetrievalPrecision", 0) * 0.25,
                result.get("RetrievalUtilization", 0) * 0.20,
                (1 - min(result.get("RetrievalLatency", 1), 1)) * 0.15,  # 延迟越低越好
                result.get("ChunkUtilization", 0) * 0.15,
                result.get("AnswerRelevance", 0) * 0.10,
                result.get("AnswerCompleteness", 0) * 0.10,
                result.get("AnswerAccuracy", 0) * 0.05
            ]
            result["composite_score"] = sum(scores)
        
        # 按综合得分排序
        valid_results.sort(key=lambda x: x["composite_score"], reverse=True)
        return valid_results[0]
    
    def run_evaluation(self, questions_file: str, golden_answers_file: str = None):
        """运行完整的chunk长度评估"""
        self.logger.info("开始Chunk长度评估")
        
        # 备份当前配置
        self.backup_current_config()
        
        # 加载问题
        questions, golden_answers = load_questions_and_answers(questions_file, golden_answers_file)
        self.logger.info(f"加载了 {len(questions)} 个问题")
        
        # 评估所有配置
        all_results = []
        for config in self.chunk_configs:
            result = self.evaluate_config(config, questions, golden_answers)
            all_results.append(result)
        
        # 生成对比报告
        self.generate_comparison_report(all_results)
        
        # 恢复原始配置
        self._restore_original_config()
        
        self.logger.info("Chunk长度评估完成")
        return all_results
    
    def _restore_original_config(self):
        """恢复原始配置"""
        backup_file = self.base_dir / "config_backup" / "settings_backup.py"
        if backup_file.exists():
            shutil.copy2(backup_file, "config/settings.py")
            self.logger.info("已恢复原始配置")

def main():
    parser = argparse.ArgumentParser(description="评估不同Chunk长度的表现")
    parser.add_argument("--questions_file", type=str, required=True,
                        help="问题文件路径")
    parser.add_argument("--golden_answers_file", type=str, default=None,
                        help="标准答案文件路径（可选）")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results/chunk_size_evaluation",
                        help="输出目录")
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    
    args = parser.parse_args()
    
    # 设置调试模式
    set_debug_mode(args.verbose)
    
    # 创建评估器并运行
    evaluator = ChunkSizeEvaluator(args.output_dir)
    results = evaluator.run_evaluation(args.questions_file, args.golden_answers_file)
    
    print(f"\n评估完成！结果保存在: {args.output_dir}")
    print("查看对比报告: chunk_size_comparison_report.md")

if __name__ == "__main__":
    main() 