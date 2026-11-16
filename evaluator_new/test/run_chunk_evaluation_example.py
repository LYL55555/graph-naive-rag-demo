#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小化测试：
- 构造少量样本，验证锅炉化清理、语言检测、极端相似折叠、alias 映射、仅 canonical 写入的流程。
运行：python -m evaluator_new.test.run_chunk_evaluation_example
"""

import sys
from pathlib import Path
import json

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from processor.canonical_dedup_pipeline import CanonicalDedupPipeline, DedupConfig


class _FakeEmbeddings:
    """离线最小化测试用的简易向量器，避免真实API依赖。
    将文本映射为26维字符计数向量（a-z），再做归一化。
    仅用于单元测试，不用于正式构建。
    """

    def _vec(self, text: str):
        import math
        v = [0.0] * 26
        for ch in text.lower():
            if 'a' <= ch <= 'z':
                v[ord(ch) - 97] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t or "") for t in texts]


def main() -> int:
    tmp_dir = project_root / "data" / "_mini_test"
    files_dir = tmp_dir / "files"
    out_dir = tmp_dir / "out"
    files_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 构造三个片段：
    # 1) 中文原文；2) 英文翻译成中文版（语义极近）；3) 锅炉化页脚
    (files_dir / "a.txt").write_text(
        "华东理工大学学生违纪处分规定：第一条 本规定适用于在校学生。\n版权所有：XX出版社\n第 1 页/10\n",
        encoding="utf-8",
    )
    (files_dir / "b.txt").write_text(
        "根据学校条例，对学生的纪律处分按规定执行。This line is English but overall content mirrors.\n",
        encoding="utf-8",
    )
    (files_dir / "c.txt").write_text(
        "版权所有：XX出版社\n第 2 页/10\n扫码下载应用\n",
        encoding="utf-8",
    )

    cfg = DedupConfig(
        files_dir=str(files_dir),
        output_dir=str(out_dir),
        write_faiss=False,  # 最小化测试不写入外部依赖
        write_neo4j=False,
        extreme_threshold_within=0.985,
        extreme_threshold_cross=0.983,
    )
    pipe = CanonicalDedupPipeline(cfg)
    # 替换真实Embedding为离线假的Embedding，保证最小化测试可运行
    pipe.embeddings = _FakeEmbeddings()
    stats = pipe.run()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    # 至少应该产生一个 canonical（去除锅炉化后），且存在 alias 或 1 组等价
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Chunk长度评估使用示例

这个脚本演示了如何使用chunk长度评估工具来测试不同配置的性能。
"""

import os
import sys
import time
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def run_quick_test():
    """运行快速测试示例"""
    print("=" * 60)
    print("Chunk长度快速测试示例")
    print("=" * 60)
    
    try:
        from quick_chunk_test import QuickChunkTester
        
        print("开始快速测试...")
        start_time = time.time()
        
        tester = QuickChunkTester()
        results = tester.run_quick_test()
        
        total_time = time.time() - start_time
        print(f"\n快速测试完成！总耗时: {total_time:.2f}秒")
        
        # 显示简要结果
        print("\n测试结果摘要:")
        for result in results:
            if "error" not in result:
                config_name = result["config"]["name"]
                avg_time = result["avg_search_time"]
                build_time = result["build_time"]
                total_chunks = result["total_chunks"]
                
                print(f"  {config_name:8}: 检索={avg_time:.3f}s, 构建={build_time:.2f}s, chunks={total_chunks}")
            else:
                print(f"  {result['config']['name']:8}: 错误 - {result['error']}")
        
        print(f"\n详细报告已生成: quick_chunk_test_report.md")
        
    except Exception as e:
        print(f"快速测试失败: {e}")
        import traceback
        traceback.print_exc()

def run_full_evaluation():
    """运行全面评估示例"""
    print("=" * 60)
    print("Chunk长度全面评估示例")
    print("=" * 60)
    
    try:
        from evaluate_chunk_sizes import ChunkSizeEvaluator
        
        # 检查问题文件是否存在
        questions_file = "questions.json"
        if not os.path.exists(questions_file):
            print(f"问题文件 {questions_file} 不存在，请先创建测试问题集")
            return
        
        print("开始全面评估...")
        start_time = time.time()
        
        evaluator = ChunkSizeEvaluator("./evaluation_results/chunk_evaluation_example")
        results = evaluator.run_evaluation(questions_file)
        
        total_time = time.time() - start_time
        print(f"\n全面评估完成！总耗时: {total_time:.2f}秒")
        
        # 显示评估结果
        print("\n评估结果摘要:")
        for result in results:
            if "error" not in result:
                config_name = result["config"]["name"]
                composite_score = result.get("composite_score", 0)
                retrieval_precision = result.get("RetrievalPrecision", 0)
                retrieval_latency = result.get("RetrievalLatency", 0)
                
                print(f"  {config_name:8}: 综合得分={composite_score:.4f}, 精确率={retrieval_precision:.4f}, 延迟={retrieval_latency:.3f}s")
            else:
                print(f"  {result['config']['name']:8}: 错误 - {result['error']}")
        
        print(f"\n详细报告已生成: evaluation_results/chunk_evaluation_example/chunk_size_comparison_report.md")
        
    except Exception as e:
        print(f"全面评估失败: {e}")
        import traceback
        traceback.print_exc()

def create_sample_questions():
    """创建示例问题文件"""
    questions_file = "questions.json"
    
    if os.path.exists(questions_file):
        print(f"问题文件 {questions_file} 已存在")
        return
    
    sample_questions = [
        "旷课多少学时会被退学？",
        "国家奖学金和国家励志奖学金互斥吗？",
        "优秀学生要怎么申请？",
        "那上海市奖学金呢？",
        "学生申诉的程序是什么？",
        "学生违纪处分有哪些类型？",
        "学生请假需要什么手续？",
        "勤工助学的申请条件是什么？",
        "学生宿舍管理规定有哪些？",
        "学生社团活动有什么规定？"
    ]
    
    import json
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(sample_questions, f, ensure_ascii=False, indent=2)
    
    print(f"已创建示例问题文件: {questions_file}")

def main():
    """主函数"""
    print("Chunk长度评估工具使用示例")
    print("=" * 60)
    
    # 创建示例问题文件
    create_sample_questions()
    
    # 选择评估方式
    print("\n请选择评估方式:")
    print("1. 快速测试 (推荐，10-30分钟)")
    print("2. 全面评估 (1-3小时)")
    print("3. 退出")
    
    while True:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            run_quick_test()
            break
        elif choice == "2":
            run_full_evaluation()
            break
        elif choice == "3":
            print("退出程序")
            break
        else:
            print("无效选择，请输入 1-3")

if __name__ == "__main__":
    main() 