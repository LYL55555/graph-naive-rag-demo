#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
加载 canonical FAISS 索引并做简单查询验证：
- 验证可加载
- 运行若干查询，查看是否有结果，打印来源与chunk信息

运行：python examples/test_faiss_canonical_search.py --index ./data/canonical_index/canonical_faiss
"""

import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processor.faiss_indexer import FAISSIndexBuilder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=str, default=str(project_root / "data" / "canonical_index" / "canonical_faiss"))
    p.add_argument("--k", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    builder = FAISSIndexBuilder(cache_dir=args.index)
    idx = builder.load_index()
    if not idx:
        print("✗ 无法加载 canonical 索引")
        return 1

    print("✓ 已加载 canonical 索引")
    queries = ["学生", "奖学金", "答辩", "请假"]
    for q in queries:
        results = builder.test_search(q, k=args.k)
        print(f"\n查询: {q}")
        if not results:
            print("  无结果")
            continue
        for r in results:
            print(f"  排名{r['rank']}: {r['content']}")
            print(f"    来源: {r['metadata'].get('filename')}  chunk: {r['metadata'].get('chunk_index')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


