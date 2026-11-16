#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检索阶段折叠最小化验证：
- 从 canonical 索引检索 Top-K
- 加载 alias 映射
- 按 canonical 折叠，并输出折叠统计

运行：python examples/test_reference_folding.py --index ./data/canonical_index/canonical_faiss --k 5
"""

import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processor.faiss_indexer import FAISSIndexBuilder
from postprocess.reference_postprocess import load_alias_mapping, fold_duplicates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=str, default=str(project_root / "data" / "canonical_index" / "canonical_faiss"))
    p.add_argument("--k", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    builder = FAISSIndexBuilder(cache_dir=args.index)
    idx = builder.load_index()
    if not idx:
        print("✗ 无法加载 canonical 索引")
        return 1

    alias_map = load_alias_mapping()
    queries = ["学生", "奖学金", "处分"]
    for q in queries:
        results = builder.test_search(q, k=args.k)
        folded, stats = fold_duplicates(results, alias_map=alias_map, duplicate_level="extreme")
        print(f"\n查询: {q}")
        print(f"  原始Top{args.k}: {len(results)}，折叠后: {len(folded)}，折叠数: {stats['folded']}")
        for r in folded:
            print(f"  排名{r['rank'] if 'rank' in r else '?'}: {r.get('content','')[:80]}...")
            print(f"    canonical: {r.get('canonical_chunk_id')}  组内数量: {r.get('folded_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


