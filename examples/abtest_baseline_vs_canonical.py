#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比 baseline 索引（现有 data/faiss_indices）与 canonical 索引（data/canonical_index/canonical_faiss）的AB测试。
指标：
- TopK 命中是否为空
- 去重率（同一文件同一chunk_id出现的重复条数；简化：按内容哈希折叠的比例）
- 平均内容长度（可选）

运行：
  python examples/abtest_baseline_vs_canonical.py --baseline ./data/faiss_indices --canonical ./data/canonical_index/canonical_faiss --k 5
"""

import sys
from pathlib import Path
import argparse
import hashlib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processor.faiss_indexer import FAISSIndexBuilder


def _content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=str, default=str(project_root / "data" / "faiss_indices"))
    p.add_argument("--canonical", type=str, default=str(project_root / "data" / "canonical_index" / "canonical_faiss"))
    p.add_argument("--k", type=int, default=5)
    return p.parse_args()


def _search(builder: FAISSIndexBuilder, q: str, k: int):
    idx = builder.load_index()
    if not idx:
        return []
    return builder.test_search(q, k=k)


def _dup_rate(results):
    if not results:
        return 0.0
    hashes = [_content_hash(r.get("content", "")) for r in results]
    unique = len(set(hashes))
    return 1.0 - unique / max(1, len(hashes))


def main() -> int:
    args = parse_args()
    baseline = FAISSIndexBuilder(cache_dir=args.baseline)
    canonical = FAISSIndexBuilder(cache_dir=args.canonical)

    queries = ["学生", "奖学金", "处分", "课程", "答辩", "请假"]
    print("查询\t库\t重复率\t返回数")
    for q in queries:
        b = _search(baseline, q, args.k)
        c = _search(canonical, q, args.k)
        print(f"{q}\tbaseline\t{_dup_rate(b):.2f}\t{len(b)}")
        print(f"{q}\tcanonical\t{_dup_rate(c):.2f}\t{len(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


