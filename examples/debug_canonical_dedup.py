#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试构建期消岐：
- 对指定目录/文件分块后，报告：
  1) 指纹（精确）重复组
  2) 语义（余弦≥阈值）近重复组

用法：
  python examples/debug_canonical_dedup.py --root ./files --within 0.985 --cross 0.98 --max 50
可选：只分析某个文件
  python examples/debug_canonical_dedup.py --root ./files --only "txt文件/历史上的今天.txt"
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, Any, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processor.canonical_dedup_pipeline import CanonicalDedupPipeline, DedupConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(project_root / "files"))
    p.add_argument("--only", type=str, default=None, help="仅分析该相对路径文件")
    p.add_argument("--within", type=float, default=0.988)
    p.add_argument("--cross", type=float, default=0.983)
    p.add_argument("--max", type=int, default=200, help="最多分析的chunk数（采样）")
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = DedupConfig(files_dir=args.root, output_dir=str(project_root / "data" / "_debug"),
                      write_faiss=False, write_neo4j=False,
                      extreme_threshold_within=args.within, extreme_threshold_cross=args.cross,
                      intra_doc_remove_duplicates=False,
                      chunk_size=args.chunk_size, overlap=args.overlap)
    pipe = CanonicalDedupPipeline(cfg)

    # 读取并分块
    files = pipe._iter_files(args.root)
    if args.only:
        files = [f for f in files if f[0] == args.only]
        if not files:
            print(f"未找到指定文件: {args.only}")
            return 1

    chunks = pipe._create_chunks(files)
    print(f"总chunks: {len(chunks)}")
    # 打印该文件的所有chunk摘要（若 --only 指定）
    if args.only:
        sample = [c for c in chunks if c["filename"] == args.only]
        print(f"{args.only} 的chunk数: {len(sample)}")
        for c in sample[:50]:
            text = c["text"]
            print({
                "idx": c["chunk_index"],
                "len": len(text),
                "fp": pipe._sha1(pipe._normalize_text(text)),
                "head": text[:60].replace("\n", " "),
            })

    # 采样
    chunks = chunks[: args.max]

    # 指纹去重
    kept, alias_exact = pipe._dedup_exact(chunks)
    print(f"精确重复组数: {len(set(alias_exact.values()))}，别名条数: {len(alias_exact)}")
    # 列出发生精确重复的文件来源分布
    if alias_exact:
        inv: Dict[str, List[str]] = {}
        for ali, cano in alias_exact.items():
            inv.setdefault(cano, []).append(ali)
        print("精确重复示例（代表->别名前3个）：")
        for i, (cano, alis) in enumerate(inv.items()):
            if i >= 3:
                break
            print(f"  {cano} <- {alis[:3]}")
    # 打印每组一个代表
    if alias_exact:
        reps = set(alias_exact.values())
        print("精确重复代表样例：", list(reps)[:5])

    # 语义去重
    canonical, alias_sem = pipe._group_extreme_similar(kept)
    print(f"语义近重复组（额外）数: {len(set(alias_sem.values()))}，别名条数: {len(alias_sem)}")

    # 定位某文件内的重复
    if args.only:
        same_file_chunks = [c for c in kept if c["filename"] == args.only]
        if same_file_chunks:
            print(f"文件 {args.only} 分块数: {len(same_file_chunks)}")
        else:
            print(f"文件 {args.only} 分块为空（可能被清理/过短过滤）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


