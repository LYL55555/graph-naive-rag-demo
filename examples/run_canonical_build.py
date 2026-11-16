#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
独立运行的构建期消岐示例：
- 清理锅炉化文本
- 语言检测与标准化
- 极端相似语义去重（canonical/alias）
- 仅写 canonical 到 FAISS 与 Neo4j

运行：
  python examples/run_canonical_build.py --files ./files --out ./data/canonical_index \
      --faiss --neo4j
"""

import sys
import os
from pathlib import Path
import argparse
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from processor.canonical_dedup_pipeline import run_canonical_build


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files", type=str, default=str(project_root / "files"))
    p.add_argument("--out", type=str, default=str(project_root / "data" / "canonical_index"))
    # 显式开启/关闭 FAISS 写入
    g1 = p.add_mutually_exclusive_group(required=False)
    g1.add_argument("--faiss", dest="faiss", action="store_true", help="启用写入FAISS（默认开启）")
    g1.add_argument("--no-faiss", dest="faiss", action="store_false", help="禁用写入FAISS")
    # 显式开启/关闭 Neo4j 写入
    g2 = p.add_mutually_exclusive_group(required=False)
    g2.add_argument("--neo4j", dest="neo4j", action="store_true", help="启用写入Neo4j（默认开启）")
    g2.add_argument("--no-neo4j", dest="neo4j", action="store_false", help="禁用写入Neo4j")
    p.add_argument("--within", type=float, default=0.988, help="同文档极端相似阈值")
    p.add_argument("--cross", type=float, default=0.983, help="跨文档极端相似阈值")
    p.add_argument("--chunk-size", type=int, default=None, help="覆盖构建期分块大小（token数）")
    p.add_argument("--overlap", type=int, default=None, help="覆盖分块重叠大小")
    p.set_defaults(faiss=True, neo4j=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stats = run_canonical_build(
        files_dir=args.files,
        output_dir=args.out,
        write_faiss=args.faiss,
        write_neo4j=args.neo4j,
        extreme_within=args.within,
        extreme_cross=args.cross,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"执行失败: {e}")
        raise


