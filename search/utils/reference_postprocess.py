import os
from functools import lru_cache
from typing import Dict, List, Tuple, Any
import json

from config.settings import similarity_threshold as _default_sim, KB_NAME  # 保留导入以避免未使用警告


DEFAULT_ALIAS_PATH = os.path.join("data", "canonical_index", "alias_mapping.json")


@lru_cache(maxsize=1)
def load_alias_mapping(path: str = DEFAULT_ALIAS_PATH) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


LEVEL_TO_SIM = {
    # 仅用于在未来需要语义折叠时的占位；当前折叠基于 canonical 映射
    "extreme": 0.985,
    "strong": 0.970,
    "medium": 0.940,
    "off": 2.0,  # 关闭
}


def get_duplicate_level() -> str:
    # 从环境变量或默认 extreme 获取折叠级别
    return os.getenv("DUPLICATE_LEVEL", "extreme").lower()


def fold_duplicates(chunks: List[Dict[str, Any]],
                    alias_map: Dict[str, str] = None,
                    duplicate_level: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    对 Top-K 结果进行按 canonical 折叠。
    - 如果存在 alias_map，优先使用 alias → canonical；否则使用 chunk_id 自身作为canonical。
    - duplicate_level 预留用于未来的语义折叠（当前仅按映射折叠）。
    返回：折叠后的列表、统计信息（折叠数量、组大小等）。
    """
    if not chunks:
        return [], {"folded": 0, "groups": 0}

    level = (duplicate_level or get_duplicate_level())
    _ = LEVEL_TO_SIM.get(level, LEVEL_TO_SIM["extreme"])  # 预留未使用
    alias_map = alias_map or load_alias_mapping()

    groups: Dict[str, List[Dict[str, Any]]] = {}

    for item in chunks:
        cid = item.get("chunk_id") or item.get("metadata", {}).get("chunk_id")
        if not cid:
            # 回退：按文本内容聚类的粗糙折叠（可选）
            cid = f"content::{hash(item.get('text') or item.get('content') or '')}"
        canonical = alias_map.get(cid, cid)
        groups.setdefault(canonical, []).append(item)

    folded: List[Dict[str, Any]] = []
    for canonical_id, items in groups.items():
        # 选择得分最高/权重最高/或第一条作为代表
        # 兼容不同结果字段：score、rerank_score、weight
        def key_fn(x):
            return (
                float(x.get("rerank_score", 0.0)),
                float(x.get("score", 0.0)),
                float(x.get("weight", 0.0)),
            )
        items_sorted = sorted(items, key=key_fn, reverse=True)
        rep = dict(items_sorted[0])
        rep["canonical_chunk_id"] = canonical_id
        rep["folded_count"] = len(items)
        folded.append(rep)

    stats = {"folded": len(chunks) - len(folded), "groups": len(groups)}
    return folded, stats


def filter_boilerplate(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 若上游已清理，则此处仅占位；如检测到 metadata.boilerplate=True，可在此过滤
    return [c for c in chunks if not c.get("metadata", {}).get("boilerplate", False)]


def time_filter(chunks: List[Dict[str, Any]], as_of: str = None) -> List[Dict[str, Any]]:
    # 预留：依据 metadata.valid_from/valid_to/is_current 过滤
    if not as_of:
        # 默认优先 is_current
        current = [c for c in chunks if c.get("metadata", {}).get("is_current") is True]
        return current or chunks
    # 如需解析 as_of，可在此实现
    return chunks


