import os
from functools import lru_cache
from typing import Dict, List, Tuple, Any
import json


DEFAULT_ALIAS_PATH = os.path.join("data", "canonical_index", "alias_mapping.json")


@lru_cache(maxsize=1)
def load_alias_mapping(path: str = DEFAULT_ALIAS_PATH) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


LEVEL_TO_SIM = {
    # 预留：未来如需在检索端做语义折叠时可使用
    "extreme": 0.985,
    "strong": 0.970,
    "medium": 0.940,
    "off": 2.0,
}


def get_duplicate_level() -> str:
    return os.getenv("DUPLICATE_LEVEL", "extreme").lower()


def fold_duplicates(chunks: List[Dict[str, Any]],
                    alias_map: Dict[str, str] = None,
                    duplicate_level: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not chunks:
        return [], {"folded": 0, "groups": 0}

    _ = LEVEL_TO_SIM.get((duplicate_level or get_duplicate_level()), LEVEL_TO_SIM["extreme"])  # 未使用占位
    alias_map = alias_map or load_alias_mapping()

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in chunks:
        cid = item.get("chunk_id") or item.get("metadata", {}).get("chunk_id")
        if not cid:
            cid = f"content::{hash(item.get('text') or item.get('content') or '')}"
        canonical = alias_map.get(cid, cid)
        groups.setdefault(canonical, []).append(item)

    folded: List[Dict[str, Any]] = []
    for canonical_id, items in groups.items():
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
    return [c for c in chunks if not c.get("metadata", {}).get("boilerplate", False)]


def time_filter(chunks: List[Dict[str, Any]], as_of: str = None) -> List[Dict[str, Any]]:
    if not as_of:
        current = [c for c in chunks if c.get("metadata", {}).get("is_current") is True]
        return current or chunks
    return chunks


