import os
import re
import json
import math
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from langchain_core.documents import Document

from processor.text_chunker import ChineseTextChunker
from model.get_models import get_embeddings_model

# 可选：仅当需要写入Neo4j时再加载，避免在无图环境下报错
try:
    from graph.core import connection_manager
except Exception:
    connection_manager = None  # 延迟使用


@dataclass
class DedupConfig:
    files_dir: str
    output_dir: str = "data/canonical_index"
    faiss_index_name: str = "canonical_faiss"
    extreme_threshold_within: float = 0.988
    extreme_threshold_cross: float = 0.983
    use_cross_encoder_confirm: bool = False  # 预留，将来接入
    write_faiss: bool = True
    write_neo4j: bool = True
    alias_mapping_name: str = "alias_mapping.json"
    batch_size: int = 64
    max_text_length: int = 20000
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    intra_doc_remove_duplicates: bool = True


class CanonicalDedupPipeline:
    """
    构建期消岐流水线：
      - 锅炉化文本清理
      - 语言检测与标准化
      - 极端语义相似（跨语言）去重 → canonical/alias
      - 仅写入 canonical 到 FAISS & Neo4j；alias 建立映射

    该模块尽量做到“最小侵入”，可被独立调用。
    """

    def __init__(self, config: DedupConfig):
        self.config = config
        if self.config.chunk_size or self.config.overlap:
            # 使用覆盖的分块参数
            from config.settings import CHUNK_SIZE, OVERLAP
            chunk_size = self.config.chunk_size or CHUNK_SIZE
            overlap = self.config.overlap or OVERLAP
            self.chunker = ChineseTextChunker(chunk_size=chunk_size, overlap=overlap)
        else:
            self.chunker = ChineseTextChunker()
        self.embeddings = get_embeddings_model()
        self.alias_map: Dict[str, str] = {}
        self.equivalent_groups: Dict[str, List[str]] = {}
        self.canonical_set: set = set()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 文本预处理
    # -----------------------------
    @staticmethod
    def _to_half_width(s: str) -> str:
        result = []
        for char in s:
            code = ord(char)
            if code == 0x3000:  # 全角空格
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            result.append(chr(code))
        return "".join(result)

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        # 全半角统一 + 去除重复空白 + 统一标点周围空格
        t = CanonicalDedupPipeline._to_half_width(text)
        t = t.replace("\r", "\n")
        t = re.sub(r"\s+", " ", t)
        # 去掉两端空白
        t = t.strip()
        return t

    @staticmethod
    def _is_boilerplate_line(line: str) -> bool:
        l = line.strip()
        if not l:
            return True
        # 常见锅炉化特征：页码/版权/目录/网址/邮箱/统一备案/扫码
        patterns = [
            r"^第?\s*\d+\s*页.*$",
            r"^\d+\s*/\s*\d+$",
            r"^版权所有.*$",
            r"^版权.*$",
            r"^目录$",
            r"^contents?$",
            r"^http(s)?://",
            r"@\w+\.\w+",
            r"^扫码(下载|查看|关注).*$",
            r"^本文件由.*生成.*$",
            r"^华东理工大学\s*(校徽|标识)?$",
        ]
        for p in patterns:
            if re.search(p, l, flags=re.IGNORECASE):
                return True
        # 过短的重复性行
        if len(l) <= 2:
            return True
        return False

    def _remove_boilerplate(self, text: str) -> str:
        # 行级过滤
        lines = re.split(r"[\n\r]+", text or "")
        kept = [ln for ln in lines if not self._is_boilerplate_line(ln)]
        cleaned = "\n".join(kept)
        # 再做一次规范化
        return self._normalize_text(cleaned)

    def _dedupe_intra_document(self, text: str) -> str:
        """
        在单文档内去除重复段落/块：
        - 先按空行分段；若空行较少，则按单行分段
        - 以规范化+SHA1 指纹去重，保留首次出现的块
        """
        if not text:
            return text
        # 优先按空行分段，否则退化为按行
        blocks = re.split(r"\n\n+", text)
        if len(blocks) <= 2:
            blocks = text.split("\n")
        seen: set = set()
        kept: List[str] = []
        for b in blocks:
            nb = self._normalize_text(b)
            if not nb:
                continue
            fp = self._sha1(nb)
            if fp in seen:
                continue
            seen.add(fp)
            kept.append(b)
        return "\n\n".join(kept)

    @staticmethod
    def _detect_language_simple(text: str) -> str:
        if not text:
            return "unknown"
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        if cjk > max(10, latin * 2):
            return "zh"
        if latin > max(10, cjk * 2):
            return "en"
        return "mixed"

    # -----------------------------
    # 相似与分组
    # -----------------------------
    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _sha1(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    def _choose_canonical(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 代表选择：信息量大 > 来源权威（暂无则忽略）> 语言更自然（倾向 zh）> 首次
        sorted_group = sorted(
            group,
            key=lambda x: (
                len(x.get("text", "")),
                1 if x.get("language") == "zh" else 0,
            ),
            reverse=True,
        )
        return sorted_group[0]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        # langchain Embeddings 接口：embed_documents 批量
        vectors: List[List[float]] = []
        bs = self.config.batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            vecs = self.embeddings.embed_documents(batch)
            vectors.extend(vecs)
        return vectors

    def _group_extreme_similar(self, items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        if not items:
            return [], {}

        texts = [it["text"] for it in items]
        vectors = self._embed_texts(texts)

        # 简单的 canopy-style 聚类：
        assigned = [False] * len(items)
        canonical_items: List[Dict[str, Any]] = []
        alias_map: Dict[str, str] = {}

        # 先按文档粒度分两档阈值
        def threshold(i: int, j: int) -> float:
            same_file = items[i].get("filename") == items[j].get("filename")
            return self.config.extreme_threshold_within if same_file else self.config.extreme_threshold_cross

        for i in range(len(items)):
            if assigned[i]:
                continue
            # 以 i 为种子，找到与之极端相似的集合
            group_idx = [i]
            for j in range(i + 1, len(items)):
                if assigned[j]:
                    continue
                sim = self._cosine(vectors[i], vectors[j])
                if sim >= threshold(i, j):
                    group_idx.append(j)
            # 构造分组
            group = [items[k] for k in group_idx]
            canonical = self._choose_canonical(group)
            canonical_id = canonical["chunk_id"]
            canonical_items.append(canonical)
            for k in group_idx:
                assigned[k] = True
                cid = items[k]["chunk_id"]
                if cid != canonical_id:
                    alias_map[cid] = canonical_id
        return canonical_items, alias_map

    # -----------------------------
    # IO 与写入
    # -----------------------------
    def _iter_files(self, root: str) -> List[Tuple[str, str]]:
        root_path = Path(root)
        results: List[Tuple[str, str]] = []
        for p in root_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}:  # 简化：先处理文本、md、pdf（pdf需前置抽取）
                try:
                    # 此处只处理纯文本；pdf/word 的抽取应在上游完成
                    if p.suffix.lower() == ".pdf":
                        continue
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    results.append((str(p.relative_to(root_path)), content))
                except Exception:
                    continue
        return results

    def _create_chunks(self, file_contents: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for filename, content in file_contents:
            # 先粗清理 →（可选）文档内重复去除 → 再切分 → 每块行级清理
            content = self._normalize_text(content)[: self.config.max_text_length]
            if self.config.intra_doc_remove_duplicates:
                content = self._dedupe_intra_document(content)
            token_chunks = self.chunker.chunk_text(content)
            # 针对同一文件的频繁短行可在此做一轮“重复短行黑名单”增强（简单实现略）
            for i, tokens in enumerate(token_chunks):
                text = "".join(tokens).strip()
                if not text or len(text) <= 10:
                    continue
                text = self._remove_boilerplate(text)
                if not text or len(text) <= 10:
                    continue
                lang = self._detect_language_simple(text)
                chunk_id = f"{filename}_chunk_{i}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "filename": filename,
                    "chunk_index": i,
                    "language": lang,
                    "length": len(text),
                    "fingerprint": self._sha1(text),
                })
        return chunks

    def _dedup_exact(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """基于规范化文本的指纹做精确去重，返回保留的列表与 alias 映射。"""
        fp_to_items: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            fp = c.get("fingerprint") or self._sha1(c.get("text", ""))
            fp_to_items.setdefault(fp, []).append(c)

        kept: List[Dict[str, Any]] = []
        alias_map: Dict[str, str] = {}
        for fp, items in fp_to_items.items():
            if not items:
                continue
            if len(items) == 1:
                kept.append(items[0])
                continue
            # 多个完全一致，选择一个代表（长度一样时按首次）
            rep = self._choose_canonical(items)
            rep_id = rep["chunk_id"]
            kept.append(rep)
            for it in items:
                if it["chunk_id"] != rep_id:
                    alias_map[it["chunk_id"]] = rep_id
        return kept, alias_map

    def _write_faiss(self, canonical: List[Dict[str, Any]]) -> Optional[str]:
        if not self.config.write_faiss:
            return None
        try:
            from langchain_community.vectorstores import FAISS
        except Exception as e:
            print(f"[FAISS] 依赖不可用或未安装: {e}")
            return None

        texts = [c["text"] for c in canonical]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in canonical]
        if not texts:
            print("[FAISS] 无 canonical 文本，跳过写入")
            return None

        print(f"[FAISS] 正在写入 {len(texts)} 条 canonical...")
        vs = FAISS.from_texts(texts, self.embeddings, metadatas=metas)
        idx_dir = self.output_dir / self.config.faiss_index_name
        vs.save_local(str(idx_dir))
        print(f"[FAISS] 已保存到 {idx_dir}")
        return str(idx_dir)

    def _write_neo4j(self, canonical: List[Dict[str, Any]]) -> int:
        if not self.config.write_neo4j:
            return 0
        if connection_manager is None:
            print("[Neo4j] 未检测到图连接管理器，跳过写入")
            return 0
        graph = connection_manager.get_connection()
        # 仅创建 __Chunk__ 节点，保持最小侵入；与文档关系可后续处理
        query = (
            "UNWIND $rows AS row "
            "MERGE (c:`__Chunk__` {id: row.chunk_id}) "
            "ON CREATE SET c.text = row.text, c.fileName = row.filename, c.position = row.chunk_index, c.length = row.length, c.language = row.language "
            "ON MATCH SET c.text = row.text, c.fileName = row.filename, c.position = row.chunk_index, c.length = row.length, c.language = row.language"
        )
        rows = [{
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "length": c["length"],
            "language": c.get("language", "unknown"),
        } for c in canonical]
        if not rows:
            return 0
        graph.query(query, params={"rows": rows})
        print(f"[Neo4j] 写入 __Chunk__ 节点: {len(rows)}")
        return len(rows)

    def _save_alias_mapping(self, mapping: Dict[str, str]) -> str:
        out = self.output_dir / self.config.alias_mapping_name
        with out.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"[Alias] 映射关系已保存: {out}")
        return str(out)

    # -----------------------------
    # 主流程
    # -----------------------------
    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        files = self._iter_files(self.config.files_dir)
        if not files:
            print(f"[Pipeline] 未在 {self.config.files_dir} 发现可处理文件")
            return {"files": 0, "canonical": 0, "aliases": 0}

        chunks = self._create_chunks(files)
        if not chunks:
            print("[Pipeline] 无有效分块")
            return {"files": len(files), "canonical": 0, "aliases": 0}

        # 先做精确指纹去重，再做极端相似聚合
        exact_kept, alias_map_exact = self._dedup_exact(chunks)
        canonical, alias_map_sem = self._group_extreme_similar(exact_kept)
        alias_map = {**alias_map_exact, **alias_map_sem}

        # 写入与落盘
        self._save_alias_mapping(alias_map)
        faiss_path = self._write_faiss(canonical)
        neo4j_count = self._write_neo4j(canonical)

        elapsed = time.time() - t0
        stats = {
            "files": len(files),
            "chunks": len(chunks),
            "canonical": len(canonical),
            "aliases": len(alias_map),
            "faiss_index_path": faiss_path,
            "neo4j_nodes": neo4j_count,
            "elapsed_sec": round(elapsed, 2),
        }
        print(f"[Pipeline] 完成：{stats}")
        return stats


def run_canonical_build(
    files_dir: str,
    output_dir: str = "data/canonical_index",
    write_faiss: bool = True,
    write_neo4j: bool = True,
    extreme_within: float = 0.988,
    extreme_cross: float = 0.983,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Dict[str, Any]:
    config = DedupConfig(
        files_dir=files_dir,
        output_dir=output_dir,
        write_faiss=write_faiss,
        write_neo4j=write_neo4j,
        extreme_threshold_within=extreme_within,
        extreme_threshold_cross=extreme_cross,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    pipeline = CanonicalDedupPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    # 最小化命令行入口
    base = Path(__file__).parent.parent
    files_dir = str(base / "files")
    stats = run_canonical_build(files_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


