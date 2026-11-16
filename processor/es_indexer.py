import os
import time
import html
from typing import Dict, Any, Iterable, List, Optional
from elasticsearch import Elasticsearch, helpers

from config.settings import (
    ELASTICSEARCH_URL,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD,
    ES_INDEX_NAME,
)
from processor.file_reader import FileReader
from processor.text_chunker import ChineseTextChunker
from config.settings import FILES_DIR


class ESIndexer:
    """
    Elasticsearch 索引管理与写入器
    - 支持创建索引（优先 IK，失败回退 n-gram）
    - 支持全量/增量写入（以 chunk 为单位）
    - 以 chunk_id 作为 _id，Upsert 写入
    """

    def __init__(self,
                 es_url: str = None,
                 username: str = None,
                 password: str = None,
                 index_name: str = None):
        es_kwargs = {"request_timeout": 30}
        es_user = username or ELASTICSEARCH_USERNAME
        es_pass = password or ELASTICSEARCH_PASSWORD
        if es_user and es_pass:
            es_kwargs["basic_auth"] = (es_user, es_pass)

        self.es = Elasticsearch(es_url or ELASTICSEARCH_URL, **es_kwargs)
        self.index = index_name or ES_INDEX_NAME
        self.pipeline_id = f"{self.index}_normalize"

    def ensure_pipeline(self) -> None:
        """
        创建/更新标准化 Ingest Pipeline：
        - 替换不可见/特殊空白为普通空格或去除：\u00A0, \u200B, \u3000, \u2028, \u2029
        - 合并多余空白为单空格
        - 去除首尾空白
        """
        try:
            body = {
                "description": "Normalize chunk text before indexing",
                "processors": [
                    {"gsub": {"field": "text", "pattern": "\u00A0", "replacement": " "}},
                    {"gsub": {"field": "text", "pattern": "\u200B", "replacement": ""}},
                    {"gsub": {"field": "text", "pattern": "\u3000", "replacement": " "}},
                    {"gsub": {"field": "text", "pattern": "[\u2028\u2029]", "replacement": " "}},
                    {"gsub": {"field": "text", "pattern": "\\s+", "replacement": " "}},
                    {"trim": {"field": "text"}},
                ]
            }
            self.es.ingest.put_pipeline(id=self.pipeline_id, body=body)
        except Exception as e:
            print(f"创建/更新 Ingest Pipeline 失败: {e}")

    # ---------- 索引管理 ----------
    def index_exists(self) -> bool:
        try:
            return self.es.indices.exists(index=self.index).body if hasattr(self.es.indices.exists(self.index), 'body') else self.es.indices.exists(self.index)
        except Exception:
            # 兼容不同客户端版本的返回
            try:
                return self.es.indices.exists(index=self.index)
            except Exception:
                return False

    def delete_index(self) -> None:
        if self.index_exists():
            self.es.indices.delete(index=self.index, ignore=[404])

    def _try_create_with_ik(self) -> bool:
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "cn": {"tokenizer": "ik_max_word"}
                    }
                },
                "index.default_pipeline": self.pipeline_id
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "cn",
                        "search_analyzer": "cn",
                        "fields": {"raw": {"type": "keyword"}}
                    },
                    "filename": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "source_path": {"type": "keyword"},
                    "source_type": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "entities": {"type": "keyword"},
                    "keywords": {"type": "keyword"},
                    "length": {"type": "integer"}
                }
            }
        }
        try:
            self.es.indices.create(index=self.index, body=mapping)
            return True
        except Exception as e:
            msg = str(e)
            if "ik_max_word" in msg or "analyzer" in msg:
                return False
            # 其它错误透传
            raise

    def _create_with_ngram(self) -> None:
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "cn_ngram": {"tokenizer": "ngram_tokenizer"}
                    },
                    "tokenizer": {
                        "ngram_tokenizer": {"type": "ngram", "min_gram": 2, "max_gram": 3}
                    }
                },
                "index.default_pipeline": self.pipeline_id
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "cn_ngram",
                        "search_analyzer": "standard",
                        "fields": {"raw": {"type": "keyword"}}
                    },
                    "filename": {"type": "keyword"},
                    "chunk_index": {"type": "integer"}
                }
            }
        }
        self.es.indices.create(index=self.index, body=mapping)

    def create_index(self, recreate: bool = False) -> str:
        if recreate and self.index_exists():
            self.delete_index()

        if self.index_exists():
            return "exists"

        # 先确保标准化 Pipeline 存在
        self.ensure_pipeline()

        # 优先 IK，失败回退 n-gram
        try:
            created = self._try_create_with_ik()
            if not created:
                self._create_with_ngram()
                return "ngram"
            return "ik"
        except Exception:
            # 最后的兜底
            self._create_with_ngram()
            return "ngram"

    # ---------- 写入 ----------
    def _iter_chunks(self, from_path: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        """从文件系统读取并切分为 chunks。"""
        root = from_path or str(FILES_DIR)
        reader = FileReader(root)
        chunker = ChineseTextChunker()
        for rel_path, content in reader.read_files(recursive=True):
            try:
                tokens_list = chunker.chunk_text(content)
                for i, tokens in enumerate(tokens_list):
                    text = ''.join(tokens).strip()
                    if not text or len(text) <= 5:
                        continue
                    doc = {
                        "chunk_id": f"{rel_path}_chunk_{i}",
                        "text": text,
                        "filename": rel_path,
                        "chunk_index": i,
                        "length": len(text)
                    }
                    yield doc
            except Exception:
                continue

    def _to_actions(self, docs: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for d in docs:
            yield {
                "_op_type": "index",
                "_index": self.index,
                "_id": d.get("chunk_id"),
                "_source": d,
            }

    def index_from_path(self, from_path: Optional[str] = None, batch_size: int = 500) -> Dict[str, Any]:
        """
        从目录构建索引。返回统计信息。
        """
        start = time.time()
        ok, fail = 0, 0
        last_err = None
        try:
            actions = self._to_actions(self._iter_chunks(from_path))
            for ok_count, errors in helpers.streaming_bulk(
                self.es, actions, chunk_size=batch_size, raise_on_error=False
            ):
                if ok_count:
                    ok += 1
                if errors:
                    fail += len(errors)
                    last_err = errors[-1]
        except Exception as e:
            last_err = str(e)

        took = int((time.time() - start) * 1000)
        return {"ok": ok, "fail": fail, "took_ms": took, "last_error": last_err}


def escape_except_mark(text: str) -> str:
    """只允许 <mark> 标签，其余全部转义。"""
    if text is None:
        return ""
    placeholder_open = "__MARK_OPEN__"
    placeholder_close = "__MARK_CLOSE__"
    text = text.replace("<mark>", placeholder_open).replace("</mark>", placeholder_close)
    text = html.escape(text)
    return text.replace(placeholder_open, "<mark>").replace(placeholder_close, "</mark>")


