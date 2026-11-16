import os
import time
import json
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import concurrent.futures
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from processor.file_reader import FileReader
from processor.text_chunker import ChineseTextChunker
from model.get_models import get_embeddings_model
from config.settings import FILES_DIR, CHUNK_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS


class FAISSIndexBuilder:
    """
    FAISS索引构建器，用于构建基于文档chunks的向量索引
    
    主要功能：
    1. 从files目录读取文档
    2. 使用ChineseTextChunker进行文本分块
    3. 生成embeddings并构建FAISS索引
    4. 保存和加载索引
    """
    
    def __init__(self, 
                 files_dir: str = FILES_DIR,
                 cache_dir: str = None,
                 batch_size: int = None,
                 max_workers: int = None):
        """
        初始化FAISS索引构建器
        
        Args:
            files_dir: 文档目录路径
            cache_dir: 索引存储目录（持久化），如果为None则使用项目根目录下的data/faiss_indices
            batch_size: 批处理大小
            max_workers: 并行工作线程数
        """
        self.files_dir = files_dir
        
        # 如果没有指定cache_dir，使用项目根目录下的data/faiss_indices
        if cache_dir is None:
            # 获取项目根目录（processor目录的上级目录）
            project_root = Path(__file__).parent.parent
            self.cache_dir = str(project_root / "data" / "faiss_indices")
        else:
            self.cache_dir = cache_dir
            
        self.batch_size = batch_size or CHUNK_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 确保缓存目录存在
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.file_reader = FileReader(self.files_dir)
        self.text_chunker = ChineseTextChunker()
        self.embeddings = get_embeddings_model()
        
        # 性能监控
        self.stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "build_time": 0,
            "errors": []
        }
        
        print(f"FAISS索引构建器初始化完成")
        print(f"文档目录: {self.files_dir}")
        print(f"缓存目录: {self.cache_dir}")
        print(f"批处理大小: {self.batch_size}")
    
    def build_index(self, 
                   file_extensions: Optional[List[str]] = None,
                   force_rebuild: bool = False) -> Optional[FAISS]:
        """
        构建FAISS索引
        
        Args:
            file_extensions: 要处理的文件扩展名列表，默认处理所有支持的格式
            force_rebuild: 是否强制重建索引，即使已存在
            
        Returns:
            FAISS: 构建好的FAISS索引对象
        """
        start_time = time.time()
        
        # 检查是否已存在索引
        if not force_rebuild and self._index_exists():
            print("发现已存在的FAISS索引，尝试加载...")
            index = self.load_index()
            if index:
                print("成功加载已存在的FAISS索引")
                return index
            else:
                print("加载失败，将重新构建索引")
        
        print("开始构建FAISS索引...")
        
        try:
            # 1. 读取文档
            print("步骤1: 读取文档...")
            file_contents = self.file_reader.read_files(
                file_extensions=file_extensions, 
                recursive=True
            )
            
            if not file_contents:
                print("警告: 没有找到任何文档文件")
                return None
            
            self.stats["files_processed"] = len(file_contents)
            print(f"成功读取 {len(file_contents)} 个文档")
            
            # 2. 生成chunks
            print("步骤2: 生成文本块...")
            documents, metadata = self._create_chunks(file_contents)
            
            if not documents:
                print("警告: 没有生成任何文本块")
                return None
            
            self.stats["chunks_created"] = len(documents)
            print(f"成功生成 {len(documents)} 个文本块")
            
            # 3. 构建FAISS索引
            print("步骤3: 构建FAISS向量索引...")
            faiss_index = self._build_faiss_index(documents, metadata)
            
            if not faiss_index:
                print("错误: FAISS索引构建失败")
                return None
            
            # 4. 保存索引
            print("步骤4: 保存FAISS索引...")
            success = self._save_index(faiss_index, metadata)
            
            if success:
                self.stats["build_time"] = time.time() - start_time
                print(f"FAISS索引构建完成!")
                print(f"构建统计:")
                print(f"  - 处理文档: {self.stats['files_processed']} 个")
                print(f"  - 生成chunks: {self.stats['chunks_created']} 个")
                print(f"  - 构建时间: {self.stats['build_time']:.2f} 秒")
                
                return faiss_index
            else:
                print("错误: 索引保存失败")
                return None
                
        except Exception as e:
            error_msg = f"构建FAISS索引时出错: {str(e)}"
            print(error_msg)
            self.stats["errors"].append(error_msg)
            return None
    
    def _read_specific_files(self, file_paths: List[str]) -> List[Tuple[str, str]]:
        """
        仅读取指定的相对路径文件（相对于 files_dir），返回 (relative_filename, content)
        """
        results: List[Tuple[str, str]] = []
        for rel in file_paths:
            try:
                abs_path = Path(self.files_dir) / rel
                if not abs_path.exists():
                    # 再尝试仅文件名匹配于根目录（兼容外部传入纯文件名的情况）
                    abs_path = Path(self.files_dir) / Path(rel).name
                if not abs_path.exists():
                    print(f"[FAISS] 指定文件不存在: {rel}")
                    continue
                ext = abs_path.suffix.lower()
                content = None
                # 复用 FileReader 的具体读取实现
                if ext == '.txt':
                    content = self.file_reader._read_txt(str(abs_path))
                elif ext == '.pdf':
                    content = self.file_reader._read_pdf(str(abs_path))
                elif ext == '.md':
                    content = self.file_reader._read_markdown(str(abs_path))
                elif ext == '.docx':
                    content = self.file_reader._read_docx(str(abs_path))
                elif ext == '.doc':
                    content = self.file_reader._read_doc(str(abs_path))
                elif ext == '.csv':
                    content = self.file_reader._read_csv(str(abs_path))
                elif ext == '.json':
                    content = self.file_reader._read_json(str(abs_path))
                elif ext in ('.yaml', '.yml'):
                    content = self.file_reader._read_yaml(str(abs_path))
                else:
                    # 不支持的扩展名，跳过
                    print(f"[FAISS] 跳过不支持的文件类型: {rel}")
                    continue
                if content is not None:
                    # 保留相对路径（相对于 files_dir）
                    try:
                        rel_stable = str(abs_path.relative_to(self.files_dir))
                    except Exception:
                        rel_stable = abs_path.name
                    results.append((rel_stable, content))
            except Exception as e:
                print(f"[FAISS] 读取文件失败 {rel}: {e}")
        return results

    def _create_chunks_for_files(self, file_paths: List[str]) -> Tuple[List[str], List[Dict]]:
        """
        针对给定文件列表创建 chunks（不会扫描全目录）
        返回 documents, metadata
        """
        documents: List[str] = []
        metadata: List[Dict] = []
        file_contents = self._read_specific_files(file_paths)
        for filename, content in file_contents:
            try:
                token_chunks = self.text_chunker.chunk_text(content)
                for i, token_chunk in enumerate(token_chunks):
                    chunk_text = ''.join(token_chunk).strip()
                    if chunk_text and len(chunk_text) > 10:
                        documents.append(chunk_text)
                        metadata.append({
                            "filename": filename,
                            "chunk_id": f"{filename}_chunk_{i}",
                            "chunk_index": i,
                            "chunk_length": len(chunk_text),
                            "token_count": len(token_chunk)
                        })
            except Exception as e:
                print(f"[FAISS] 为文件 {filename} 分块失败: {e}")
        return documents, metadata

    def update_index(self, changes: Dict[str, List[str]]) -> Optional[FAISS]:
        """
        增量更新 FAISS 索引：仅对新增/修改文件新增向量，对修改/删除文件移除旧向量。

        Args:
            changes: {"added": [...], "modified": [...], "deleted": [...]} 相对 files_dir 的路径

        Returns:
            更新后的 FAISS 索引对象；如果失败返回 None
        """
        added = list(set(changes.get("added", []) or []))
        modified = list(set(changes.get("modified", []) or []))
        deleted = list(set(changes.get("deleted", []) or []))

        # 尝试加载现有索引；如果不存在且没有新内容可建，则返回 None
        vs: Optional[FAISS] = self.load_index()
        if vs is None and not (added or modified):
            print("[FAISS] 无可更新内容且不存在索引，跳过")
            return None

        # 如果没有索引但有新增/修改，则只基于这些构建一个新索引
        if vs is None:
            print("[FAISS] 未发现索引，将基于新增/修改文件构建一个新索引")
            docs_texts, docs_meta = self._create_chunks_for_files(added + modified)
            if not docs_texts:
                print("[FAISS] 没有可用于构建向量的文本")
                return None
            try:
                docs = []
                for i, text in enumerate(docs_texts):
                    docs.append(Document(page_content=text, metadata=docs_meta[i]))
                vs = FAISS.from_documents(documents=docs, embedding=self.embeddings)
                vs.save_local(self.cache_dir)
                print("[FAISS] 已用变更集构建新索引并保存")
                return vs
            except Exception as e:
                print(f"[FAISS] 基于变更集构建索引失败: {e}")
                return None

        # 收集需要移除的文档 id（按 filename 命中修改/删除集合）
        to_remove_ids: List[str] = []
        try:
            # LangChain FAISS: docstore._dict: id -> Document
            all_ids = list(getattr(vs, "docstore")._dict.keys())
            for _id in all_ids:
                try:
                    doc = vs.docstore._dict.get(_id)
                    meta = getattr(doc, 'metadata', {}) or {}
                    fn = meta.get('filename') or meta.get('fileName')
                    if fn and (fn in modified or fn in deleted):
                        to_remove_ids.append(_id)
                except Exception:
                    continue
        except Exception as e:
            print(f"[FAISS] 枚举索引中文档失败: {e}")
            to_remove_ids = []

        # 先删除被修改/删除文件对应的向量
        if to_remove_ids:
            removed = False
            try:
                if hasattr(vs, 'delete'):
                    vs.delete(ids=to_remove_ids)
                    removed = True
                else:
                    # 旧版本不支持 delete，降级为重建 docstore + index（仅保留其余文档）
                    print("[FAISS] 当前向量库不支持 delete(ids)，将进行轻量重建以移除指定文档")
                    kept_docs = []
                    kept_ids = []
                    for _id, doc in vs.docstore._dict.items():
                        if _id not in to_remove_ids:
                            kept_docs.append(doc)
                            kept_ids.append(_id)
                    # 重新初始化：用保留文档的文本进行 add_texts（需重新计算嵌入）
                    # 为避免复杂的底层操作，这里选择正确性优先的保守实现
                    texts = [d.page_content for d in kept_docs]
                    metas = [d.metadata for d in kept_docs]
                    new_vs = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metas)
                    vs = new_vs
                    removed = True
            except Exception as e:
                print(f"[FAISS] 删除旧向量失败: {e}")

            if removed:
                try:
                    vs.save_local(self.cache_dir)
                except Exception as e:
                    print(f"[FAISS] 保存删除后的索引失败: {e}")

        # 新增/修改 → 添加新向量
        delta_files = list(set(added + modified))
        if delta_files:
            texts, metas = self._create_chunks_for_files(delta_files)
            if texts:
                try:
                    # 使用稳定 id 便于后续精确删除（chunk::<chunk_id>）
                    ids = []
                    for m in metas:
                        cid = m.get('chunk_id') or f"{m.get('filename','unknown')}::{m.get('chunk_index','0')}"
                        ids.append(f"chunk::{cid}")
                    if hasattr(vs, 'add_texts'):
                        vs.add_texts(texts=texts, metadatas=metas, ids=ids)
                    else:
                        # 回退：构造 Document 再 add_documents
                        docs = [Document(page_content=t, metadata=metas[i]) for i, t in enumerate(texts)]
                        vs.add_documents(docs)
                    vs.save_local(self.cache_dir)
                except Exception as e:
                    print(f"[FAISS] 增量添加向量失败: {e}")

        return vs

    def _create_chunks(self, file_contents: List[Tuple[str, str]]) -> Tuple[List[str], List[Dict]]:
        """
        从文档内容创建文本块
        
        Args:
            file_contents: 文件名和内容的元组列表
            
        Returns:
            Tuple[List[str], List[Dict]]: 文本块列表和元数据列表
        """
        documents = []
        metadata = []
        
        for filename, content in file_contents:
            try:
                # 使用ChineseTextChunker进行分块
                token_chunks = self.text_chunker.chunk_text(content)
                
                # 将token列表转换为字符串
                for i, token_chunk in enumerate(token_chunks):
                    # 将tokens重新组合为文本
                    chunk_text = ''.join(token_chunk).strip()
                    
                    if chunk_text and len(chunk_text) > 10:  # 过滤过短的块
                        documents.append(chunk_text)
                        metadata.append({
                            "filename": filename,
                            "chunk_id": f"{filename}_chunk_{i}",
                            "chunk_index": i,
                            "chunk_length": len(chunk_text),
                            "token_count": len(token_chunk)
                        })
                        
            except Exception as e:
                error_msg = f"处理文件 {filename} 时出错: {str(e)}"
                print(error_msg)
                self.stats["errors"].append(error_msg)
                continue
        
        return documents, metadata
    
    def _build_faiss_index(self, documents: List[str], metadata: List[Dict]) -> Optional[FAISS]:
        """
        构建FAISS向量索引
        
        Args:
            documents: 文本块列表
            metadata: 元数据列表
            
        Returns:
            FAISS: 构建的FAISS索引
        """
        try:
            # 创建Document对象列表
            docs = []
            for i, text in enumerate(documents):
                doc = Document(
                    page_content=text,
                    metadata=metadata[i] if i < len(metadata) else {}
                )
                docs.append(doc)
            
            print(f"开始为 {len(docs)} 个文档生成embeddings...")
            
            # 使用LangChain的FAISS.from_documents创建索引
            # 这会自动处理embedding生成和索引构建
            faiss_index = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            self.stats["embeddings_generated"] = len(docs)
            print(f"成功生成 {len(docs)} 个embeddings并构建FAISS索引")
            
            return faiss_index
            
        except Exception as e:
            error_msg = f"构建FAISS索引时出错: {str(e)}"
            print(error_msg)
            self.stats["errors"].append(error_msg)
            return None
    
    def _save_index(self, faiss_index: FAISS, metadata: List[Dict]) -> bool:
        """
        保存FAISS索引和元数据
        
        Args:
            faiss_index: FAISS索引对象
            metadata: 元数据列表
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 保存FAISS索引
            faiss_index.save_local(self.cache_dir)
            
            # 保存元数据
            metadata_path = os.path.join(self.cache_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": metadata,
                    "stats": self.stats,
                    "build_timestamp": time.time()
                }, f, ensure_ascii=False, indent=2)
            
            print(f"FAISS索引已保存到: {self.cache_dir}")
            print(f"元数据已保存到: {metadata_path}")
            
            return True
            
        except Exception as e:
            error_msg = f"保存FAISS索引时出错: {str(e)}"
            print(error_msg)
            self.stats["errors"].append(error_msg)
            return False
    
    def load_index(self) -> Optional[FAISS]:
        """
        加载已存在的FAISS索引
        
        Returns:
            FAISS: 加载的FAISS索引对象
        """
        try:
            if not self._index_exists():
                print("FAISS索引不存在")
                return None
            
            # 加载FAISS索引
            faiss_index = FAISS.load_local(
                self.cache_dir, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # 加载元数据
            metadata_path = os.path.join(self.cache_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"加载索引元数据: {len(data.get('metadata', []))} 个chunks")
                    if 'stats' in data:
                        print(f"原始构建统计: {data['stats']}")
            
            print(f"成功从 {self.cache_dir} 加载FAISS索引")
            return faiss_index
            
        except Exception as e:
            error_msg = f"加载FAISS索引时出错: {str(e)}"
            print(error_msg)
            return None
    
    def _index_exists(self) -> bool:
        """检查FAISS索引是否已存在"""
        index_file = os.path.join(self.cache_dir, "index.faiss")
        pkl_file = os.path.join(self.cache_dir, "index.pkl")
        return os.path.exists(index_file) and os.path.exists(pkl_file)
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        获取索引信息
        
        Returns:
            Dict: 索引信息字典
        """
        info = {
            "index_exists": self._index_exists(),
            "cache_dir": self.cache_dir,
            "files_dir": self.files_dir
        }
        
        if self._index_exists():
            metadata_path = os.path.join(self.cache_dir, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        info.update({
                            "chunks_count": len(data.get('metadata', [])),
                            "build_stats": data.get('stats', {}),
                            "build_timestamp": data.get('build_timestamp', 0)
                        })
                except Exception as e:
                    info["metadata_error"] = str(e)
        
        return info
    
    def test_search(self, query: str, k: int = 3) -> List[Dict]:
        """
        测试搜索功能
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        faiss_index = self.load_index()
        if not faiss_index:
            print("无法加载FAISS索引")
            return []
        
        try:
            # 执行相似度搜索
            docs = faiss_index.similarity_search(query, k=k)
            
            results = []
            for i, doc in enumerate(docs):
                results.append({
                    "rank": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content)
                })
            
            return results
            
        except Exception as e:
            print(f"搜索测试失败: {str(e)}")
            return []


# 便捷函数
def build_faiss_index(files_dir: str = FILES_DIR, 
                     cache_dir: str = None,  # 改为None，让构建器自动使用绝对路径
                     force_rebuild: bool = False) -> Optional[FAISS]:
    """
    便捷函数：构建FAISS索引
    
    Args:
        files_dir: 文档目录
        cache_dir: 索引存储目录（持久化），如果为None则自动使用项目根目录下的data/faiss_indices
        force_rebuild: 是否强制重建
        
    Returns:
        FAISS: 构建的索引对象
    """
    builder = FAISSIndexBuilder(files_dir=files_dir, cache_dir=cache_dir)
    return builder.build_index(force_rebuild=force_rebuild)


def load_faiss_index(cache_dir: str = None) -> Optional[FAISS]:  # 改为None，让构建器自动使用绝对路径
    """
    便捷函数：加载FAISS索引
    
    Args:
        cache_dir: 索引存储目录（持久化），如果为None则自动使用项目根目录下的data/faiss_indices
        
    Returns:
        FAISS: 加载的索引对象
    """
    builder = FAISSIndexBuilder(cache_dir=cache_dir)
    return builder.load_index()


if __name__ == "__main__":
    # 测试代码
    print("=== FAISS索引构建器测试 ===")
    
    builder = FAISSIndexBuilder()
    
    # 显示索引信息
    info = builder.get_index_info()
    print(f"索引信息: {info}")
    
    # 构建索引
    faiss_index = builder.build_index(force_rebuild=True)
    
    if faiss_index:
        print("\n=== 测试搜索功能 ===")
        test_queries = [
            "学生手册",
            "奖学金",
            "课程",
            "申请表"
        ]
        
        for query in test_queries:
            print(f"\n查询: '{query}'")
            results = builder.test_search(query, k=2)
            for result in results:
                print(f"  排名{result['rank']}: {result['content']}")
                print(f"    来源: {result['metadata'].get('filename', 'Unknown')}") 