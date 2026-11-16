from dotenv import load_dotenv
load_dotenv()

import os

print(">>> SCRIPT CONNECTS TO DB:", os.getenv("NEO4J_DATABASE"), "@", os.getenv("NEO4J_URI"))

print("ENV loaded DB:", os.getenv("NEO4J_DATABASE"))
print("ENV loaded URI:", os.getenv("NEO4J_URI"))
print("ENV loaded PW:", os.getenv("NEO4J_PASSWORD"))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from build_graph import KnowledgeGraphBuilder
from build_index_and_community import IndexCommunityBuilder
from build_chunk_index import ChunkIndexBuilder

# 添加FAISS索引构建器导入
from processor.faiss_indexer import FAISSIndexBuilder
from processor.canonical_dedup_pipeline import run_canonical_build

class KnowledgeGraphProcessor:
    """
    知识图谱处理器，整合了图谱构建和索引处理的完整流程。
    可以选择完整流程或单独执行其中一个步骤。
    """
    
    def __init__(self):
        """初始化知识图谱处理器"""
        self.console = Console()
        
    def process_all(self):
        """执行完整的处理流程"""
        try:
            # 显示开始面板
            start_text = Text("开始完整知识图谱处理流程 (Neo4j + FAISS)", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # === 步骤1: 构建基础图谱 ===
            self.console.print("\n" + "="*60)
            self.console.print("[bold yellow]步骤 1/4: 构建Neo4j基础图谱[/bold yellow]")
            self.console.print("="*60)
            try:
                graph_builder = KnowledgeGraphBuilder()
                self.console.print("[cyan]正在执行实体提取和关系构建...[/cyan]")
                graph_builder.process()
                self.console.print("[green]✓ Neo4j基础图谱构建成功[/green]")
            except Exception as e:
                self.console.print(f"[red]✗ Neo4j基础图谱构建失败: {str(e)}[/red]")
                raise Exception(f"Neo4j基础图谱构建阶段失败: {str(e)}")
            
            # === 步骤2: 构建实体索引和社区 ===
            self.console.print("\n" + "="*60)
            self.console.print("[bold yellow]步骤 2/4: 构建Neo4j实体索引和社区检测[/bold yellow]")
            self.console.print("="*60)
            try:
                index_builder = IndexCommunityBuilder()
                self.console.print("[cyan]正在创建向量索引和社区检测...[/cyan]")
                index_builder.process()
                self.console.print("[green]✓ Neo4j实体索引和社区构建成功[/green]")
            except Exception as e:
                self.console.print(f"[red]✗ Neo4j实体索引和社区构建失败: {str(e)}[/red]")
                raise Exception(f"Neo4j索引和社区构建阶段失败: {str(e)}")
            
            # === 步骤3: 构建Chunk索引 ===
            self.console.print("\n" + "="*60)
            self.console.print("[bold yellow]步骤 3/4: 构建Neo4j文档块索引[/bold yellow]")
            self.console.print("="*60)
            try:
                chunk_index_builder = ChunkIndexBuilder()
                self.console.print("[cyan]正在构建文档块索引...[/cyan]")
                chunk_index_builder.process()
                self.console.print("[green]✓ Neo4j文档块索引构建成功[/green]")
            except Exception as e:
                self.console.print(f"[red]✗ Neo4j文档块索引构建失败: {str(e)}[/red]")
                raise Exception(f"Neo4j文档块索引构建阶段失败: {str(e)}")
            
            # === 步骤4: 构建FAISS向量索引 ===
            self.console.print("\n" + "="*60)
            self.console.print("[bold yellow]步骤 4/4: 构建FAISS向量索引[/bold yellow]")
            self.console.print("="*60)
            try:
                self.console.print("[cyan]正在初始化FAISS索引构建器...[/cyan]")
                faiss_builder = FAISSIndexBuilder()
                
                # 显示FAISS配置信息
                info = faiss_builder.get_index_info()
                self.console.print(f"[dim]FAISS配置信息:[/dim]")
                self.console.print(f"[dim]  - 文档目录: {info['files_dir']}[/dim]")
                self.console.print(f"[dim]  - 索引目录: {info['cache_dir']}[/dim]")
                self.console.print(f"[dim]  - 批处理大小: {faiss_builder.batch_size}[/dim]")
                
                self.console.print("[cyan]正在读取文档并生成向量索引...[/cyan]")
                faiss_index = faiss_builder.build_index(force_rebuild=False)
                
                if faiss_index:
                    self.console.print("[green]✓ FAISS向量索引构建成功[/green]")
                    
                    # 显示FAISS构建统计
                    updated_info = faiss_builder.get_index_info()
                    if 'build_stats' in updated_info:
                        stats = updated_info['build_stats']
                        self.console.print(f"[dim]FAISS构建统计:[/dim]")
                        self.console.print(f"[dim]  - 处理文档: {stats.get('files_processed', 0)} 个[/dim]")
                        self.console.print(f"[dim]  - 生成文档块: {stats.get('chunks_created', 0)} 个[/dim]")
                        self.console.print(f"[dim]  - 构建时间: {stats.get('build_time', 0):.2f} 秒[/dim]")
                else:
                    self.console.print("[red]✗ FAISS向量索引构建失败[/red]")
                    raise Exception("FAISS索引构建返回None，可能是文档读取或向量生成失败")
                    
            except Exception as e:
                self.console.print(f"[red]✗ FAISS向量索引构建失败: {str(e)}[/red]")
                raise Exception(f"FAISS索引构建阶段失败: {str(e)}")
            
            # === 显示完成面板 ===
            self.console.print("\n" + "="*60)
            success_text = Text("🎉 完整知识图谱处理流程成功完成！", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))
            
            # 显示最终总结
            self.console.print("\n[bold green]构建完成总结:[/bold green]")
            self.console.print("[green]✓ Neo4j基础图谱 - 实体和关系已建立[/green]")
            self.console.print("[green]✓ Neo4j实体索引 - 向量搜索已就绪[/green]") 
            self.console.print("[green]✓ Neo4j社区检测 - 主题聚类已完成[/green]")
            self.console.print("[green]✓ Neo4j文档块索引 - 文档检索已就绪[/green]")
            self.console.print("[green]✓ FAISS向量索引 - 高速向量检索已就绪[/green]")
            self.console.print("\n[bold cyan]现在可以启动各种Agent进行测试了！[/bold cyan]")
            
        except Exception as e:
            # 详细的错误信息
            self.console.print("\n" + "="*60)
            error_text = Text(f"❌ 构建流程中断", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            
            self.console.print(f"[red]错误详情: {str(e)}[/red]")
            self.console.print(f"[yellow]建议: 检查上述错误信息，修复问题后重新运行[/yellow]")
            self.console.print(f"[dim]提示: 可以使用 'python build/main.py neo4j' 或 'python build/main.py faiss' 单独构建特定部分[/dim]")
            raise
    
    def build_faiss_only(self):
        """仅构建FAISS索引"""
        try:
            # 显示开始面板
            start_text = Text("🚀 仅构建FAISS向量索引", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # 构建FAISS索引
            self.console.print("\n" + "="*50)
            self.console.print("[bold yellow]FAISS向量索引构建流程[/bold yellow]")
            self.console.print("="*50)
            
            self.console.print("[cyan]正在初始化FAISS索引构建器...[/cyan]")
            faiss_builder = FAISSIndexBuilder()
            
            # 显示配置信息
            info = faiss_builder.get_index_info()
            self.console.print(f"[dim]配置信息:[/dim]")
            self.console.print(f"[dim]  - 文档目录: {info['files_dir']}[/dim]")
            self.console.print(f"[dim]  - 索引存储目录: {info['cache_dir']}[/dim]")
            self.console.print(f"[dim]  - 批处理大小: {faiss_builder.batch_size}[/dim]")
            self.console.print(f"[dim]  - 并行工作线程: {faiss_builder.max_workers}[/dim]")
            
            # 检查是否已存在索引
            if info['index_exists']:
                self.console.print("[yellow]⚠️  发现已存在的FAISS索引，将强制重建[/yellow]")
            else:
                self.console.print("[blue]ℹ️  未发现已存在的索引，开始全新构建[/blue]")
            
            self.console.print("[cyan]正在读取文档文件...[/cyan]")
            self.console.print("[cyan]正在进行文本分块...[/cyan]")
            self.console.print("[cyan]正在生成向量嵌入...[/cyan]")
            self.console.print("[cyan]正在构建FAISS索引...[/cyan]")
            
            faiss_index = faiss_builder.build_index(force_rebuild=True)
            
            if faiss_index:
                # 成功构建
                success_text = Text("✅ FAISS索引构建成功！", style="bold green")
                self.console.print(Panel(success_text, border_style="green"))
                
                # 显示详细统计信息
                updated_info = faiss_builder.get_index_info()
                self.console.print("\n[bold green]构建统计信息:[/bold green]")
                self.console.print(f"[green]✓ 索引目录: {updated_info['cache_dir']}[/green]")
                
                if 'chunks_count' in updated_info:
                    self.console.print(f"[green]✓ 文档块数量: {updated_info['chunks_count']:,} 个[/green]")
                    
                if 'build_stats' in updated_info:
                    stats = updated_info['build_stats']
                    self.console.print(f"[green]✓ 处理文档: {stats.get('files_processed', 0)} 个[/green]")
                    self.console.print(f"[green]✓ 生成文档块: {stats.get('chunks_created', 0):,} 个[/green]")
                    self.console.print(f"[green]✓ 向量嵌入: {stats.get('embeddings_generated', 0):,} 个[/green]")
                    self.console.print(f"[green]✓ 构建耗时: {stats.get('build_time', 0):.2f} 秒[/green]")
                    
                    if stats.get('errors'):
                        self.console.print(f"[yellow]⚠️  处理错误: {len(stats['errors'])} 个文件处理失败[/yellow]")
                        for error in stats['errors'][:3]:  # 只显示前3个错误
                            self.console.print(f"[dim]   - {error}[/dim]")
                
                self.console.print(f"\n[bold cyan]FAISS索引已就绪，可以用于高速向量检索！[/bold cyan]")
                
            else:
                # 构建失败
                error_text = Text("❌ FAISS索引构建失败", style="bold red")
                self.console.print(Panel(error_text, border_style="red"))
                
                # 显示可能的错误原因
                self.console.print("[red]可能的错误原因:[/red]")
                self.console.print("[red]  - 文档目录为空或不存在[/red]")
                self.console.print("[red]  - 嵌入模型加载失败[/red]")
                self.console.print("[red]  - 内存不足[/red]")
                self.console.print("[red]  - 磁盘空间不足[/red]")
                
                raise Exception("FAISS索引构建返回None")
                
        except Exception as e:
            # 详细的错误处理
            self.console.print("\n" + "="*50)
            error_text = Text(f"❌ FAISS构建过程失败", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            
            self.console.print(f"[red]错误详情: {str(e)}[/red]")
            self.console.print(f"[yellow]建议操作:[/yellow]")
            self.console.print(f"[yellow]  1. 检查 files/ 目录是否包含文档文件[/yellow]")
            self.console.print(f"[yellow]  2. 检查嵌入模型是否正确配置[/yellow]")
            self.console.print(f"[yellow]  3. 检查磁盘空间是否充足[/yellow]")
            self.console.print(f"[yellow]  4. 尝试减小batch_size参数[/yellow]")
            raise

    def build_canonical(self,
                        files_dir: str = None,
                        output_dir: str = None,
                        write_faiss: bool = True,
                        write_neo4j: bool = True,
                        extreme_within: float = None,
                        extreme_cross: float = None,
                        chunk_size: int = None,
                        overlap: int = None,
                        as_of: str = None) -> None:
        """运行 canonical 构建：分块→清理→指纹/语义去重→写入FAISS/Neo4j→输出alias映射"""
        from pathlib import Path as _P
        from config.settings import FILES_DIR as DEFAULT_FILES
        self.console.print("\n" + "="*60)
        self.console.print("[bold yellow]Canonical 构建：去重+统一索引[/bold yellow]")
        self.console.print("="*60)

        files_dir = files_dir or str(DEFAULT_FILES)
        output_dir = output_dir or str(_P(__file__).parent.parent / "data" / "canonical_index")
        extreme_within = extreme_within if extreme_within is not None else float(os.getenv("CANONICAL_WITHIN", "0.988"))
        extreme_cross = extreme_cross if extreme_cross is not None else float(os.getenv("CANONICAL_CROSS", "0.983"))
        if as_of:
            os.environ["CANONICAL_AS_OF"] = as_of

        self.console.print(f"[dim]files_dir: {files_dir}[/dim]")
        self.console.print(f"[dim]output_dir: {output_dir}[/dim]")
        self.console.print(f"[dim]within: {extreme_within}  cross: {extreme_cross}[/dim]")
        if chunk_size:
            self.console.print(f"[dim]chunk_size: {chunk_size}  overlap: {overlap}[/dim]")

        stats = run_canonical_build(
            files_dir=files_dir,
            output_dir=output_dir,
            write_faiss=write_faiss,
            write_neo4j=write_neo4j,
            extreme_within=extreme_within,
            extreme_cross=extreme_cross,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # 打印结果摘要
        self.console.print("\n[bold green]Canonical 构建完成[/bold green]")
        for k in ["files", "chunks", "canonical", "aliases", "faiss_index_path", "neo4j_nodes", "equivalence_summary_path", "elapsed_sec"]:
            if k in stats and stats[k] is not None:
                self.console.print(f"[green]✓ {k}: {stats[k]}[/green]")

if __name__ == "__main__":
    import sys
    
    try:
        processor = KnowledgeGraphProcessor()

        from py2neo import Graph
        from py2neo import Graph
        import os

        debug_graph = Graph(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
        version_result = debug_graph.run("CALL dbms.components() YIELD versions RETURN versions[0];").data()
        print(">>> CONNECTED NEO4J VERSION:", version_result)

        # 检查命令行参数
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            if command == "faiss":
                processor.build_faiss_only()
            elif command == "neo4j":
                # 只构建Neo4j相关（不包括FAISS）
                from rich.console import Console
                console = Console()
                console.print("[cyan]仅构建Neo4j知识图谱（不包括FAISS）[/cyan]")
                
                # 1. 构建基础图谱
                graph_builder = KnowledgeGraphBuilder()
                graph_builder.process()
                
                # 2. 构建实体索引和社区
                index_builder = IndexCommunityBuilder()
                index_builder.process()
                
                # 3. 构建Chunk索引
                chunk_index_builder = ChunkIndexBuilder()
                chunk_index_builder.process()
                
                console.print("[green]Neo4j知识图谱构建完成[/green]")
            elif command == "canonical":
                # 读取可选参数（通过环境变量传入，或使用默认）
                files_dir = os.getenv("CANONICAL_FILES_DIR")
                output_dir = os.getenv("CANONICAL_OUTPUT_DIR")
                within = os.getenv("CANONICAL_WITHIN")
                cross = os.getenv("CANONICAL_CROSS")
                chunk_size = os.getenv("CANONICAL_CHUNK_SIZE")
                overlap = os.getenv("CANONICAL_OVERLAP")
                as_of = os.getenv("CANONICAL_AS_OF")
                processor.build_canonical(
                    files_dir=files_dir,
                    output_dir=output_dir,
                    write_faiss=os.getenv("CANONICAL_WRITE_FAISS", "1") != "0",
                    write_neo4j=os.getenv("CANONICAL_WRITE_NEO4J", "1") != "0",
                    extreme_within=float(within) if within else None,
                    extreme_cross=float(cross) if cross else None,
                    chunk_size=int(chunk_size) if chunk_size else None,
                    overlap=int(overlap) if overlap else None,
                    as_of=as_of,
                )
            else:
                print("🔧 知识图谱构建工具使用说明")
                print("=" * 50)
                print("用法:")
                print("  python build/main.py          # 🚀 构建完整系统（推荐）")
                print("                                #    包含: Neo4j图谱 + FAISS索引")
                print()
                print("  python build/main.py neo4j    # 🗄️  仅构建Neo4j知识图谱")
                print("                                #    包含: 实体关系 + 索引 + 社区")
                print()
                print("  python build/main.py faiss    # ⚡ 仅构建FAISS向量索引")
                print("                                #    用于: 高速向量检索")
                print()
                print("  python build/main.py canonical  # 🧩 构建canonical去重+索引(FAISS/Neo4j) + alias映射")
                print("                                   #    分块→清理→指纹/语义去重→索引→等价组摘要")
                print()
                print("💡 建议首次使用时运行完整构建流程")
                print("⚠️  确保Neo4j数据库已启动并配置正确")
        else:
            # 默认执行完整流程
            processor.process_all()
            
    except Exception as e:
        console = Console()
        console.print(f"[red]执行过程中出现错误: {str(e)}[/red]")