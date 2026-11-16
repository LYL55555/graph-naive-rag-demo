import numpy as np
import os
import sys
import argparse
import warnings
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Union, Optional
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class VectorVisualizer:
    """向量相似度可视化工具类"""
    
    def __init__(self, figsize: tuple = (12, 8), verbose: bool = False):
        """
        初始化可视化器
        
        参数:
            figsize: 图形大小
        """
        self.figsize = figsize
        self.verbose = verbose
        # 抑制中文字体缺失警告
        warnings.filterwarnings("ignore", message=r"Glyph.*missing from current font.*")
        # 设置中文字体（自动选择当前系统已安装的字体）
        candidate_fonts = [
            'PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'Songti SC',
            'STHeiti', 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC',
            'Arial Unicode MS', 'DejaVu Sans'
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = None
        for f in candidate_fonts:
            if f in available:
                chosen = f
                break
        if not chosen:
            chosen = 'DejaVu Sans'
        plt.rcParams['font.sans-serif'] = [chosen]
        plt.rcParams['axes.unicode_minus'] = False
        
    def visualize_similarity_matrix(self, 
                                  query_vector: np.ndarray,
                                  candidate_vectors: List[np.ndarray],
                                  candidate_names: List[str],
                                  query_name: str = "查询",
                                  similarity_scores: Optional[List[float]] = None,
                                  save_path: Optional[str] = None):
        """
        可视化相似度矩阵
        
        参数:
            query_vector: 查询向量
            candidate_vectors: 候选向量列表
            candidate_names: 候选文本名称列表
            query_name: 查询名称
            similarity_scores: 相似度分数列表
            save_path: 保存路径
        """
        if getattr(self, 'verbose', False):
            print("[DEBUG] 开始计算 query→候选 相似度条形图数据…")
        # 计算相似度（如果没有提供）
        if similarity_scores is None:
            from search.utils import VectorUtils
            similarity_scores = []
            for vec in candidate_vectors:
                score = VectorUtils.cosine_similarity(query_vector, vec)
                similarity_scores.append(score)
        if getattr(self, 'verbose', False):
            print(f"[DEBUG] 相似度数量: {len(similarity_scores)}，候选数量: {len(candidate_names)}")
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 相似度条形图
        colors = ['red' if score < 0.5 else 'orange' if score < 0.7 else 'green' 
                 for score in similarity_scores]
        
        bars = ax1.bar(range(len(candidate_names)), similarity_scores, color=colors, alpha=0.7)
        ax1.set_xlabel('候选文本')
        ax1.set_ylabel('相似度分数')
        ax1.set_title(f'{query_name}与候选文本的相似度')
        ax1.set_xticks(range(len(candidate_names)))
        ax1.set_xticklabels(candidate_names, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # 在条形图上添加数值标签
        for i, (bar, score) in enumerate(zip(bars, similarity_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        if getattr(self, 'verbose', False):
            print("[DEBUG] 相似度条形图绘制完成")
        
        # # 2. 相似度热力图
        # # 创建相似度矩阵
        # similarity_matrix = np.array(similarity_scores).reshape(1, -1)
        #
        # sns.heatmap(similarity_matrix,
        #            annot=True,
        #            fmt='.3f',
        #            xticklabels=candidate_names,
        #            yticklabels=[query_name],
        #            cmap='RdYlGn',
        #            ax=ax2,
        #            cbar_kws={'label': '相似度分数'})
        # ax2.set_title('相似度热力图')
        #
        # plt.tight_layout()
        #
        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
        
    def visualize_vector_space_2d(self,
                                 query_vector: np.ndarray,
                                 candidate_vectors: List[np.ndarray],
                                 candidate_names: List[str],
                                 query_name: str = "查询",
                                 method: str = "PCA",
                                 save_path: Optional[str] = None):
        """
        在2D空间中可视化向量分布
        
        参数:
            query_vector: 查询向量
            candidate_vectors: 候选向量列表
            candidate_names: 候选文本名称列表
            query_name: 查询名称
            method: 降维方法 ("PCA" 或 "TSNE")
            save_path: 保存路径
        """
        if getattr(self, 'verbose', False):
            print(f"[DEBUG] 准备二维可视化，方法: {method}，候选数量: {len(candidate_vectors)}")
        # 合并所有向量
        all_vectors = np.vstack([query_vector.reshape(1, -1)] + candidate_vectors)
        all_names = [query_name] + candidate_names
        
        # 降维
        if method.upper() == "PCA":
            reducer = PCA(n_components=2)
        elif method.upper() == "TSNE":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("method 必须是 'PCA' 或 'TSNE'")
            
        vectors_2d = reducer.fit_transform(all_vectors)
        if getattr(self, 'verbose', False):
            print("[DEBUG] 降维完成，开始绘制散点与连线…")
        
        # 创建图形
        plt.figure(figsize=self.figsize)
        
        # 绘制向量点
        colors = ['red'] + ['blue'] * len(candidate_vectors)
        sizes = [200] + [100] * len(candidate_vectors)
        
        for i, (point, name, color, size) in enumerate(zip(vectors_2d, all_names, colors, sizes)):
            plt.scatter(point[0], point[1], c=color, s=size, alpha=0.7, label=name)
            plt.annotate(name, (point[0], point[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
        
        # 添加连接线（从查询到候选）
        query_point = vectors_2d[0]
        for i, candidate_point in enumerate(vectors_2d[1:], 1):
            plt.plot([query_point[0], candidate_point[0]], 
                    [query_point[1], candidate_point[1]], 
                    'k--', alpha=0.3)
        
        plt.xlabel(f'{method} 维度 1')
        plt.ylabel(f'{method} 维度 2')
        plt.title(f'向量空间可视化 ({method})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if getattr(self, 'verbose', False):
                print(f"[DEBUG] 二维分布图已保存: {save_path}")
        plt.show()
        if getattr(self, 'verbose', False):
            print("[DEBUG] 二维可视化完成")
        
    def visualize_interactive_similarity(self,
                                       query_vector: np.ndarray,
                                       candidate_vectors: List[np.ndarray],
                                       candidate_names: List[str],
                                       query_name: str = "查询",
                                       similarity_scores: Optional[List[float]] = None):
        """
        创建交互式相似度可视化（使用Plotly）
        
        参数:
            query_vector: 查询向量
            candidate_vectors: 候选向量列表
            candidate_names: 候选文本名称列表
            query_name: 查询名称
            similarity_scores: 相似度分数列表
        """
        if getattr(self, 'verbose', False):
            print("[DEBUG] 计算交互式可视化所需相似度数据…")
        if similarity_scores is None:
            from search.utils import VectorUtils
            similarity_scores = []
            for vec in candidate_vectors:
                score = VectorUtils.cosine_similarity(query_vector, vec)
                similarity_scores.append(score)
        if getattr(self, 'verbose', False):
            print(f"[DEBUG] 相似度数量: {len(similarity_scores)}")
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('相似度条形图', '相似度雷达图', '向量分布', '相似度热力图'),
            specs=[[{"type": "bar"}, {"type": "scatterpolar"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. 条形图
        colors = ['red' if score < 0.5 else 'orange' if score < 0.7 else 'green' 
                 for score in similarity_scores]
        
        fig.add_trace(
            go.Bar(x=candidate_names, y=similarity_scores, 
                   marker_color=colors, name="相似度分数"),
            row=1, col=1
        )
        
        # 2. 雷达图
        fig.add_trace(
            go.Scatterpolar(
                r=similarity_scores,
                theta=candidate_names,
                fill='toself',
                name='相似度雷达图'
            ),
            row=1, col=2
        )
        
        # 3. 向量分布（使用PCA降维）
        all_vectors = np.vstack([query_vector.reshape(1, -1)] + candidate_vectors)
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(all_vectors)
        
        # 查询点
        fig.add_trace(
            go.Scatter(x=[vectors_2d[0, 0]], y=[vectors_2d[0, 1]], 
                      mode='markers+text', name=query_name,
                      marker=dict(size=15, color='red'),
                      text=[query_name], textposition="top center"),
            row=2, col=1
        )
        
        # 候选点
        for i, (point, name) in enumerate(zip(vectors_2d[1:], candidate_names)):
            fig.add_trace(
                go.Scatter(x=[point[0]], y=[point[1]], 
                          mode='markers+text', name=name,
                          marker=dict(size=10, color='blue'),
                          text=[name], textposition="top center"),
                row=2, col=1
            )
        
        # 4. 热力图
        similarity_matrix = np.array(similarity_scores).reshape(1, -1)
        fig.add_trace(
            go.Heatmap(z=similarity_matrix, x=candidate_names, y=[query_name],
                      colorscale='RdYlGn', showscale=True),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text=f"{query_name}与候选文本的相似度分析",
            height=800,
            showlegend=True
        )
        
        # 更新坐标轴
        fig.update_xaxes(title_text="候选文本", row=1, col=1)
        fig.update_yaxes(title_text="相似度分数", row=1, col=1)
        fig.update_xaxes(title_text="PCA维度1", row=2, col=1)
        fig.update_yaxes(title_text="PCA维度2", row=2, col=1)
        
        fig.show()
        if getattr(self, 'verbose', False):
            print("[DEBUG] 交互式相似度可视化完成")
        
    def visualize_query_and_pairwise(self,
                                     query_vector: np.ndarray,
                                     candidate_vectors: List[np.ndarray],
                                     candidate_names: List[str],
                                     query_name: str = "查询",
                                     annot_values: bool = False,
                                      save_path: Optional[str] = None,
                                      candidate_texts: Optional[List[str]] = None,
                                      preview_len: int = 80):
        """
        同时可视化：
        1) 查询与候选的相似度条形图（颜色按阈值分级）
        2) 候选之间两两相似度热力图

        参数:
            query_vector: 查询向量
            candidate_vectors: 候选向量列表
            candidate_names: 候选名称（建议使用 chunk_id，短且唯一）
            query_name: 查询名称
            annot_values: 是否在热力图中显示数值
            save_path: 图片保存路径；若未提供，则保存在当前文件同目录下
        """
        if getattr(self, 'verbose', False):
            print("[DEBUG] 开始绘制：query→候选 相似度 + 候选两两热力图…")
        if len(candidate_vectors) == 0:
            raise ValueError("candidate_vectors 不能为空")
        if len(candidate_vectors) != len(candidate_names):
            raise ValueError("candidate_vectors 与 candidate_names 数量不一致")

        # 归一化
        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q) + 1e-12
        q_unit = q / q_norm

        X = np.vstack([vec.astype(np.float32) for vec in candidate_vectors])
        X_norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X_unit = X / X_norm

        # 查询对候选相似度（余弦）
        similarity_scores = (X_unit @ q_unit.reshape(-1, 1)).flatten()
        if similarity_scores.size and getattr(self, 'verbose', False):
            print(f"[DEBUG] query→候选 相似度：min={similarity_scores.min():.3f}, max={similarity_scores.max():.3f}, avg={similarity_scores.mean():.3f}")

        # 候选之间两两相似度矩阵（余弦）
        pairwise_sim = X_unit @ X_unit.T
        if getattr(self, 'verbose', False):
            print("[DEBUG] 两两相似度矩阵计算完成")

        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 左：查询-候选 条形图（阈值着色）
        colors = ['red' if s < 0.5 else 'orange' if s < 0.7 else 'green' for s in similarity_scores]
        # x 轴标签尽量简短：仅显示文件名+chunk号
        display_names = []
        for name in candidate_names:
            base = name.split('/')[-1]
            display_names.append(base)
        bars = ax1.bar(range(len(display_names)), similarity_scores, color=colors, alpha=0.8)
        ax1.set_xlabel('候选片段（chunk_id）')
        ax1.set_ylabel('相似度（query → 候选）')
        ax1.set_title(f'{query_name} 与候选的相似度')
        ax1.set_xticks(range(len(display_names)))
        ax1.set_xticklabels(display_names, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        for bar, score in zip(bars, similarity_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.3f}', ha='center', va='bottom')

        # 右：候选-候选 两两相似度热力图
        sns.heatmap(pairwise_sim,
                    xticklabels=display_names,
                    yticklabels=display_names,
                    annot=annot_values,
                    fmt='.2f' if annot_values else '',
                    cmap='RdYlGn',
                    vmin=0.0,
                    vmax=1.0,
                    square=True,
                    cbar_kws={'label': '余弦相似度'},
                    ax=ax2)
        ax2.set_title('候选片段两两相似度')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

        # 为底部文本留白
        plt.tight_layout(rect=(0.0, 0.25, 1.0, 1.0))

        # 底部排名文本区
        try:
            sorted_idx = np.argsort(-similarity_scores)
            lines = []
            for rank, idx in enumerate(sorted_idx, start=1):
                cid = candidate_names[idx]
                score = similarity_scores[idx]
                snippet = ""
                if candidate_texts and idx < len(candidate_texts):
                    snippet_raw = (candidate_texts[idx] or "").replace("\n", " ")
                    snippet = snippet_raw[:preview_len] + ("…" if len(snippet_raw) > preview_len else "")
                lines.append(f"{rank}.[score={score:.3f}]-{cid}-{snippet}")

            text_block = "按相似度排名：\n" + "\n".join(lines)
            fig.text(0.01, 0.02, text_block, ha='left', va='bottom', fontsize=10)
        except Exception:
            # 排名文本失败不影响主图
            pass

        # 保存图片到本文件同目录
        if save_path is None:
            current_dir = os.path.dirname(__file__)
            save_path = os.path.join(current_dir, 'similarity_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        if getattr(self, 'verbose', False):
            print(f"[DEBUG] 可视化图片已保存: {save_path}")
        
    def create_similarity_report(self,
                                query_vector: np.ndarray,
                                candidate_vectors: List[np.ndarray],
                                candidate_names: List[str],
                                query_name: str = "查询",
                                similarity_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        生成详细的相似度分析报告
        
        参数:
            query_vector: 查询向量
            candidate_vectors: 候选向量列表
            candidate_names: 候选文本名称列表
            query_name: 查询名称
            similarity_scores: 相似度分数列表
            
        返回:
            包含分析结果的字典
        """
        # 计算相似度
        if similarity_scores is None:
            from search.utils import VectorUtils
            similarity_scores = []
            for vec in candidate_vectors:
                score = VectorUtils.cosine_similarity(query_vector, vec)
                similarity_scores.append(score)
        
        # 创建结果字典
        results = {
            "query_name": query_name,
            "candidate_count": len(candidate_names),
            "similarity_scores": dict(zip(candidate_names, similarity_scores)),
            "statistics": {
                "max_similarity": max(similarity_scores),
                "min_similarity": min(similarity_scores),
                "avg_similarity": np.mean(similarity_scores),
                "std_similarity": np.std(similarity_scores),
                "median_similarity": np.median(similarity_scores)
            },
            "ranking": sorted(zip(candidate_names, similarity_scores), 
                            key=lambda x: x[1], reverse=True)
        }
        
        # 添加分类信息
        high_similarity = [(name, score) for name, score in zip(candidate_names, similarity_scores) if score >= 0.7]
        medium_similarity = [(name, score) for name, score in zip(candidate_names, similarity_scores) if 0.5 <= score < 0.7]
        low_similarity = [(name, score) for name, score in zip(candidate_names, similarity_scores) if score < 0.5]
        
        results["categories"] = {
            "high_similarity": high_similarity,
            "medium_similarity": medium_similarity,
            "low_similarity": low_similarity
        }
        
        return results

def demo_visualization(query: str, k: int = 5, cache_dir: Optional[str] = None,
                       save_path: Optional[str] = None, annot_values: bool = False):
    """基于 FAISS 索引，按查询选 Top-K 候选并可视化
    
    参数:
        query: 查询文本
        k: 选取数量
        cache_dir: FAISS 索引目录（默认 data/faiss_indices）
        save_path: 图片保存路径（默认保存在当前文件同目录）
        annot_values: 是否在热力图显示数值
    """
    print("[INFO] 可视化开始…")
    if not query:
        print("[WARN] 未提供 query，请使用 --query 传入查询文本")
        return
    
    from model.get_models import get_embeddings_model
    from processor.faiss_indexer import load_faiss_index
    
    print("[INFO] 加载 embeddings 模型…")
    embeddings = get_embeddings_model()
    
    print("[INFO] 加载 FAISS 索引…")
    faiss_index = load_faiss_index(cache_dir=cache_dir)
    if faiss_index is None:
        print("[ERROR] 未能加载 FAISS 索引，请先构建索引或检查 --cache_dir")
        return
    
    print(f"[INFO] 基于查询检索 Top-{k} 候选…")
    docs = faiss_index.similarity_search(query, k=k)
    if not docs:
        print("[WARN] 未检索到候选片段")
        return
    
    candidate_names = [doc.metadata.get("chunk_id", f"doc#{i}") for i, doc in enumerate(docs)]
    candidate_texts = [doc.page_content for doc in docs]
    
    print("[INFO] 计算向量…")
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32)
    candidate_vectors = [np.array(v, dtype=np.float32) for v in embeddings.embed_documents(candidate_texts)]

    # 计算 query→候选 相似度，并在终端输出精简预览（前60字）
    q = query_vector.astype(np.float32)
    q /= (np.linalg.norm(q) + 1e-12)
    X = np.vstack(candidate_vectors).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    scores = (X @ q.reshape(-1, 1)).flatten()

    print("[INFO] 候选片段（按柱状图顺序）预览：")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        cid = doc.metadata.get("chunk_id", f"doc#{i}")
        preview = doc.page_content.replace("\n", " ")
        preview = (preview[:60] + "…") if len(preview) > 60 else preview
        print(f"  - {i+1}. {cid}  [score={score:.3f}]  {preview}")
    
    visualizer = VectorVisualizer()
    print("[INFO] 绘制 query→候选 与 两两热力图…")
    visualizer.visualize_query_and_pairwise(
        query_vector=query_vector,
        candidate_vectors=candidate_vectors,
        candidate_names=candidate_names,
        query_name=query,
        annot_values=annot_values,
        save_path=save_path,
        candidate_texts=candidate_texts,
        preview_len=100
    )
    print("[INFO] 可视化完成")
    
    # # 3. 交互式可视化
    # print("3. 生成交互式可视化...")
    # visualizer.visualize_interactive_similarity(
    #     query_vector, candidate_vectors, candidates, query
    # )
    #
    # # 4. 生成分析报告
    # print("4. 生成相似度分析报告...")
    # report = visualizer.create_similarity_report(
    #     query_vector, candidate_vectors, candidates, query
    # )
    #
    # print("\n=== 相似度分析报告 ===")
    # print(f"查询: {report['query_name']}")
    # print(f"候选数量: {report['candidate_count']}")
    # print(f"\n统计信息:")
    # for key, value in report['statistics'].items():
    #     print(f"  {key}: {value:.3f}")
    # print(f"\n排名:")
    # for i, (name, score) in enumerate(report['ranking'], 1):
    #     print(f"  {i}. {name}: {score:.3f}")
    # print(f"\n分类:")
    # for category, items in report['categories'].items():
    #     print(f"  {category}: {len(items)} 个")
    #     for name, score in items:
    #         print(f"    - {name}: {score:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于 FAISS 索引的向量相似度可视化")
    parser.add_argument("--query", type=str, required=True, help="查询文本")
    parser.add_argument("--k", type=int, default=5, help="选取前K个候选")
    parser.add_argument("--cache_dir", type=str, default=None, help="FAISS 索引目录（默认 data/faiss_indices）")
    parser.add_argument("--save_path", type=str, default=None, help="图片保存路径（默认当前文件同目录）")
    parser.add_argument("--annot", action="store_true", help="热力图显示数值")
    args = parser.parse_args()
    
    demo_visualization(
        query=args.query, 
        k=args.k, 
        cache_dir=args.cache_dir, 
        save_path=args.save_path, 
        annot_values=args.annot
    ) 