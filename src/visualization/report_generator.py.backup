"""
报告生成器
生成HTML格式的综合分析报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging
import json

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_comprehensive_report(self, results: Dict[str, Any], output_dir: Path) -> Path:
        """
        生成综合分析报告
        
        Args:
            results: 分析结果
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        self.logger.info("生成综合分析报告...")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成各种图表
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        
        chart_files = self._generate_charts(results, charts_dir)
        
        # 生成HTML报告
        html_content = self._generate_html_report(results, chart_files)
        
        # 保存报告
        report_path = output_dir / "analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 保存原始数据
        self._save_raw_data(results, output_dir)
        
        self.logger.info(f"报告生成完成: {report_path}")
        return report_path
    
    def _generate_charts(self, results: Dict[str, Any], charts_dir: Path) -> Dict[str, str]:
        """生成图表"""
        chart_files = {}
        
        # 1. 基础统计图表
        if 'basic_statistics' in results:
            basic_charts = self._create_basic_statistics_charts(
                results['basic_statistics'], charts_dir
            )
            chart_files.update(basic_charts)
        
        # 2. 代码质量图表
        if 'quality_analysis' in results:
            quality_charts = self._create_quality_charts(
                results['quality_analysis'], charts_dir
            )
            chart_files.update(quality_charts)
        
        # 3. 聚类分析图表
        if 'clustering_analysis' in results:
            clustering_charts = self._create_clustering_charts(
                results['clustering_analysis'], charts_dir
            )
            chart_files.update(clustering_charts)
        
        # 4. 关联规则网络图
        if 'association_analysis' in results:
            association_charts = self._create_association_charts(
                results['association_analysis'], charts_dir
            )
            chart_files.update(association_charts)
        
        # 5. 机器学习结果图表
        if 'ml_analysis' in results:
            ml_charts = self._create_ml_charts(
                results['ml_analysis'], charts_dir
            )
            chart_files.update(ml_charts)
        
        return chart_files
    
    def _create_basic_statistics_charts(self, stats: Dict[str, Any], charts_dir: Path) -> Dict[str, str]:
        """创建基础统计图表"""
        chart_files = {}
        
        try:
            # 文件类型分布饼图
            if 'file_type_distribution' in stats:
                plt.figure(figsize=(10, 8))
                file_types = stats['file_type_distribution']
                
                # 只显示前10个最常见的文件类型
                sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]
                labels, sizes = zip(*sorted_types) if sorted_types else ([], [])
                
                if labels:
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    plt.title('File Type Distribution', fontsize=16, fontweight='bold')
                    plt.axis('equal')
                    
                    chart_path = charts_dir / "file_type_distribution.png"
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files['file_type_distribution'] = str(chart_path.name)
            
            # 变更类型分布柱状图
            if 'change_type_distribution' in stats:
                plt.figure(figsize=(10, 6))
                change_types = stats['change_type_distribution']
                
                if change_types:
                    types, counts = zip(*change_types.items())
                    bars = plt.bar(types, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                    
                    plt.title('变更类型分布', fontsize=16, fontweight='bold')
                    plt.xlabel('变更类型', fontsize=12)
                    plt.ylabel('数量', fontsize=12)
                    plt.xticks(rotation=45)
                    
                    # 添加数值标签
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom')
                    
                    chart_path = charts_dir / "change_type_distribution.png"
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files['change_type_distribution'] = str(chart_path.name)
        
        except Exception as e:
            self.logger.warning(f"创建基础统计图表失败: {str(e)}")
        
        return chart_files
    
    def _create_quality_charts(self, quality_data: Dict[str, Any], charts_dir: Path) -> Dict[str, str]:
        """创建代码质量图表"""
        chart_files = {}
        
        try:
            # DMM指标分布图
            if 'dmm_analysis' in quality_data:
                dmm_data = quality_data['dmm_analysis']
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('DMM Code Quality Metrics', fontsize=16, fontweight='bold')
                
                dmm_metrics = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
                
                for i, metric in enumerate(dmm_metrics):
                    if metric in dmm_data and isinstance(dmm_data[metric], dict):
                        row, col = i // 2, i % 2
                        ax = axes[row, col]
                        
                        metric_stats = dmm_data[metric]
                        labels = ['Mean', 'Median', 'Q25', 'Q75']
                        values = [
                            metric_stats.get('mean', 0),
                            metric_stats.get('median', 0),
                            metric_stats.get('q25', 0),
                            metric_stats.get('q75', 0)
                        ]
                        
                        bars = ax.bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                        ax.set_ylabel('Value')
                        
                        # 添加数值标签
                        for bar, value in zip(bars, values):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}', ha='center', va='bottom')
                
                # 隐藏空的子图
                if len(dmm_metrics) < 4:
                    axes[1, 1].set_visible(False)
                
                chart_path = charts_dir / "dmm_analysis.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['dmm_analysis'] = str(chart_path.name)
            
            # 质量分数分布图
            if 'overall_quality_score' in quality_data:
                score_data = quality_data['overall_quality_score']
                
                if 'score_distribution' in score_data:
                    plt.figure(figsize=(10, 6))
                    
                    distribution = score_data['score_distribution']
                    categories = ['优秀', '良好', '一般', '较差']
                    values = [
                        distribution.get('excellent', 0),
                        distribution.get('good', 0),
                        distribution.get('fair', 0),
                        distribution.get('poor', 0)
                    ]
                    
                    colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
                    bars = plt.bar(categories, values, color=colors)
                    
                    plt.title('代码质量分数分布', fontsize=16, fontweight='bold')
                    plt.xlabel('质量等级', fontsize=12)
                    plt.ylabel('提交数量', fontsize=12)
                    
                    # 添加数值标签
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom')
                    
                    chart_path = charts_dir / "quality_score_distribution.png"
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files['quality_score_distribution'] = str(chart_path.name)
        
        except Exception as e:
            self.logger.warning(f"创建质量图表失败: {str(e)}")
        
        return chart_files
    
    def _create_clustering_charts(self, clustering_data: Dict[str, Any], charts_dir: Path) -> Dict[str, str]:
        """创建聚类分析图表"""
        chart_files = {}
        
        try:
            # 降维可视化
            if 'dimensionality_reduction' in clustering_data:
                dim_data = clustering_data['dimensionality_reduction']
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle('降维分析结果', fontsize=16, fontweight='bold')
                
                methods = ['pca', 'tsne', 'umap']
                titles = ['PCA', 't-SNE', 'UMAP']
                
                for i, (method, title) in enumerate(zip(methods, titles)):
                    if method in dim_data and 'coordinates' in dim_data[method]:
                        coords = np.array(dim_data[method]['coordinates'])
                        
                        axes[i].scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=30)
                        axes[i].set_title(title, fontweight='bold')
                        axes[i].set_xlabel(f'{title} 1')
                        axes[i].set_ylabel(f'{title} 2')
                        axes[i].grid(True, alpha=0.3)
                    else:
                        axes[i].text(0.5, 0.5, f'{title}\n不可用', 
                                   ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_xticks([])
                        axes[i].set_yticks([])
                
                chart_path = charts_dir / "dimensionality_reduction.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['dimensionality_reduction'] = str(chart_path.name)
            
            # K-means聚类结果
            if 'kmeans_clustering' in clustering_data:
                kmeans_data = clustering_data['kmeans_clustering']
                
                if 'silhouette_scores' in kmeans_data:
                    plt.figure(figsize=(10, 6))
                    
                    k_values = list(kmeans_data['silhouette_scores'].keys())
                    scores = list(kmeans_data['silhouette_scores'].values())
                    
                    plt.plot(k_values, scores, 'bo-', linewidth=2, markersize=8)
                    plt.title('K-means聚类轮廓系数', fontsize=16, fontweight='bold')
                    plt.xlabel('聚类数 (k)', fontsize=12)
                    plt.ylabel('轮廓系数', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    
                    # 标记最优k值
                    if 'optimal_k' in kmeans_data:
                        optimal_k = kmeans_data['optimal_k']
                        optimal_score = kmeans_data['silhouette_scores'][optimal_k]
                        plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
                        plt.text(optimal_k, optimal_score, f'最优k={optimal_k}', 
                                ha='center', va='bottom', fontweight='bold')
                    
                    chart_path = charts_dir / "kmeans_silhouette.png"
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files['kmeans_silhouette'] = str(chart_path.name)
        
        except Exception as e:
            self.logger.warning(f"创建聚类图表失败: {str(e)}")
        
        return chart_files
    
    def _create_association_charts(self, association_data: Dict[str, Any], charts_dir: Path) -> Dict[str, str]:
        """创建关联规则网络图"""
        chart_files = {}
        
        try:
            if 'association_rules' in association_data and association_data['association_rules']:
                rules = association_data['association_rules']
                
                # 创建网络图
                G = nx.DiGraph()
                
                # 添加边（只取前20条规则以避免图过于复杂）
                for rule in rules[:20]:
                    if 'antecedents' in rule and 'consequents' in rule:
                        antecedents = rule['antecedents']
                        consequents = rule['consequents']
                        lift = rule.get('lift', 1.0)
                        
                        for ant in antecedents:
                            for cons in consequents:
                                G.add_edge(str(ant), str(cons), weight=lift)
                
                if G.number_of_nodes() > 0:
                    plt.figure(figsize=(15, 12))
                    
                    # 设置布局
                    pos = nx.spring_layout(G, k=3, iterations=50)
                    
                    # 绘制节点
                    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                         node_size=1000, alpha=0.7)
                    
                    # 绘制边
                    edges = G.edges()
                    weights = [G[u][v]['weight'] for u, v in edges]
                    nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights], 
                                         alpha=0.6, edge_color='gray')
                    
                    # 绘制标签
                    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
                    
                    plt.title('文件关联网络图', fontsize=16, fontweight='bold')
                    plt.axis('off')
                    
                    chart_path = charts_dir / "association_network.png"
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files['association_network'] = str(chart_path.name)
        
        except Exception as e:
            self.logger.warning(f"创建关联图表失败: {str(e)}")
        
        return chart_files
    
    def _create_ml_charts(self, ml_data: Dict[str, Any], charts_dir: Path) -> Dict[str, str]:
        """创建机器学习结果图表"""
        chart_files = {}
        
        try:
            # 设置中文字体
            plt.rcParams['font.family'] = 'Songti SC'
            plt.rcParams['axes.unicode_minus'] = False
            
            # 模型性能对比图
            if ml_data:
                self.logger.info(f"ML数据结构: {list(ml_data.keys())}")
                
                # 收集所有模型的性能指标
                model_performance = {}
                
                # 处理不同的数据结构
                for task_name, task_results in ml_data.items():
                    self.logger.info(f"处理任务: {task_name}, 类型: {type(task_results)}")
                    
                    if isinstance(task_results, dict):
                        # 检查是否有models字段
                        if 'models' in task_results:
                            models_data = task_results['models']
                            for model_name, model_results in models_data.items():
                                if isinstance(model_results, dict) and 'error' not in model_results:
                                    if model_name not in model_performance:
                                        model_performance[model_name] = {}
                                    
                                    # 收集性能指标
                                    if 'accuracy' in model_results:
                                        model_performance[model_name][f'{task_name}_accuracy'] = model_results['accuracy']
                                    if 'avg_accuracy' in model_results:
                                        model_performance[model_name][f'{task_name}_accuracy'] = model_results['avg_accuracy']
                                    if 'f1_score' in model_results:
                                        model_performance[model_name][f'{task_name}_f1'] = model_results['f1_score']
                                    if 'avg_f1' in model_results:
                                        model_performance[model_name][f'{task_name}_f1'] = model_results['avg_f1']
                                    if 'r2_score' in model_results:
                                        model_performance[model_name][f'{task_name}_r2'] = model_results['r2_score']
                                    if 'avg_r2' in model_results:
                                        model_performance[model_name][f'{task_name}_r2'] = model_results['avg_r2']
                        else:
                            # 直接处理模型结果
                            for model_name, model_results in task_results.items():
                                if isinstance(model_results, dict) and 'error' not in model_results:
                                    if model_name not in model_performance:
                                        model_performance[model_name] = {}
                                    
                                    # 收集性能指标
                                    if 'accuracy' in model_results:
                                        model_performance[model_name][f'{task_name}_accuracy'] = model_results['accuracy']
                                    if 'avg_accuracy' in model_results:
                                        model_performance[model_name][f'{task_name}_accuracy'] = model_results['avg_accuracy']
                                    if 'f1_score' in model_results:
                                        model_performance[model_name][f'{task_name}_f1'] = model_results['f1_score']
                                    if 'avg_f1' in model_results:
                                        model_performance[model_name][f'{task_name}_f1'] = model_results['avg_f1']
                                    if 'r2_score' in model_results:
                                        model_performance[model_name][f'{task_name}_r2'] = model_results['r2_score']
                                    if 'avg_r2' in model_results:
                                        model_performance[model_name][f'{task_name}_r2'] = model_results['avg_r2']
                
                self.logger.info(f"收集到的模型性能数据: {model_performance}")
                
                if model_performance:
                    # 创建性能对比图
                    plt.figure(figsize=(14, 10))
                    
                    models = list(model_performance.keys())
                    metrics = set()
                    for model_metrics in model_performance.values():
                        metrics.update(model_metrics.keys())
                    
                    metrics = sorted(list(metrics))
                    
                    if models and metrics:
                        # 创建子图
                        n_metrics = len(metrics)
                        n_cols = min(3, n_metrics)
                        n_rows = (n_metrics + n_cols - 1) // n_cols
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                        if n_metrics == 1:
                            axes = [axes]
                        elif n_rows == 1:
                            axes = axes if n_cols > 1 else [axes]
                        else:
                            axes = axes.flatten()
                        
                        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
                        
                        for i, metric in enumerate(metrics):
                            ax = axes[i] if i < len(axes) else axes[0]
                            values = [model_performance[model].get(metric, 0) for model in models]
                            
                            bars = ax.bar(models, values, color=colors)
                            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
                            ax.set_ylabel('分数', fontsize=10)
                            ax.tick_params(axis='x', rotation=45)
                            
                            # 添加数值标签
                            for bar, value in zip(bars, values):
                                if value > 0:
                                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                        
                        # 隐藏多余的子图
                        for i in range(n_metrics, len(axes)):
                            axes[i].set_visible(False)
                        
                        plt.suptitle('机器学习模型性能对比', fontsize=16, fontweight='bold')
                        plt.tight_layout()
                        
                        chart_path = charts_dir / "ml_performance_comparison.png"
                        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        chart_files['ml_performance'] = str(chart_path.name)
                        self.logger.info(f"机器学习性能对比图已生成: {chart_path}")
                    else:
                        self.logger.warning("没有找到有效的模型或指标数据")
                else:
                    self.logger.warning("没有收集到模型性能数据")
        
        except Exception as e:
            self.logger.error(f"创建机器学习图表失败: {str(e)}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
        
        return chart_files
    
    def _generate_html_report(self, results: Dict[str, Any], chart_files: Dict[str, str]) -> str:
        """生成HTML报告"""
        repo_url = results.get('repository_url', 'Unknown')
        analysis_time = results.get('analysis_time', datetime.now())
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GitAnalytics Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            text-align: center;
            min-width: 120px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            display: block;
        }}
        .metric-label {{
            font-size: 12px;
            opacity: 0.8;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .info {{
            background-color: #d5e8d4;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #27ae60;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GitAnalytics Analysis Report</h1>
        
        <div class="info">
            <strong>Repository:</strong> {repo_url}<br>
            <strong>Analysis Time:</strong> {analysis_time.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        {self._generate_summary_section(results)}
        {self._generate_quality_section(results, chart_files)}
        {self._generate_clustering_section(results, chart_files)}
        {self._generate_association_section(results, chart_files)}
        {self._generate_ml_section(results, chart_files)}
        
        <h2>📊 Data Overview</h2>
        <p>This report was generated using the GitAnalytics framework, providing comprehensive Git repository analysis results.</p>
        
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _generate_summary_section(self, results: Dict[str, Any]) -> str:
        """生成摘要部分"""
        if 'basic_statistics' not in results:
            return ""
        
        stats = results['basic_statistics']
        
        html = """
        <h2>📈 Basic Statistics</h2>
        <div class="summary">
        """
        
        metrics = [
            ('total_commits', 'Total Commits'),
            ('unique_authors', 'Unique Authors'),
            ('total_files_changed', 'Files Changed'),
            ('total_lines_added', 'Lines Added'),
            ('total_lines_deleted', 'Lines Deleted'),
        ]
        
        for key, label in metrics:
            if key in stats:
                value = stats[key]
                if isinstance(value, float):
                    value = f"{value:.2f}"
                elif isinstance(value, int):
                    value = f"{value:,}"
                
                html += f"""
                <div class="metric">
                    <span class="metric-value">{value}</span>
                    <span class="metric-label">{label}</span>
                </div>
                """
        
        html += "</div>"
        
        # 添加图表
        if 'file_type_distribution' in results.get('chart_files', {}):
            html += f"""
            <div class="chart">
                <h3>File Type Distribution</h3>
                <img src="charts/{results['chart_files']['file_type_distribution']}" alt="File Type Distribution">
            </div>
            """
        
        return html
    
    def _generate_quality_section(self, results: Dict[str, Any], chart_files: Dict[str, str]) -> str:
        """生成代码质量部分"""
        if 'quality_analysis' not in results:
            return ""
        
        html = "<h2>🔍 Code Quality Analysis</h2>"
        
        quality_data = results['quality_analysis']
        
        # DMM分析图表
        if 'dmm_analysis' in chart_files:
            html += f"""
            <div class="chart">
                <h3>DMM Code Quality Metrics</h3>
                <img src="charts/{chart_files['dmm_analysis']}" alt="DMM Analysis">
            </div>
            """
        
        # 质量分数分布
        if 'quality_score_distribution' in chart_files:
            html += f"""
            <div class="chart">
                <h3>代码质量分数分布</h3>
                <img src="charts/{chart_files['quality_score_distribution']}" alt="质量分数分布">
            </div>
            """
        
        # 质量摘要
        if 'overall_quality_score' in quality_data:
            score_data = quality_data['overall_quality_score']
            if 'mean_score' in score_data:
                html += f"""
                <div class="info">
                    <strong>整体质量评分:</strong> {score_data['mean_score']:.3f} / 1.000<br>
                    <strong>中位数评分:</strong> {score_data.get('median_score', 0):.3f}
                </div>
                """
        
        return html
    
    def _generate_clustering_section(self, results: Dict[str, Any], chart_files: Dict[str, str]) -> str:
        """生成聚类分析部分"""
        if 'clustering_analysis' not in results:
            return ""
        
        html = "<h2>🎯 聚类分析</h2>"
        
        # 降维可视化
        if 'dimensionality_reduction' in chart_files:
            html += f"""
            <div class="chart">
                <h3>降维分析结果</h3>
                <img src="charts/{chart_files['dimensionality_reduction']}" alt="降维分析">
            </div>
            """
        
        # K-means结果
        if 'kmeans_silhouette' in chart_files:
            html += f"""
            <div class="chart">
                <h3>K-means聚类分析</h3>
                <img src="charts/{chart_files['kmeans_silhouette']}" alt="K-means聚类">
            </div>
            """
        
        return html
    
    def _generate_association_section(self, results: Dict[str, Any], chart_files: Dict[str, str]) -> str:
        """生成关联分析部分"""
        if 'association_analysis' not in results:
            return ""
        
        html = "<h2>🔗 文件关联分析</h2>"
        
        # 关联网络图
        if 'association_network' in chart_files:
            html += f"""
            <div class="chart">
                <h3>文件关联网络</h3>
                <img src="charts/{chart_files['association_network']}" alt="文件关联网络">
            </div>
            """
        
        # 关联规则表格
        association_data = results['association_analysis']
        if 'analysis' in association_data and 'strong_associations' in association_data['analysis']:
            strong_rules = association_data['analysis']['strong_associations']
            
            if strong_rules:
                html += """
                <h3>强关联规则 (Top 10)</h3>
                <table>
                    <tr>
                        <th>前项</th>
                        <th>后项</th>
                        <th>置信度</th>
                        <th>提升度</th>
                        <th>支持度</th>
                    </tr>
                """
                
                for rule in strong_rules[:10]:
                    antecedents = ', '.join(rule['antecedents'])
                    consequents = ', '.join(rule['consequents'])
                    
                    html += f"""
                    <tr>
                        <td>{antecedents}</td>
                        <td>{consequents}</td>
                        <td>{rule['confidence']:.3f}</td>
                        <td>{rule['lift']:.3f}</td>
                        <td>{rule['support']:.3f}</td>
                    </tr>
                    """
                
                html += "</table>"
        
        return html
    
    def _generate_ml_section(self, results: Dict[str, Any], chart_files: Dict[str, str]) -> str:
        """生成机器学习部分"""
        if 'ml_analysis' not in results:
            return ""
        
        html = "<h2>🤖 机器学习分析</h2>"
        
        # 性能对比图
        if 'ml_performance' in chart_files:
            html += f"""
            <div class="chart">
                <h3>模型性能对比</h3>
                <img src="charts/{chart_files['ml_performance']}" alt="模型性能对比">
            </div>
            """
        
        # 详细结果表格
        ml_data = results['ml_analysis']
        
        # 处理不同的机器学习任务
        task_mapping = {
            'dmm_regression': 'DMM回归分析',
            'dmm_classification': 'DMM分类分析', 
            'file_type_prediction': '文件类型预测',
            'change_type_prediction': '变更类型预测'
        }
        
        # 格式化数值的辅助函数
        def format_metric(value):
            try:
                if value is None:
                    return "N/A"
                if isinstance(value, str):
                    if value.lower() in ['nan', 'n/a', 'none']:
                        return "N/A"
                    try:
                        float_val = float(value)
                        if np.isnan(float_val) or np.isinf(float_val):
                            return "N/A"
                        return f"{float_val:.3f}"
                    except:
                        return "N/A"
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        return "N/A"
                    return f"{value:.3f}"
                return "N/A"
            except Exception as e:
                self.logger.debug(f"格式化指标时出错: {value}, 错误: {e}")
                return "N/A"
        
        for task_name, task_results in ml_data.items():
            if not isinstance(task_results, dict):
                continue
                
            display_name = task_mapping.get(task_name, task_name.replace('_', ' ').title())
            html += f"<h3>{display_name}</h3>"
            html += """
            <table>
                <tr>
                    <th>模型</th>
                    <th>准确率/R²</th>
                    <th>精确率</th>
                    <th>召回率</th>
                    <th>F1分数</th>
                </tr>
            """
            
            # 处理不同的数据结构
            models_to_process = {}
            
            if 'models' in task_results and isinstance(task_results['models'], dict):
                # 新的数据结构：有models字典
                models_to_process = task_results['models']
            else:
                # 旧的数据结构：直接是模型结果
                # 过滤掉非模型结果的键
                exclude_keys = {'class_distribution', 'feature_importance', 'confusion_matrix', 'analysis'}
                models_to_process = {k: v for k, v in task_results.items() 
                                   if k not in exclude_keys and isinstance(v, dict)}
            
            # 如果没有找到模型，显示一行说明
            if not models_to_process:
                html += """
                <tr>
                    <td colspan="5" style="text-align: center; color: #666;">暂无可用的模型结果</td>
                </tr>
                """
            else:
                # 生成表格行
                for model_name, model_results in models_to_process.items():
                    if isinstance(model_results, dict) and 'error' not in model_results:
                        # 获取指标
                        accuracy = model_results.get('accuracy', model_results.get('r2_score', None))
                        precision = model_results.get('precision', None)
                        recall = model_results.get('recall', None)
                        f1 = model_results.get('f1_score', None)
                        
                        html += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{format_metric(accuracy)}</td>
                            <td>{format_metric(precision)}</td>
                            <td>{format_metric(recall)}</td>
                            <td>{format_metric(f1)}</td>
                        </tr>
                        """
                    elif 'error' in model_results:
                        html += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td colspan="4" style="color: red;">错误: {model_results['error']}</td>
                        </tr>
                        """
            
            html += "</table>"
        
        return html
    
    def _save_raw_data(self, results: Dict[str, Any], output_dir: Path):
        """保存原始数据"""
        try:
            # 保存为JSON格式（用于程序读取）
            json_data = {}
            for key, value in results.items():
                if key not in ['raw_data', 'processed_data']:  # 跳过大型DataFrame
                    json_data[key] = value
            
            # 处理不能序列化的对象
            def json_serializer(obj):
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    # 递归处理字典，确保所有键都是字符串
                    return {str(k): json_serializer(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [json_serializer(item) for item in obj]
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                return str(obj)
            
            json_path = output_dir / "analysis_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, default=json_serializer)
            
            # 保存DataFrame数据
            if 'raw_data' in results:
                csv_path = output_dir / "raw_data.csv"
                results['raw_data'].to_csv(csv_path, index=False, encoding='utf-8')
            
            if 'processed_data' in results:
                processed_path = output_dir / "processed_data.csv"
                results['processed_data'].to_csv(processed_path, index=False, encoding='utf-8')
            
        except Exception as e:
            self.logger.warning(f"保存原始数据失败: {str(e)}") 