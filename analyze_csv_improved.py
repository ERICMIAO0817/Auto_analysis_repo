#!/usr/bin/env python3
"""
改进的CSV数据分析工具
整合DMM风险预测和文件影响预测功能
支持从GitHub仓库直接收集数据
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.analyzer import GitAnalyzer
from analysis.risk_predictor import RiskPredictor
from analysis.file_impact_predictor import FileImpactPredictor
from data_collection.repository_collector import RepositoryCollector
from utils.logger import setup_logger


def load_csv_data(csv_file: str) -> pd.DataFrame:
    """
    加载CSV数据文件
    
    Args:
        csv_file: CSV文件路径
        
    Returns:
        DataFrame对象
    """
    try:
        # 尝试不同的编码方式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                print(f"成功使用 {encoding} 编码加载文件: {csv_file}")
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"无法解码文件 {csv_file}")
        
    except Exception as e:
        print(f"加载CSV文件失败: {str(e)}")
        return pd.DataFrame()


def collect_repository_data(repo_url: str, 
                          output_file: str = None,
                          since: str = None,
                          to: str = None,
                          branch: str = None,
                          file_types: list = None,
                          skip_empty_dmm: bool = True,
                          resume: bool = True,
                          logger=None) -> str:
    """
    从GitHub仓库收集commit数据
    
    Args:
        repo_url: GitHub仓库URL
        output_file: 输出CSV文件路径
        since: 开始时间 (YYYY-MM-DD)
        to: 结束时间 (YYYY-MM-DD)
        branch: 指定分支
        file_types: 文件类型列表
        skip_empty_dmm: 跳过DMM值为空的提交
        logger: 日志记录器
        
    Returns:
        生成的CSV文件路径
    """
    if logger:
        logger.info(f"开始从GitHub仓库收集数据: {repo_url}")
    
    # 确定输出文件名
    if not output_file:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        output_file = f"data/raw/commits_{repo_name}.csv"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 解析时间参数
    since_date = None
    to_date = None
    if since:
        try:
            since_date = datetime.strptime(since, '%Y-%m-%d')
        except ValueError:
            if logger:
                logger.warning(f"无效的开始时间格式: {since}")
    
    if to:
        try:
            to_date = datetime.strptime(to, '%Y-%m-%d')
        except ValueError:
            if logger:
                logger.warning(f"无效的结束时间格式: {to}")
    
    # 创建收集器并收集数据
    collector = RepositoryCollector()
    
    try:
        result_file = collector.collect_repository_data(
            repo_url=repo_url,
            output_file=output_file,
            since=since_date,
            to=to_date,
            only_in_branch=branch,
            only_modifications_with_file_types=file_types,
            skip_empty_dmm=skip_empty_dmm,
            resume=resume
        )
        
        if logger:
            logger.info(f"数据收集完成，保存到: {result_file}")
        
        return result_file
        
    except Exception as e:
        if logger:
            logger.error(f"收集仓库数据失败: {str(e)}")
        raise


def analyze_dmm_risk(data: pd.DataFrame, logger) -> dict:
    """分析DMM风险"""
    logger.info("开始DMM风险分析...")
    
    # 检查是否有DMM字段
    dmm_fields = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
    if not all(field in data.columns for field in dmm_fields):
        logger.warning("缺少DMM字段，无法进行风险预测")
        return {'error': '缺少DMM字段'}
    
    risk_predictor = RiskPredictor(logger)
    results = risk_predictor.predict_dmm_risk(data)
    
    return results


def analyze_file_impact(commit_data: pd.DataFrame, rule_file: str, target_file: str, logger) -> dict:
    """分析文件影响"""
    logger.info(f"开始分析文件 {target_file} 的影响...")
    
    # 加载关联规则数据
    if not os.path.exists(rule_file):
        logger.warning(f"关联规则文件不存在: {rule_file}")
        return {'error': '关联规则文件不存在'}
    
    rule_data = load_csv_data(rule_file)
    if rule_data.empty:
        return {'error': '关联规则数据为空'}
    
    impact_predictor = FileImpactPredictor(logger)
    results = impact_predictor.predict_file_impact(commit_data, rule_data, target_file)
    
    return results


def comprehensive_analysis(csv_file: str, project_name: str, output_dir: str = None):
    """
    执行综合分析
    
    Args:
        csv_file: CSV文件路径
        project_name: 项目名称
        output_dir: 输出目录
    """
    logger = setup_logger('ComprehensiveAnalyzer')
    
    try:
        # 设置输出目录
        if output_dir:
            output_path = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"analysis_results_{project_name}_{timestamp}")
        
        output_path.mkdir(exist_ok=True)
        logger.info(f"分析结果将保存到: {output_path}")
        
        # 1. 加载数据
        logger.info(f"加载数据文件: {csv_file}")
        data = load_csv_data(csv_file)
        logger.info(f"数据加载成功，共 {len(data)} 行，{len(data.columns)} 列")
        
        # 数据概览
        print("数据概览:")
        print(f"- 行数: {len(data)}")
        print(f"- 列数: {len(data.columns)}")
        print(f"- 列名: {list(data.columns)}")
        
        # 2. 使用GitAnalyzer进行完整分析
        logger.info("开始基础Git分析...")
        from src.core.analyzer import GitAnalyzer
        
        analyzer = GitAnalyzer()
        
        # 执行CSV数据分析
        results = analyzer.analyze_csv_data(data, project_name)
        
        # 添加原始数据到结果中
        results['raw_data'] = data
        
        # 3. DMM风险分析（单独调用以确保生成图表）
        logger.info("开始DMM风险分析...")
        dmm_results = analyze_dmm_risk(data, logger)
        results['dmm_risk_analysis'] = dmm_results
        
        # 4. 文件影响分析
        possible_rule_files = [
            'steam_notype_all.csv',
            'httpclient_rule_all.csv', 
            'jackrabbit_rule_all.csv',
            'jruby_rule_all.csv',
            'lizard_rule_all.csv'
        ]
        
        rule_file_found = None
        for rule_file in possible_rule_files:
            if os.path.exists(rule_file):
                rule_file_found = rule_file
                break
            # 也检查上级目录
            parent_rule_file = os.path.join('..', rule_file)
            if os.path.exists(parent_rule_file):
                rule_file_found = parent_rule_file
                break
        
        if rule_file_found:
            # 选择一个示例文件进行影响分析
            target_files = [
                'SteamServiceImpl.cs',  # Steam项目
                'DefaultHttpClient.java',  # HTTP项目
                'Main.java',  # 通用
                'App.java'   # 通用
            ]
            
            for target_file in target_files:
                impact_results = analyze_file_impact(data, rule_file_found, target_file, logger)
                if 'error' not in impact_results and impact_results.get('influenced_files'):
                    results['file_impact_analysis'] = impact_results
                    logger.info(f"文件影响预测完成 - 目标文件: {target_file}")
                    break
        
        # 5. 生成完整的分析报告
        logger.info("生成完整分析报告...")
        report_path = analyzer.generate_report(results, output_path)
        logger.info(f"HTML报告已生成: {report_path}")
        
        # 6. 生成额外的可视化图表（基于您的代码风格）
        logger.info("生成额外的可视化图表...")
        _generate_additional_charts(data, results, output_path, logger)
        
        # 7. 生成控制台摘要报告
        logger.info("生成分析报告...")
        print("\n" + "="*50)
        print(f"项目 {project_name} 分析结果摘要")
        print("="*50)
        
        # 基础统计
        if 'basic_statistics' in results:
            stats = results['basic_statistics']
            print(f"提交总数: {stats.get('total_commits', 'N/A')}")
            print(f"作者数量: {stats.get('unique_authors', 'N/A')}")
            print(f"修改文件数: {stats.get('total_files_changed', 'N/A')}")
        
        # DMM风险分析
        if 'dmm_risk_analysis' in results:
            dmm_results = results['dmm_risk_analysis']
            if 'dmm_regression' in dmm_results and 'models' in dmm_results['dmm_regression']:
                best_model = None
                best_r2 = -1
                for model_name, metrics in dmm_results['dmm_regression']['models'].items():
                    if metrics.get('avg_r2', -1) > best_r2:
                        best_r2 = metrics['avg_r2']
                        best_model = model_name
                
                print(f"\nDMM风险预测:")
                print(f"- 最佳回归模型: {best_model}")
                print(f"- R²分数: {best_r2:.3f}")
        
        # 文件影响分析
        if 'file_impact_analysis' in results:
            impact_results = results['file_impact_analysis']
            target_file = impact_results.get('target_file', 'Unknown')
            influenced_files = impact_results.get('influenced_files', [])
            
            print(f"\n文件影响分析:")
            print(f"- 目标文件: {target_file}")
            print(f"- 受影响文件数: {len(influenced_files)}")
            
            if influenced_files:
                print(f"- 主要受影响文件: {influenced_files[:3]}")
        
        # 关联规则分析
        if 'association_analysis' in results:
            assoc_results = results['association_analysis']
            if 'analysis' in assoc_results and 'strong_associations' in assoc_results['analysis']:
                strong_rules = assoc_results['analysis']['strong_associations']
                print(f"\n关联规则分析:")
                print(f"- 强关联规则数: {len(strong_rules)}")
                if strong_rules:
                    avg_confidence = sum(rule['confidence'] for rule in strong_rules) / len(strong_rules)
                    avg_lift = sum(rule['lift'] for rule in strong_rules) / len(strong_rules)
                    print(f"- 平均置信度: {avg_confidence:.3f}")
                    print(f"- 平均提升度: {avg_lift:.3f}")
        
        # 聚类分析
        if 'clustering_analysis' in results:
            cluster_results = results['clustering_analysis']
            if 'kmeans_clustering' in cluster_results:
                kmeans = cluster_results['kmeans_clustering']
                print(f"\n聚类分析:")
                print(f"- K-means最优聚类数: {kmeans.get('optimal_k', 'N/A')}")
                print(f"- 轮廓系数: {kmeans.get('best_silhouette_score', 0):.3f}")
        
        print(f"\n分析完成！")
        print(f"📊 完整结果已保存到: {output_path}")
        print(f"📄 HTML报告: {report_path}")
        print(f"📈 可视化图表: {output_path}/charts/")
        print(f"📋 数据文件: {output_path}/")
        
    except Exception as e:
        logger.error(f"分析过程中出现错误: {str(e)}")
        print(f"分析失败: {str(e)}")


def _generate_additional_charts(data: pd.DataFrame, results: Dict, output_path: Path, logger):
    """
    生成额外的可视化图表（基于用户的代码风格）
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import numpy as np
    
    # 设置中文字体
    plt.rcParams['font.family'] = 'Songti SC'
    plt.rcParams['axes.unicode_minus'] = False
    
    charts_dir = output_path / "additional_charts"
    charts_dir.mkdir(exist_ok=True)
    
    try:
        # 1. 作者活跃度分析（类似您的代码风格）
        if 'author' in data.columns:
            plt.figure(figsize=(12, 8))
            author_counts = data['author'].value_counts().head(15)
            
            plt.subplot(2, 1, 1)
            author_counts.plot(kind='bar', color='skyblue')
            plt.title('作者提交次数分布 (Top 15)', fontsize=14)
            plt.xlabel('作者')
            plt.ylabel('提交次数')
            plt.xticks(rotation=45)
            
            # 作者提交时间分布
            if 'committer_date' in data.columns:
                plt.subplot(2, 1, 2)
                data['committer_date'] = pd.to_datetime(data['committer_date'], utc=True)
                monthly_commits = data.groupby(data['committer_date'].dt.to_period('M')).size()
                monthly_commits.plot(kind='line', color='green', marker='o')
                plt.title('月度提交趋势', fontsize=14)
                plt.xlabel('时间')
                plt.ylabel('提交次数')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(charts_dir / 'author_activity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("生成作者活跃度分析图表")
        
        # 2. 文件类型和变更类型分析
        if 'main_file_type' in data.columns and 'main_change_type' in data.columns:
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            file_type_counts = data['main_file_type'].value_counts()
            plt.pie(file_type_counts.values, labels=file_type_counts.index, autopct='%1.1f%%')
            plt.title('文件类型分布')
            
            plt.subplot(1, 2, 2)
            change_type_counts = data['main_change_type'].value_counts()
            plt.pie(change_type_counts.values, labels=change_type_counts.index, autopct='%1.1f%%')
            plt.title('变更类型分布')
            
            plt.tight_layout()
            plt.savefig(charts_dir / 'file_and_change_type_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("生成文件类型和变更类型分布图表")
        
        # 3. 代码变更规模分析
        if all(col in data.columns for col in ['insertions', 'deletions', 'files']):
            plt.figure(figsize=(15, 10))
            
            # 插入和删除行数分布
            plt.subplot(2, 2, 1)
            plt.scatter(data['insertions'], data['deletions'], alpha=0.6, color='coral')
            plt.xlabel('插入行数')
            plt.ylabel('删除行数')
            plt.title('插入vs删除行数分布')
            
            # 修改文件数分布
            plt.subplot(2, 2, 2)
            plt.hist(data['files'], bins=30, color='lightblue', alpha=0.7)
            plt.xlabel('修改文件数')
            plt.ylabel('频次')
            plt.title('单次提交修改文件数分布')
            
            # 代码行数变化趋势
            if 'committer_date' in data.columns:
                plt.subplot(2, 2, 3)
                data_sorted = data.sort_values('committer_date')
                data_sorted['cumulative_insertions'] = data_sorted['insertions'].cumsum()
                data_sorted['cumulative_deletions'] = data_sorted['deletions'].cumsum()
                
                plt.plot(data_sorted['committer_date'], data_sorted['cumulative_insertions'], 
                        label='累计插入', color='green')
                plt.plot(data_sorted['committer_date'], data_sorted['cumulative_deletions'], 
                        label='累计删除', color='red')
                plt.xlabel('时间')
                plt.ylabel('代码行数')
                plt.title('累计代码变更趋势')
                plt.legend()
                plt.xticks(rotation=45)
            
            # 提交规模分类
            plt.subplot(2, 2, 4)
            data['commit_size'] = data['insertions'] + data['deletions']
            size_categories = pd.cut(data['commit_size'], 
                                   bins=[0, 10, 50, 200, float('inf')], 
                                   labels=['小型', '中型', '大型', '超大型'])
            size_counts = size_categories.value_counts()
            plt.bar(size_counts.index, size_counts.values, color=['lightgreen', 'yellow', 'orange', 'red'])
            plt.xlabel('提交规模')
            plt.ylabel('提交次数')
            plt.title('提交规模分布')
            
            plt.tight_layout()
            plt.savefig(charts_dir / 'code_change_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("生成代码变更规模分析图表")
        
        # 4. 关联规则网络图（基于您的网络图代码）
        if 'association_analysis' in results and 'association_rules' in results['association_analysis']:
            rules_data = results['association_analysis']['association_rules']
            if rules_data:
                plt.figure(figsize=(16, 16))
                
                # 创建网络图
                G = nx.DiGraph()
                
                for rule in rules_data:
                    antecedents = list(rule['antecedents'])
                    consequents = list(rule['consequents'])
                    lift = rule['lift']
                    
                    for ant in antecedents:
                        for cons in consequents:
                            G.add_edge(ant, cons, weight=lift)
                
                if G.nodes():
                    # 计算节点度数
                    in_degree = dict(G.in_degree())
                    out_degree = dict(G.out_degree())
                    total_degree = {node: in_degree.get(node, 0) + out_degree.get(node, 0)
                                  for node in set(in_degree) | set(out_degree)}
                    
                    # 设置布局
                    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
                    
                    # 获取边权重
                    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                    edge_colors = [plt.cm.Blues(weight/max(edge_weights)) for weight in edge_weights]
                    
                    # 绘制网络图
                    nx.draw(
                        G, pos,
                        with_labels=True,
                        node_size=[total_degree.get(node, 1) * 500 + 1000 for node in G.nodes()],
                        node_color='lightcoral',
                        font_size=8,
                        font_weight="bold",
                        edge_color=edge_colors,
                        width=[w/2 for w in edge_weights],
                        alpha=0.7
                    )
                    
                    plt.title("文件关联网络图 (基于关联规则)", fontsize=16)
                    plt.axis("off")
                    plt.savefig(charts_dir / 'association_network_enhanced.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("生成增强版关联网络图")
        
        # 5. DMM指标相关性热力图
        dmm_cols = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        if all(col in data.columns for col in dmm_cols):
            plt.figure(figsize=(10, 8))
            
            # 选择数值型列进行相关性分析
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            correlation_matrix = data[numeric_cols].corr()
            
            # 创建热力图
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('特征相关性热力图')
            plt.tight_layout()
            plt.savefig(charts_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("生成特征相关性热力图")
        
        # 6. 高级聚类分析（基于您的UMAP代码风格）
        _generate_advanced_clustering_analysis(data, charts_dir, logger)
        
        # 7. 文件修改模式分析
        if 'modified_files' in data.columns:
            _generate_file_modification_analysis(data, charts_dir, logger)
        
        logger.info(f"额外图表已保存到: {charts_dir}")
        
    except Exception as e:
        logger.warning(f"生成额外图表时出现错误: {str(e)}")


def _generate_advanced_clustering_analysis(data: pd.DataFrame, charts_dir: Path, logger):
    """
    生成高级聚类分析（基于用户的UMAP代码风格）
    """
    try:
        import umap
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        import numpy as np
        
        # 准备数据进行聚类分析
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除可能的ID列
        exclude_cols = ['id', 'index', 'Unnamed: 0']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) < 2:
            logger.warning("数值列不足，跳过高级聚类分析")
            return
        
        # 处理分类变量
        categorical_cols = ['author', 'main_file_type', 'main_change_type']
        available_cat_cols = [col for col in categorical_cols if col in data.columns]
        
        # 创建用于聚类的数据集
        cluster_data = data[numeric_cols].copy()
        
        # 处理NaN值 - 使用均值填充
        logger.info(f"处理缺失值，原始数据形状: {cluster_data.shape}")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        cluster_data_imputed = pd.DataFrame(
            imputer.fit_transform(cluster_data),
            columns=cluster_data.columns,
            index=cluster_data.index
        )
        
        # 检查是否还有NaN值
        nan_count = cluster_data_imputed.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"填充后仍有 {nan_count} 个NaN值，使用0填充")
            cluster_data_imputed = cluster_data_imputed.fillna(0)
        
        # 检查是否有无穷大值
        inf_count = np.isinf(cluster_data_imputed.values).sum()
        if inf_count > 0:
            logger.warning(f"发现 {inf_count} 个无穷大值，替换为有限值")
            cluster_data_imputed = cluster_data_imputed.replace([np.inf, -np.inf], np.nan)
            cluster_data_imputed = cluster_data_imputed.fillna(cluster_data_imputed.mean())
        
        logger.info(f"数据预处理完成，最终形状: {cluster_data_imputed.shape}")
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data_imputed)
        
        # 检查标准化后的数据
        if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
            logger.error("标准化后数据仍包含NaN或无穷大值")
            return
        
        # 创建大图
        plt.figure(figsize=(20, 15))
        
        # 1. PCA分析
        plt.subplot(2, 3, 1)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.7)
        plt.title(f'PCA聚类分析 (解释方差: {pca.explained_variance_ratio_.sum():.2f})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.colorbar(scatter)
        
        # 2. UMAP分析（如果可用）
        try:
            plt.subplot(2, 3, 2)
            umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            umap_result = umap_reducer.fit_transform(scaled_data)
            
            scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.7)
            plt.title('UMAP聚类分析')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.colorbar(scatter)
        except ImportError:
            logger.warning("UMAP库未安装，跳过UMAP分析")
        
        # 3. 按作者分组的聚类（如果有作者信息）
        if 'author' in data.columns:
            plt.subplot(2, 3, 3)
            # 编码作者信息
            le = LabelEncoder()
            author_encoded = le.fit_transform(data['author'].fillna('Unknown'))
            
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=author_encoded, cmap='Set3', alpha=0.7)
            plt.title('按作者分组的PCA分布')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        
        # 4. 聚类中心分析
        plt.subplot(2, 3, 4)
        centers_pca = pca.transform(scaler.inverse_transform(kmeans.cluster_centers_))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                   c=cluster_labels, cmap='tab10', alpha=0.5)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='聚类中心')
        plt.title('聚类中心分析')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.legend()
        
        # 5. 特征重要性分析
        plt.subplot(2, 3, 5)
        feature_importance = np.abs(pca.components_[0]) + np.abs(pca.components_[1])
        feature_names = numeric_cols
        
        sorted_idx = np.argsort(feature_importance)[-10:]  # 取前10个重要特征
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title('特征重要性 (PCA)')
        plt.xlabel('重要性分数')
        
        # 6. 聚类大小分布
        plt.subplot(2, 3, 6)
        cluster_counts = np.bincount(cluster_labels)
        plt.bar(range(len(cluster_counts)), cluster_counts, color='lightblue')
        plt.title('聚类大小分布')
        plt.xlabel('聚类ID')
        plt.ylabel('样本数量')
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'advanced_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("生成高级聚类分析图表")
        
    except Exception as e:
        logger.warning(f"生成高级聚类分析时出现错误: {str(e)}")


def _generate_file_modification_analysis(data: pd.DataFrame, charts_dir: Path, logger):
    """
    生成文件修改模式分析
    """
    try:
        import matplotlib.pyplot as plt
        import ast
        from collections import Counter
        import networkx as nx
        
        # 解析modified_files列
        all_files = []
        file_pairs = []
        
        for idx, row in data.iterrows():
            try:
                if pd.isna(row['modified_files']):
                    continue
                    
                # 尝试解析文件列表
                if isinstance(row['modified_files'], str):
                    if row['modified_files'].startswith('['):
                        files = ast.literal_eval(row['modified_files'])
                    else:
                        files = [f.strip() for f in row['modified_files'].split(',')]
                else:
                    files = [str(row['modified_files'])]
                
                all_files.extend(files)
                
                # 生成文件对（用于共同修改分析）
                for i in range(len(files)):
                    for j in range(i+1, len(files)):
                        file_pairs.append((files[i], files[j]))
                        
            except Exception:
                continue
        
        if not all_files:
            logger.warning("无法解析modified_files数据，跳过文件修改分析")
            return
        
        # 创建分析图表
        plt.figure(figsize=(20, 12))
        
        # 1. 最常修改的文件
        plt.subplot(2, 3, 1)
        file_counts = Counter(all_files)
        top_files = file_counts.most_common(15)
        
        if top_files:
            files, counts = zip(*top_files)
            # 截断长文件名
            short_files = [f[:30] + '...' if len(f) > 30 else f for f in files]
            
            plt.barh(range(len(short_files)), counts, color='skyblue')
            plt.yticks(range(len(short_files)), short_files)
            plt.title('最常修改的文件 (Top 15)')
            plt.xlabel('修改次数')
        
        # 2. 文件扩展名分布
        plt.subplot(2, 3, 2)
        extensions = [f.split('.')[-1] if '.' in f else 'no_ext' for f in all_files]
        ext_counts = Counter(extensions)
        top_exts = ext_counts.most_common(10)
        
        if top_exts:
            exts, counts = zip(*top_exts)
            plt.pie(counts, labels=exts, autopct='%1.1f%%')
            plt.title('文件扩展名分布')
        
        # 3. 单次提交修改文件数分布
        plt.subplot(2, 3, 3)
        files_per_commit = []
        for idx, row in data.iterrows():
            try:
                if pd.isna(row['modified_files']):
                    continue
                if isinstance(row['modified_files'], str):
                    if row['modified_files'].startswith('['):
                        files = ast.literal_eval(row['modified_files'])
                    else:
                        files = [f.strip() for f in row['modified_files'].split(',')]
                    files_per_commit.append(len(files))
            except Exception:
                continue
        
        if files_per_commit:
            plt.hist(files_per_commit, bins=30, color='lightgreen', alpha=0.7)
            plt.title('单次提交修改文件数分布')
            plt.xlabel('修改文件数')
            plt.ylabel('提交次数')
        
        # 4. 文件共同修改网络（前20个最常见的文件对）
        plt.subplot(2, 3, 4)
        if file_pairs:
            pair_counts = Counter(file_pairs)
            top_pairs = pair_counts.most_common(20)
            
            G = nx.Graph()
            for (file1, file2), count in top_pairs:
                # 截断文件名
                short_file1 = file1.split('/')[-1][:15]
                short_file2 = file2.split('/')[-1][:15]
                G.add_edge(short_file1, short_file2, weight=count)
            
            if G.nodes():
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                
                # 绘制边
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                nx.draw_networkx_edges(G, pos, width=[w/max(weights)*3 for w in weights], 
                                     alpha=0.6, edge_color='gray')
                
                # 绘制节点
                nx.draw_networkx_nodes(G, pos, node_color='lightcoral', 
                                     node_size=300, alpha=0.8)
                
                # 绘制标签
                nx.draw_networkx_labels(G, pos, font_size=8)
                
                plt.title('文件共同修改网络')
                plt.axis('off')
        
        # 5. 目录层级分析
        plt.subplot(2, 3, 5)
        directories = []
        for f in all_files:
            if '/' in f:
                directories.append('/'.join(f.split('/')[:-1]))
            else:
                directories.append('root')
        
        dir_counts = Counter(directories)
        top_dirs = dir_counts.most_common(10)
        
        if top_dirs:
            dirs, counts = zip(*top_dirs)
            # 截断长目录名
            short_dirs = [d[:25] + '...' if len(d) > 25 else d for d in dirs]
            
            plt.barh(range(len(short_dirs)), counts, color='orange')
            plt.yticks(range(len(short_dirs)), short_dirs)
            plt.title('最活跃的目录 (Top 10)')
            plt.xlabel('修改次数')
        
        # 6. 文件修改时间模式（如果有时间信息）
        if 'committer_date' in data.columns:
            plt.subplot(2, 3, 6)
            
            # 按小时统计修改活动
            data['committer_date'] = pd.to_datetime(data['committer_date'], utc=True)
            hourly_activity = data['committer_date'].dt.hour.value_counts().sort_index()
            
            plt.plot(hourly_activity.index, hourly_activity.values, marker='o', color='purple')
            plt.title('每日修改活动模式')
            plt.xlabel('小时')
            plt.ylabel('提交次数')
            plt.xticks(range(0, 24, 2))
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'file_modification_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("生成文件修改模式分析图表")
        
    except Exception as e:
        logger.warning(f"生成文件修改分析时出现错误: {str(e)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='改进的CSV数据分析工具 - 支持从GitHub仓库直接收集数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分析现有CSV文件
  %(prog)s commits_steam_nomsg.csv --project-name steam
  
  # 从GitHub仓库收集数据并分析
  %(prog)s --repo-url https://github.com/BeyondDimension/SteamTools --project-name steamtools
  
  # 指定时间范围收集数据
  %(prog)s --repo-url https://github.com/apache/flink --since 2023-01-01 --to 2023-12-31 --project-name flink
        """
    )
    
    # 数据源选项 (互斥)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('csv_file', nargs='?', help='CSV文件路径')
    source_group.add_argument('--repo-url', help='GitHub仓库URL')
    
    # 项目信息
    parser.add_argument('--project-name', '-p', required=True, help='项目名称')
    parser.add_argument('--output', '-o', help='输出目录')
    
    # 仓库收集选项
    parser.add_argument('--since', help='开始时间 (格式: YYYY-MM-DD)')
    parser.add_argument('--to', help='结束时间 (格式: YYYY-MM-DD)')
    parser.add_argument('--branch', help='指定分析的分支')
    parser.add_argument('--file-types', nargs='+', help='指定文件类型 (如: .py .java .cpp)')
    parser.add_argument('--skip-empty-dmm', action='store_true', default=True,
                       help='跳过DMM值为空的提交 (默认: True)')
    parser.add_argument('--include-empty-dmm', action='store_true',
                       help='包含DMM值为空的提交')
    parser.add_argument('--csv-output', help='收集数据时的CSV输出文件路径')
    
    # 断点续传选项
    parser.add_argument('--resume', action='store_true', default=True,
                       help='启用断点续传 (默认: True)')
    parser.add_argument('--no-resume', action='store_true',
                       help='禁用断点续传，重新开始收集')
    
    # 其他选项
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    logger = setup_logger('MainAnalyzer')
    
    try:
        csv_file_to_analyze = None
        
        # 如果指定了仓库URL，先收集数据
        if args.repo_url:
            logger.info("从GitHub仓库收集数据模式")
            
            skip_empty_dmm = args.skip_empty_dmm and not args.include_empty_dmm
            
            csv_file_to_analyze = collect_repository_data(
                repo_url=args.repo_url,
                output_file=args.csv_output,
                since=args.since,
                to=args.to,
                branch=args.branch,
                file_types=args.file_types,
                skip_empty_dmm=skip_empty_dmm,
                resume=args.resume and not args.no_resume,
                logger=logger
            )
            
            logger.info(f"数据收集完成，开始分析文件: {csv_file_to_analyze}")
            
        else:
            # 使用现有CSV文件
            csv_file_to_analyze = args.csv_file
            
            if not os.path.exists(csv_file_to_analyze):
                print(f"错误: 文件不存在 {csv_file_to_analyze}")
                return 1
        
        # 执行综合分析
        comprehensive_analysis(csv_file_to_analyze, args.project_name, args.output)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 