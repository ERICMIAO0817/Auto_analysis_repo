#!/usr/bin/env python3
"""
GitAnalytics 数据收集和分析示例

这个示例展示了如何：
1. 从GitHub仓库收集commit数据
2. 对收集的数据进行综合分析
3. 生成分析报告和可视化图表
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_collection.repository_collector import RepositoryCollector
from core.analyzer import GitAnalyzer
from analysis.risk_predictor import RiskPredictor
from analysis.file_impact_predictor import FileImpactPredictor
from utils.logger import setup_logger

def main():
    """主函数 - 演示完整的数据收集和分析流程"""
    
    # 设置日志
    logger = setup_logger('ExampleAnalysis')
    logger.info("开始GitAnalytics示例分析")
    
    # 配置参数
    repo_url = "https://github.com/BeyondDimension/SteamTools"
    project_name = "steamtools_example"
    output_dir = "example_results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    try:
        # 步骤1: 收集GitHub仓库数据
        logger.info(f"步骤1: 从GitHub仓库收集数据 - {repo_url}")
        
        collector = RepositoryCollector()
        
        # 收集最近6个月的数据
        since_date = datetime.now() - timedelta(days=180)
        
        csv_file = collector.collect_repository_data(
            repo_url=repo_url,
            output_file=f"{project_name}_commits.csv",
            since=since_date,
            skip_empty_dmm=True  # 只收集有DMM值的提交
        )
        
        logger.info(f"数据收集完成，保存到: {csv_file}")
        
        # 步骤2: 加载和预处理数据
        logger.info("步骤2: 加载和预处理数据")
        
        import pandas as pd
        data = pd.read_csv(csv_file)
        logger.info(f"加载了 {len(data)} 条commit记录")
        
        # 显示数据概览
        print("\n" + "="*50)
        print("数据概览")
        print("="*50)
        print(f"总提交数: {len(data)}")
        print(f"时间范围: {data['author_date'].min()} 到 {data['author_date'].max()}")
        print(f"作者数量: {data['author'].nunique()}")
        print(f"主要文件类型: {data['main_file_type'].value_counts().head()}")
        
        # 步骤3: 综合分析
        logger.info("步骤3: 执行综合分析")
        
        config = {
            'enable_quality_analysis': True,
            'enable_association_mining': True,
            'enable_clustering': True,
            'enable_prediction': True,
            'enable_risk_prediction': True,
            'enable_file_impact': False  # 需要关联规则文件
        }
        
        analyzer = GitAnalyzer(config)
        results = analyzer.analyze_csv_data(data, project_name)
        
        # 步骤4: DMM风险预测
        logger.info("步骤4: DMM风险预测分析")
        
        # 检查DMM字段
        dmm_fields = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        if all(field in data.columns for field in dmm_fields):
            risk_predictor = RiskPredictor(logger)
            dmm_results = risk_predictor.predict_dmm_risk(data)
            
            if 'dmm_regression' in dmm_results:
                print("\n" + "="*50)
                print("DMM风险预测结果")
                print("="*50)
                
                regression_results = dmm_results['dmm_regression']
                if 'models' in regression_results:
                    for model_name, metrics in regression_results['models'].items():
                        print(f"{model_name}:")
                        print(f"  - R²分数: {metrics.get('avg_r2', 0):.3f}")
                        print(f"  - RMSE: {metrics.get('avg_rmse', 0):.3f}")
        
        # 步骤5: 生成报告
        logger.info("步骤5: 生成分析报告")
        
        print("\n" + "="*50)
        print("分析完成总结")
        print("="*50)
        
        # 基础统计
        if 'basic_stats' in results:
            stats = results['basic_stats']
            print(f"✓ 基础统计分析完成")
            print(f"  - 总提交数: {stats.get('total_commits', 'N/A')}")
            print(f"  - 作者数量: {stats.get('unique_authors', 'N/A')}")
        
        # 聚类分析
        if 'clustering_analysis' in results:
            print(f"✓ 聚类分析完成")
            cluster_results = results['clustering_analysis']
            if 'kmeans_clustering' in cluster_results:
                kmeans = cluster_results['kmeans_clustering']
                print(f"  - 最优聚类数: {kmeans.get('optimal_k', 'N/A')}")
                print(f"  - 轮廓系数: {kmeans.get('best_silhouette_score', 0):.3f}")
        
        # 关联规则分析
        if 'association_analysis' in results:
            print(f"✓ 关联规则分析完成")
            assoc_results = results['association_analysis']
            if 'rule_statistics' in assoc_results:
                rule_stats = assoc_results['rule_statistics']
                print(f"  - 规则总数: {rule_stats.get('total_rules', 'N/A')}")
                print(f"  - 平均置信度: {rule_stats.get('avg_confidence', 0):.3f}")
        
        # 机器学习预测
        if 'prediction_analysis' in results:
            print(f"✓ 机器学习预测完成")
        
        print(f"\n📊 可视化图表已保存到当前目录")
        print(f"📁 分析结果保存在: {os.getcwd()}")
        
        logger.info("示例分析完成!")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        print(f"❌ 分析失败: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 