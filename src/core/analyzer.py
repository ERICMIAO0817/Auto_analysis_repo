"""
GitAnalytics核心分析器
整合所有分析功能的主要类
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

from pydriller import Repository
from utils.data_extractor import DataExtractor
from utils.data_preprocessor import DataPreprocessor
from utils.logger import setup_logger
from analysis.quality_analyzer import QualityAnalyzer
from analysis.association_miner import AssociationMiner
from analysis.clustering_analyzer import ClusteringAnalyzer
from analysis.prediction_models import PredictionModels
from analysis.risk_predictor import RiskPredictor
from analysis.file_impact_predictor import FileImpactPredictor
from visualization.report_generator import ReportGenerator


class GitAnalyzer:
    """Git仓库智能分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化Git分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = setup_logger('GitAnalyzer')
        
        # 初始化各个分析模块
        self.data_extractor = DataExtractor(self.logger)
        self.preprocessor = DataPreprocessor(self.logger)
        self.quality_analyzer = QualityAnalyzer(self.logger) if self.config.get('enable_quality_analysis', True) else None
        self.association_miner = AssociationMiner(self.logger) if self.config.get('enable_association_mining', True) else None
        self.clustering_analyzer = ClusteringAnalyzer(self.logger) if self.config.get('enable_clustering', True) else None
        self.prediction_models = PredictionModels(self.logger) if self.config.get('enable_prediction', True) else None
        self.risk_predictor = RiskPredictor(self.logger) if self.config.get('enable_risk_prediction', True) else None
        self.file_impact_predictor = FileImpactPredictor(self.logger) if self.config.get('enable_file_impact', True) else None
        self.report_generator = ReportGenerator(self.logger)
    
    def analyze_repository(self, repo_url: str) -> Dict[str, Any]:
        """
        分析Git仓库
        
        Args:
            repo_url: 仓库URL
            
        Returns:
            分析结果字典
        """
        self.logger.info("开始提取仓库数据...")
        
        # 1. 数据提取
        raw_data = self.data_extractor.extract_repository_data(repo_url)
        
        if raw_data.empty:
            raise ValueError("无法从仓库中提取数据")
        
        self.logger.info(f"提取到 {len(raw_data)} 条提交记录")
        
        # 2. 数据预处理
        self.logger.info("开始数据预处理...")
        processed_data = self.preprocessor.preprocess(raw_data)
        
        # 3. 基础统计分析
        self.logger.info("进行基础统计分析...")
        basic_stats = self._get_basic_statistics(raw_data)
        
        # 4. 代码质量分析
        self.logger.info("进行代码质量分析...")
        quality_results = self.quality_analyzer.analyze(processed_data)
        
        # 5. 文件关联分析
        self.logger.info("进行文件关联分析...")
        association_results = self.association_miner.mine_associations(raw_data)
        
        # 6. 聚类分析
        self.logger.info("进行聚类分析...")
        clustering_results = self.clustering_analyzer.analyze(processed_data)
        
        # 7. 机器学习预测
        if self.prediction_models:
            self.logger.info("开始预测分析...")
            ml_results = self.prediction_models.run_all_predictions(processed_data)
        
        # 8. DMM风险预测（新增）
        risk_results = {}
        if self.risk_predictor:
            self.logger.info("进行DMM风险预测...")
            risk_results = self.risk_predictor.predict_dmm_risk(processed_data)
        
        # 9. 文件影响预测（新增）
        impact_results = {}
        if self.file_impact_predictor and association_results.get('rules_data') is not None:
            self.logger.info("进行文件影响预测...")
            # 选择一个示例文件进行影响分析
            if 'frequent_files' in association_results:
                frequent_files = association_results['frequent_files']
                if frequent_files:
                    target_file = frequent_files[0]  # 选择最频繁的文件
                    impact_results = self.file_impact_predictor.predict_file_impact(
                        processed_data, association_results['rules_data'], target_file
                    )
        
        # 整合所有结果
        results = {
            'repository_url': repo_url,
            'analysis_time': datetime.now(),
            'basic_statistics': basic_stats,
            'raw_data': raw_data,
            'processed_data': processed_data,
            'quality_analysis': quality_results,
            'association_analysis': association_results,
            'clustering_analysis': clustering_results,
            'ml_analysis': ml_results,
            'risk_prediction': risk_results,
            'file_impact': impact_results
        }
        
        self.logger.info("仓库分析完成")
        return results
    
    def generate_report(self, results: Dict[str, Any], output_dir: Path) -> Path:
        """
        生成分析报告
        
        Args:
            results: 分析结果
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        self.logger.info("生成分析报告...")
        report_path = self.report_generator.generate_comprehensive_report(
            results, output_dir
        )
        
        return report_path
    
    def _get_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取基础统计信息"""
        stats = {
            'total_commits': len(data),
            'unique_authors': data['author'].nunique(),
            'date_range': {
                'start': data['author_date'].min(),
                'end': data['author_date'].max()
            },
            'total_files_changed': data['files'].sum(),
            'total_lines_added': data['insertions'].sum(),
            'total_lines_deleted': data['deletions'].sum(),
            'avg_files_per_commit': data['files'].mean(),
            'avg_lines_per_commit': data['lines'].mean(),
        }
        
        # 文件类型统计
        if 'main_file_type' in data.columns:
            file_type_stats = data['main_file_type'].value_counts().to_dict()
            stats['file_type_distribution'] = file_type_stats
        
        # 变更类型统计
        if 'main_change_type' in data.columns:
            change_type_stats = data['main_change_type'].value_counts().to_dict()
            stats['change_type_distribution'] = change_type_stats
        
        return stats

    def analyze_csv_data(self, df: pd.DataFrame, project_name: str) -> Dict[str, Any]:
        """
        分析CSV数据
        
        Args:
            df: 包含提交数据的DataFrame
            project_name: 项目名称
            
        Returns:
            分析结果字典
        """
        self.logger.info(f"开始分析CSV数据，项目: {project_name}")
        
        results = {
            'project_name': project_name,
            'data_source': 'csv',
            'total_commits': len(df),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            # 数据预处理
            self.logger.info("开始数据预处理...")
            processed_df = self.preprocessor.preprocess(df)
            results['preprocessed_commits'] = len(processed_df)
            
            # 代码质量分析
            if self.quality_analyzer:
                self.logger.info("开始代码质量分析...")
                quality_results = self.quality_analyzer.analyze(processed_df)
                results['quality_analysis'] = quality_results
            
            # 关联规则挖掘
            if self.association_miner:
                self.logger.info("开始关联规则挖掘...")
                association_results = self.association_miner.mine_associations(processed_df)
                results['association_analysis'] = association_results
            
            # 聚类分析
            if self.clustering_analyzer:
                self.logger.info("开始聚类分析...")
                clustering_results = self.clustering_analyzer.analyze(processed_df)
                results['clustering_analysis'] = clustering_results
            
            # 预测模型
            if self.prediction_models:
                self.logger.info("开始预测分析...")
                ml_results = self.prediction_models.run_all_predictions(processed_df)
                results['ml_analysis'] = ml_results
            
            self.logger.info("CSV数据分析完成")
            return results
            
        except Exception as e:
            self.logger.error(f"CSV数据分析失败: {str(e)}")
            raise 