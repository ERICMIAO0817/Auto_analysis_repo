"""
代码质量分析器
基于DMM模型分析代码质量
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging


class QualityAnalyzer:
    """代码质量分析器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析代码质量
        
        Args:
            data: 预处理后的数据
            
        Returns:
            质量分析结果
        """
        self.logger.info("开始代码质量分析...")
        
        results = {}
        
        # DMM指标分析
        dmm_results = self._analyze_dmm_metrics(data)
        results['dmm_analysis'] = dmm_results
        
        # 质量趋势分析
        if 'author_date' in data.columns:
            trend_results = self._analyze_quality_trends(data)
            results['quality_trends'] = trend_results
        
        # 质量分布分析
        distribution_results = self._analyze_quality_distribution(data)
        results['quality_distribution'] = distribution_results
        
        # 计算综合质量分数
        quality_score = self._calculate_quality_score(data)
        results['overall_quality_score'] = quality_score
        
        self.logger.info("代码质量分析完成")
        return results
    
    def _analyze_dmm_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析DMM指标"""
        dmm_cols = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        existing_dmm_cols = [col for col in dmm_cols if col in data.columns]
        
        if not existing_dmm_cols:
            return {'error': 'No DMM metrics found'}
        
        results = {}
        
        for col in existing_dmm_cols:
            if data[col].notna().sum() > 0:
                results[col] = {
                    'mean': float(data[col].mean()),
                    'median': float(data[col].median()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'q25': float(data[col].quantile(0.25)),
                    'q75': float(data[col].quantile(0.75))
                }
        
        # 计算DMM综合分数
        if existing_dmm_cols:
            # 简单的加权平均（可以根据需要调整权重）
            weights = {'dmm_unit_size': 0.3, 'dmm_unit_complexity': 0.4, 'dmm_unit_interfacing': 0.3}
            
            weighted_scores = []
            for col in existing_dmm_cols:
                if col in weights and data[col].notna().sum() > 0:
                    # 标准化到0-1范围
                    normalized = (data[col] - data[col].min()) / (data[col].max() - data[col].min() + 1e-8)
                    weighted_scores.append(normalized * weights[col])
            
            if weighted_scores:
                composite_score = sum(weighted_scores)
                results['composite_dmm_score'] = {
                    'mean': float(composite_score.mean()),
                    'median': float(composite_score.median()),
                    'std': float(composite_score.std())
                }
        
        return results
    
    def _analyze_quality_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析质量趋势"""
        # 这里需要原始数据中的时间信息
        # 由于预处理后时间列可能被删除，这里做简化处理
        results = {
            'note': 'Quality trend analysis requires temporal data'
        }
        
        # 如果有时间相关特征，可以进行趋势分析
        time_features = ['author_year', 'author_month']
        existing_time_features = [col for col in time_features if col in data.columns]
        
        if existing_time_features and 'dmm_unit_complexity' in data.columns:
            # 按年份分组分析质量变化
            if 'author_year' in data.columns:
                yearly_quality = data.groupby('author_year')['dmm_unit_complexity'].agg([
                    'mean', 'median', 'count'
                ]).to_dict('index')
                results['yearly_trends'] = yearly_quality
        
        return results
    
    def _analyze_quality_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析质量分布"""
        results = {}
        
        dmm_cols = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        existing_dmm_cols = [col for col in dmm_cols if col in data.columns]
        
        for col in existing_dmm_cols:
            if data[col].notna().sum() > 0:
                # 分位数分析
                percentiles = [10, 25, 50, 75, 90, 95, 99]
                percentile_values = {}
                for p in percentiles:
                    percentile_values[f'p{p}'] = float(data[col].quantile(p/100))
                
                results[f'{col}_distribution'] = percentile_values
                
                # 质量等级分类
                q25, q75 = data[col].quantile([0.25, 0.75])
                
                def classify_quality(value):
                    if pd.isna(value):
                        return 'unknown'
                    elif value <= q25:
                        return 'high'  # 低复杂度 = 高质量
                    elif value <= q75:
                        return 'medium'
                    else:
                        return 'low'   # 高复杂度 = 低质量
                
                quality_classes = data[col].apply(classify_quality)
                class_distribution = quality_classes.value_counts().to_dict()
                results[f'{col}_quality_classes'] = class_distribution
        
        return results
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算综合质量分数"""
        dmm_cols = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        existing_dmm_cols = [col for col in dmm_cols if col in data.columns]
        
        if not existing_dmm_cols:
            return {'error': 'No DMM metrics available for quality score calculation'}
        
        # 计算每个指标的标准化分数（越低越好，所以取倒数）
        quality_scores = []
        
        for col in existing_dmm_cols:
            if data[col].notna().sum() > 0:
                # 标准化并反转（使得低复杂度对应高分数）
                col_data = data[col].fillna(data[col].median())
                normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min() + 1e-8)
                inverted_score = 1 - normalized  # 反转分数
                quality_scores.append(inverted_score)
        
        if quality_scores:
            # 计算平均质量分数
            overall_score = np.mean(quality_scores, axis=0)
            
            return {
                'mean_score': float(overall_score.mean()),
                'median_score': float(np.median(overall_score)),
                'std_score': float(overall_score.std()),
                'score_distribution': {
                    'excellent': int((overall_score >= 0.8).sum()),
                    'good': int(((overall_score >= 0.6) & (overall_score < 0.8)).sum()),
                    'fair': int(((overall_score >= 0.4) & (overall_score < 0.6)).sum()),
                    'poor': int((overall_score < 0.4).sum())
                }
            }
        
        return {'error': 'Unable to calculate quality score'} 