"""
关联规则挖掘器
分析文件间的协同变更模式
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import ast
from collections import defaultdict

try:
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


class AssociationMiner:
    """文件关联规则挖掘器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def mine_associations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        挖掘文件关联规则
        
        Args:
            data: 原始数据（包含modified_files列）
            
        Returns:
            关联规则挖掘结果
        """
        self.logger.info("开始文件关联规则挖掘...")
        
        if not MLXTEND_AVAILABLE:
            self.logger.warning("MLxtend库不可用，跳过关联规则挖掘")
            return {'error': 'MLxtend library not available'}
        
        if 'modified_files' not in data.columns:
            self.logger.warning("没有找到modified_files列，跳过关联规则挖掘")
            return {'error': 'modified_files column not found'}
        
        try:
            # 1. 构建文件-提交矩阵
            file_commit_matrix = self._build_file_commit_matrix(data)
            
            if file_commit_matrix.empty:
                return {'error': 'Unable to build file-commit matrix'}
            
            # 2. 挖掘频繁项集
            frequent_itemsets = self._mine_frequent_itemsets(file_commit_matrix)
            
            # 3. 生成关联规则
            rules = self._generate_association_rules(frequent_itemsets, file_commit_matrix.shape[0])
            
            # 4. 分析结果
            analysis_results = self._analyze_association_results(rules, frequent_itemsets)
            
            results = {
                'frequent_itemsets': frequent_itemsets.to_dict('records') if not frequent_itemsets.empty else [],
                'association_rules': rules.to_dict('records') if not rules.empty else [],
                'analysis': analysis_results,
                'file_statistics': self._get_file_statistics(file_commit_matrix)
            }
            
            self.logger.info(f"关联规则挖掘完成，发现 {len(rules)} 条规则")
            return results
            
        except Exception as e:
            self.logger.error(f"关联规则挖掘失败: {str(e)}")
            return {'error': f'Association mining failed: {str(e)}'}
    
    def _build_file_commit_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """构建文件-提交二值矩阵"""
        self.logger.info("构建文件-提交矩阵...")
        
        # 提取所有文件名
        all_files = set()
        file_commit_data = []
        
        for idx, row in data.iterrows():
            try:
                # 解析modified_files字符串
                modified_files_str = row['modified_files']
                if pd.isna(modified_files_str) or modified_files_str == '':
                    continue
                
                # 尝试解析为列表
                if isinstance(modified_files_str, str):
                    if modified_files_str.startswith('[') and modified_files_str.endswith(']'):
                        # 字符串形式的列表
                        modified_files = ast.literal_eval(modified_files_str)
                    else:
                        # 简单的文件名
                        modified_files = [modified_files_str]
                else:
                    modified_files = [str(modified_files_str)]
                
                # 过滤和清理文件名
                cleaned_files = []
                for file in modified_files:
                    if isinstance(file, str) and file.strip():
                        # 只保留文件名，去掉路径
                        filename = file.split('/')[-1]
                        if filename and not filename.startswith('.'):  # 过滤隐藏文件
                            cleaned_files.append(filename)
                
                if cleaned_files:
                    all_files.update(cleaned_files)
                    file_commit_data.append((idx, cleaned_files))
                    
            except Exception as e:
                self.logger.debug(f"解析提交 {idx} 的文件列表失败: {str(e)}")
                continue
        
        if not all_files:
            self.logger.warning("没有找到有效的文件信息")
            return pd.DataFrame()
        
        # 只保留出现频率较高的文件（减少噪音）
        file_counts = defaultdict(int)
        for _, files in file_commit_data:
            for file in files:
                file_counts[file] += 1
        
        # 过滤出现次数少于2次的文件
        frequent_files = [file for file, count in file_counts.items() if count >= 2]
        
        if len(frequent_files) < 2:
            self.logger.warning("频繁文件数量不足，无法进行关联分析")
            return pd.DataFrame()
        
        # 限制文件数量以提高效率
        if len(frequent_files) > 100:
            # 按频率排序，取前100个
            sorted_files = sorted(frequent_files, key=lambda x: file_counts[x], reverse=True)
            frequent_files = sorted_files[:100]
        
        self.logger.info(f"选择了 {len(frequent_files)} 个频繁文件进行分析")
        
        # 构建二值矩阵
        matrix_data = []
        for commit_idx, files in file_commit_data:
            row = {}
            for file in frequent_files:
                row[file] = 1 if file in files else 0
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(matrix_data)
        
        # 过滤掉全为0的行
        matrix_df = matrix_df[matrix_df.sum(axis=1) > 0]
        
        self.logger.info(f"构建的矩阵大小: {matrix_df.shape}")
        return matrix_df
    
    def _mine_frequent_itemsets(self, matrix: pd.DataFrame, min_support=0.01) -> pd.DataFrame:
        """挖掘频繁项集"""
        self.logger.info("挖掘频繁项集...")
        
        try:
            # 使用FP-Growth算法
            frequent_itemsets = fpgrowth(matrix, min_support=min_support, use_colnames=True)
            
            if frequent_itemsets.empty:
                # 如果没有找到频繁项集，降低支持度阈值
                min_support = 0.005
                self.logger.info(f"降低支持度阈值到 {min_support}")
                frequent_itemsets = fpgrowth(matrix, min_support=min_support, use_colnames=True)
            
            self.logger.info(f"发现 {len(frequent_itemsets)} 个频繁项集")
            return frequent_itemsets
            
        except Exception as e:
            self.logger.error(f"频繁项集挖掘失败: {str(e)}")
            return pd.DataFrame()
    
    def _generate_association_rules(self, frequent_itemsets: pd.DataFrame, total_transactions: int) -> pd.DataFrame:
        """生成关联规则"""
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        self.logger.info("生成关联规则...")
        
        try:
            # 生成关联规则
            rules = association_rules(
                frequent_itemsets,
                metric="lift",
                min_threshold=1.0,
                num_itemsets=total_transactions
            )
            
            if not rules.empty:
                # 按lift值排序
                rules = rules.sort_values('lift', ascending=False)
                
                # 只保留前50条规则以提高效率
                if len(rules) > 50:
                    rules = rules.head(50)
            
            self.logger.info(f"生成 {len(rules)} 条关联规则")
            return rules
            
        except Exception as e:
            self.logger.error(f"关联规则生成失败: {str(e)}")
            return pd.DataFrame()
    
    def _analyze_association_results(self, rules: pd.DataFrame, frequent_itemsets: pd.DataFrame) -> Dict[str, Any]:
        """分析关联规则结果"""
        analysis = {}
        
        if not rules.empty:
            # 规则统计
            analysis['rule_statistics'] = {
                'total_rules': len(rules),
                'avg_confidence': float(rules['confidence'].mean()),
                'avg_lift': float(rules['lift'].mean()),
                'avg_support': float(rules['support'].mean()),
                'max_lift': float(rules['lift'].max()),
                'min_lift': float(rules['lift'].min())
            }
            
            # 最强关联规则（前10条）
            top_rules = rules.head(10)
            strong_associations = []
            
            for _, rule in top_rules.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                strong_associations.append({
                    'antecedents': antecedents,
                    'consequents': consequents,
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift']),
                    'support': float(rule['support'])
                })
            
            analysis['strong_associations'] = strong_associations
        
        if not frequent_itemsets.empty:
            # 频繁项集统计
            analysis['itemset_statistics'] = {
                'total_itemsets': len(frequent_itemsets),
                'avg_support': float(frequent_itemsets['support'].mean()),
                'max_support': float(frequent_itemsets['support'].max()),
                'itemset_size_distribution': frequent_itemsets['itemsets'].apply(len).value_counts().to_dict()
            }
        
        return analysis
    
    def _get_file_statistics(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """获取文件统计信息"""
        if matrix.empty:
            return {}
        
        # 文件修改频率
        file_frequencies = matrix.sum().sort_values(ascending=False)
        
        # 文件共现统计
        cooccurrence_stats = {}
        total_commits = len(matrix)
        
        for file in file_frequencies.head(20).index:  # 只分析前20个最频繁的文件
            file_commits = matrix[matrix[file] == 1]
            if len(file_commits) > 1:
                # 计算与其他文件的共现频率
                cooccurrences = file_commits.sum() - 1  # 减去自己
                cooccurrences = cooccurrences[cooccurrences > 0].sort_values(ascending=False)
                
                if len(cooccurrences) > 0:
                    cooccurrence_stats[file] = {
                        'total_modifications': int(file_frequencies[file]),
                        'modification_rate': float(file_frequencies[file] / total_commits),
                        'top_cooccurrences': cooccurrences.head(5).to_dict()
                    }
        
        return {
            'total_files': len(file_frequencies),
            'total_commits': total_commits,
            'most_modified_files': file_frequencies.head(10).to_dict(),
            'file_cooccurrences': cooccurrence_stats
        }
    
    def _create_association_network(self, rules: pd.DataFrame) -> Dict[str, Any]:
        """
        创建关联规则网络图
        
        Args:
            rules: 关联规则DataFrame
            
        Returns:
            网络图数据
        """
        try:
            import networkx as nx
            
            # 创建有向图
            G = nx.DiGraph()
            
            # 限制规则数量以提高可读性
            top_rules = rules.nlargest(20, 'lift')  # 选择前20个最强关联规则
            
            # 添加节点和边
            for _, rule in top_rules.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                # 简化文件名显示
                for ant in antecedents:
                    ant_short = self._shorten_filename(ant)
                    G.add_node(ant_short, node_type='antecedent')
                
                for cons in consequents:
                    cons_short = self._shorten_filename(cons)
                    G.add_node(cons_short, node_type='consequent')
                
                # 添加边
                for ant in antecedents:
                    for cons in consequents:
                        ant_short = self._shorten_filename(ant)
                        cons_short = self._shorten_filename(cons)
                        G.add_edge(ant_short, cons_short, 
                                 weight=rule['lift'], 
                                 confidence=rule['confidence'],
                                 support=rule['support'])
            
            # 计算网络指标
            metrics = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G) if G.number_of_nodes() > 0 else 0
            }
            
            return {
                'graph': G,
                'metrics': metrics,
                'pos': nx.spring_layout(G, k=2, iterations=50) if G.number_of_nodes() > 0 else {}
            }
            
        except Exception as e:
            self.logger.error(f"创建关联网络失败: {str(e)}")
            return {'error': str(e)}
    
    def _shorten_filename(self, filename: str, max_length: int = 15) -> str:
        """缩短文件名以提高可读性"""
        if len(filename) <= max_length:
            return filename
        
        # 保留文件扩展名
        if '.' in filename:
            name, ext = filename.rsplit('.', 1)
            if len(ext) < 5:  # 常见扩展名
                available_length = max_length - len(ext) - 1
                if available_length > 3:
                    return name[:available_length] + '.' + ext
        
        return filename[:max_length-3] + '...' 