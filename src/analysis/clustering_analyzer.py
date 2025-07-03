"""
聚类分析器
对提交数据进行聚类分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False


class ClusteringAnalyzer:
    """聚类分析器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        进行聚类分析
        
        Args:
            data: 预处理后的数据
            
        Returns:
            聚类分析结果
        """
        self.logger.info("开始聚类分析...")
        
        if data.empty:
            return {'error': 'Empty dataset'}
        
        # 准备数据
        features = self._prepare_features(data)
        
        if features.empty:
            return {'error': 'No valid features for clustering'}
        
        results = {}
        
        # 1. 降维分析
        dimensionality_results = self._perform_dimensionality_reduction(features)
        results['dimensionality_reduction'] = dimensionality_results
        
        # 2. K-means聚类
        kmeans_results = self._perform_kmeans_clustering(features)
        results['kmeans_clustering'] = kmeans_results
        
        # 3. Leiden聚类（如果可用）
        if LEIDEN_AVAILABLE:
            leiden_results = self._perform_leiden_clustering(features)
            results['leiden_clustering'] = leiden_results
        
        # 4. 聚类评估
        evaluation_results = self._evaluate_clustering(features, results)
        results['evaluation'] = evaluation_results
        
        self.logger.info("聚类分析完成")
        return results
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备聚类特征"""
        # 选择数值特征进行聚类
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 过滤掉可能的ID列或无用列
        exclude_patterns = ['id', 'index', 'hash']
        numeric_cols = [col for col in numeric_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        if not numeric_cols:
            self.logger.warning("没有找到适合聚类的数值特征")
            return pd.DataFrame()
        
        features = data[numeric_cols].copy()
        
        # 处理缺失值
        features = features.fillna(features.median())
        
        # 移除常数列
        constant_cols = features.columns[features.nunique() <= 1]
        if len(constant_cols) > 0:
            features = features.drop(columns=constant_cols)
            self.logger.info(f"移除了 {len(constant_cols)} 个常数列")
        
        # 移除高度相关的特征
        if len(features.columns) > 1:
            corr_matrix = features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            if high_corr_features:
                features = features.drop(columns=high_corr_features)
                self.logger.info(f"移除了 {len(high_corr_features)} 个高度相关的特征")
        
        self.logger.info(f"聚类特征数量: {features.shape[1]}")
        return features
    
    def _perform_dimensionality_reduction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行降维分析
        
        Args:
            data: 输入数据
            
        Returns:
            降维结果
        """
        self.logger.info("执行降维分析...")
        
        results = {}
        
        try:
            # 选择数值特征
            numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
            
            if numeric_data.empty or len(numeric_data.columns) < 2:
                return {'error': '数据不足以进行降维分析'}
            
            # 标准化数据
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # PCA降维
            pca_results = self._apply_pca(scaled_data, numeric_data.columns)
            results['pca'] = pca_results
            
            # t-SNE降维（如果数据量不太大）
            if len(scaled_data) <= 1000:
                tsne_results = self._apply_tsne(scaled_data)
                results['tsne'] = tsne_results
            
            # UMAP降维（如果可用）
            try:
                import umap
                umap_results = self._apply_umap(scaled_data)
                results['umap'] = umap_results
            except ImportError:
                self.logger.info("UMAP不可用，跳过UMAP降维")
            
            return results
            
        except Exception as e:
            self.logger.error(f"降维分析失败: {str(e)}")
            return {'error': str(e)}
    
    def _perform_kmeans_clustering(self, features: pd.DataFrame) -> Dict[str, Any]:
        """执行K-means聚类"""
        self.logger.info("执行K-means聚类...")
        
        results = {}
        
        # 确定最优聚类数
        max_clusters = min(10, len(features) // 2)
        if max_clusters < 2:
            return {'error': 'Insufficient data for clustering'}
        
        silhouette_scores = []
        inertias = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                # 计算轮廓系数
                silhouette_avg = silhouette_score(features, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                inertias.append(kmeans.inertia_)
                
            except Exception as e:
                self.logger.warning(f"K-means聚类失败 (k={n_clusters}): {str(e)}")
                silhouette_scores.append(0)
                inertias.append(float('inf'))
        
        if silhouette_scores:
            # 选择最优聚类数
            best_k = cluster_range[np.argmax(silhouette_scores)]
            
            # 使用最优k进行最终聚类
            final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            final_labels = final_kmeans.fit_predict(features)
            
            results = {
                'optimal_k': int(best_k),
                'cluster_labels': final_labels.tolist(),
                'silhouette_scores': dict(zip(cluster_range, silhouette_scores)),
                'inertias': dict(zip(cluster_range, inertias)),
                'best_silhouette_score': float(max(silhouette_scores)),
                'cluster_centers': final_kmeans.cluster_centers_.tolist(),
                'cluster_sizes': dict(zip(*np.unique(final_labels, return_counts=True)))
            }
        
        return results
    
    def _perform_leiden_clustering(self, features: pd.DataFrame) -> Dict[str, Any]:
        """执行Leiden聚类"""
        self.logger.info("执行Leiden聚类...")
        
        try:
            # 构建相似性图
            from scipy.spatial.distance import pdist, squareform
            from scipy.sparse import csr_matrix
            
            # 计算距离矩阵
            distances = pdist(features, metric='euclidean')
            distance_matrix = squareform(distances)
            
            # 转换为相似性矩阵（使用高斯核）
            sigma = np.median(distances)
            similarity_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))
            
            # 构建图
            # 只保留最强的连接（阈值化）
            threshold = np.percentile(similarity_matrix, 90)
            adjacency_matrix = (similarity_matrix > threshold).astype(int)
            
            # 创建igraph图
            graph = ig.Graph.Adjacency(adjacency_matrix.tolist())
            
            # 执行Leiden聚类
            partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition)
            
            cluster_labels = np.array(partition.membership)
            
            results = {
                'cluster_labels': cluster_labels.tolist(),
                'modularity': float(partition.modularity),
                'n_clusters': len(set(cluster_labels)),
                'cluster_sizes': dict(zip(*np.unique(cluster_labels, return_counts=True)))
            }
            
            # 计算轮廓系数
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(features, cluster_labels)
                results['silhouette_score'] = float(silhouette_avg)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Leiden聚类失败: {str(e)}")
            return {'error': f'Leiden clustering failed: {str(e)}'}
    
    def _evaluate_clustering(self, features: pd.DataFrame, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估聚类结果"""
        evaluation = {}
        
        # 评估K-means结果
        if 'kmeans_clustering' in clustering_results and 'cluster_labels' in clustering_results['kmeans_clustering']:
            kmeans_labels = clustering_results['kmeans_clustering']['cluster_labels']
            
            if len(set(kmeans_labels)) > 1:
                evaluation['kmeans'] = {
                    'silhouette_score': float(silhouette_score(features, kmeans_labels)),
                    'n_clusters': len(set(kmeans_labels)),
                    'cluster_balance': self._calculate_cluster_balance(kmeans_labels)
                }
        
        # 评估Leiden结果
        if 'leiden_clustering' in clustering_results and 'cluster_labels' in clustering_results['leiden_clustering']:
            leiden_labels = clustering_results['leiden_clustering']['cluster_labels']
            
            if len(set(leiden_labels)) > 1:
                evaluation['leiden'] = {
                    'silhouette_score': float(silhouette_score(features, leiden_labels)),
                    'n_clusters': len(set(leiden_labels)),
                    'cluster_balance': self._calculate_cluster_balance(leiden_labels)
                }
        
        return evaluation
    
    def _calculate_cluster_balance(self, labels: list) -> float:
        """计算聚类平衡度"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if len(unique_labels) <= 1:
            return 1.0
        
        # 计算每个聚类的比例
        proportions = counts / len(labels)
        
        # 计算平衡度（使用熵的概念）
        entropy = -np.sum(proportions * np.log2(proportions + 1e-8))
        max_entropy = np.log2(len(unique_labels))
        
        balance = entropy / max_entropy if max_entropy > 0 else 0
        return float(balance)
    
    def _apply_pca(self, scaled_data: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """应用PCA降维"""
        try:
            n_components = min(3, scaled_data.shape[1])
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'pc1_loading': np.abs(pca.components_[0]),
                'pc2_loading': np.abs(pca.components_[1]) if n_components > 1 else 0
            }).sort_values('pc1_loading', ascending=False)
            
            return {
                'coordinates': pca_result.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': float(pca.explained_variance_ratio_.sum()),
                'feature_importance': feature_importance.head(10).to_dict('records')
            }
        except Exception as e:
            self.logger.error(f"PCA降维失败: {str(e)}")
            return {'error': str(e)}
    
    def _apply_tsne(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """应用t-SNE降维"""
        try:
            perplexity = min(30, len(scaled_data) - 1, 5)
            if perplexity < 1:
                return {'error': '数据量不足以进行t-SNE'}
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_result = tsne.fit_transform(scaled_data)
            
            return {
                'coordinates': tsne_result.tolist(),
                'perplexity': perplexity
            }
        except Exception as e:
            self.logger.error(f"t-SNE降维失败: {str(e)}")
            return {'error': str(e)}
    
    def _apply_umap(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """应用UMAP降维"""
        try:
            import umap
            n_neighbors = min(15, len(scaled_data) - 1)
            if n_neighbors < 2:
                return {'error': '数据量不足以进行UMAP'}
            
            umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
            umap_result = umap_reducer.fit_transform(scaled_data)
            
            return {
                'coordinates': umap_result.tolist(),
                'n_neighbors': n_neighbors
            }
        except Exception as e:
            self.logger.error(f"UMAP降维失败: {str(e)}")
            return {'error': str(e)} 