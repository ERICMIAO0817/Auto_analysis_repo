"""
机器学习预测模型
包含代码质量预测、文件影响预测等任务
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)

try:
    from imblearn.over_sampling import SMOTE
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False


class PredictionModels:
    """机器学习预测模型"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def run_all_predictions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        运行所有预测任务
        
        Args:
            data: 预处理后的数据
            
        Returns:
            所有预测结果
        """
        self.logger.info("开始机器学习预测分析...")
        
        results = {}
        
        # 1. DMM质量预测（回归）
        dmm_regression_results = self._predict_dmm_regression(data)
        if dmm_regression_results:
            results['dmm_regression'] = dmm_regression_results
        
        # 2. DMM质量分类预测
        dmm_classification_results = self._predict_dmm_classification(data)
        if dmm_classification_results:
            results['dmm_classification'] = dmm_classification_results
        
        # 3. 文件类型预测
        file_type_results = self._predict_file_type(data)
        if file_type_results:
            results['file_type_prediction'] = file_type_results
        
        # 4. 变更类型预测
        change_type_results = self._predict_change_type(data)
        if change_type_results:
            results['change_type_prediction'] = change_type_results
        
        self.logger.info("机器学习预测分析完成")
        return results
    
    def _predict_dmm_regression(self, data: pd.DataFrame) -> Dict[str, Any]:
        """DMM指标回归预测"""
        dmm_targets = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        existing_targets = [col for col in dmm_targets if col in data.columns]
        
        if not existing_targets:
            self.logger.warning("没有找到DMM目标变量，跳过DMM回归预测")
            return {}
        
        results = {}
        
        for target in existing_targets:
            self.logger.info(f"进行 {target} 回归预测...")
            
            # 准备数据
            X, y = self._prepare_regression_data(data, target)
            
            if X.empty or len(y) == 0:
                continue
            
            # 训练和评估模型
            target_results = self._train_regression_models(X, y, target)
            results[target] = target_results
        
        return results
    
    def _predict_dmm_classification(self, data: pd.DataFrame) -> Dict[str, Any]:
        """DMM质量分类预测"""
        dmm_targets = ['dmm_unit_complexity']  # 主要关注复杂度
        existing_targets = [col for col in dmm_targets if col in data.columns]
        
        if not existing_targets:
            self.logger.warning("没有找到DMM目标变量，跳过DMM分类预测")
            return {}
        
        results = {}
        
        for target in existing_targets:
            self.logger.info(f"进行 {target} 分类预测...")
            
            # 准备数据（转换为分类问题）
            X, y = self._prepare_classification_data(data, target)
            
            if X.empty or len(y) == 0:
                continue
            
            # 训练和评估模型
            target_results = self._train_classification_models(X, y, f"{target}_class")
            results[f"{target}_classification"] = target_results
        
        return results
    
    def _predict_file_type(self, data: pd.DataFrame) -> Dict[str, Any]:
        """文件类型预测"""
        if 'main_file_type' not in data.columns:
            self.logger.warning("没有找到main_file_type列，跳过文件类型预测")
            return {}
        
        self.logger.info("进行文件类型预测...")
        
        # 准备数据
        X, y = self._prepare_classification_data(data, 'main_file_type', is_categorical=True)
        
        if X.empty or len(y) == 0:
            return {}
        
        # 训练和评估模型
        results = self._train_classification_models(X, y, 'main_file_type')
        return results
    
    def _predict_change_type(self, data: pd.DataFrame) -> Dict[str, Any]:
        """变更类型预测"""
        if 'main_change_type' not in data.columns:
            self.logger.warning("没有找到main_change_type列，跳过变更类型预测")
            return {}
        
        self.logger.info("进行变更类型预测...")
        
        # 准备数据
        X, y = self._prepare_classification_data(data, 'main_change_type', is_categorical=True)
        
        if X.empty or len(y) == 0:
            return {}
        
        # 训练和评估模型
        results = self._train_classification_models(X, y, 'main_change_type')
        return results
    
    def _prepare_regression_data(self, data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """准备回归数据"""
        # 移除目标变量和其他不相关列
        exclude_cols = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # 选择数值特征
        X = data[feature_cols].select_dtypes(include=[np.number])
        y = data[target]
        
        # 移除缺失值
        valid_indices = y.notna() & X.notna().all(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def _prepare_classification_data(self, data: pd.DataFrame, target: str, is_categorical: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """准备分类数据"""
        if is_categorical:
            # 对于分类目标变量，直接使用
            y = data[target]
            exclude_cols = [target]
        else:
            # 对于数值目标变量，转换为分类
            target_data = data[target]
            if target_data.notna().sum() == 0:
                return pd.DataFrame(), pd.Series()
            
            # 使用四分位数进行分类
            q25, q75 = target_data.quantile([0.25, 0.75])
            
            def classify_quality(value):
                if pd.isna(value):
                    return None
                elif value <= q25:
                    return 'low'
                elif value <= q75:
                    return 'medium'
                else:
                    return 'high'
            
            y = target_data.apply(classify_quality)
            exclude_cols = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        
        # 准备特征
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        X = data[feature_cols].select_dtypes(include=[np.number])
        
        # 移除缺失值
        valid_indices = y.notna() & X.notna().all(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # 过滤掉类别数量太少的情况
        if len(y.unique()) < 2:
            return pd.DataFrame(), pd.Series()
        
        return X, y
    
    def _train_regression_models(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict[str, Any]:
        """训练回归模型"""
        if len(X) < 10:  # 数据太少
            return {'error': 'Insufficient data for regression'}
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        results = {}
        
        for model_name, model in models.items():
            try:
                # 训练模型
                if model_name in ['SVR', 'MLP']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # 评估模型
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_results = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2_score': float(r2),
                    'rmse': float(np.sqrt(mse))
                }
                
                # 特征重要性（如果可用）
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                    # 只保留前10个重要特征
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    model_results['feature_importance'] = dict(sorted_features[:10])
                
                results[model_name] = model_results
                
            except Exception as e:
                self.logger.warning(f"{model_name} 回归训练失败: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _train_classification_models(self, X: np.ndarray, y: np.ndarray, target_name: str) -> Dict[str, Any]:
        """
        训练分类模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            target_name: 目标变量名称
            
        Returns:
            分类结果字典
        """
        results = {
            'target': target_name,
            'total_samples': len(X),
            'models': {}
        }
        
        try:
            # 检查类别分布
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(unique_classes, class_counts))
            results['class_distribution'] = class_distribution
            
            self.logger.info(f"{target_name} 类别分布: {class_distribution}")
            
            # 检查是否有足够的样本进行训练
            min_samples_per_class = 2
            insufficient_classes = [cls for cls, count in class_distribution.items() if count < min_samples_per_class]
            
            if len(insufficient_classes) > 0:
                self.logger.warning(f"{target_name} 中类别 {insufficient_classes} 样本数量不足，跳过分类训练")
                results['error'] = f"类别样本数量不足: {insufficient_classes}"
                return results
            
            # 如果只有一个类别，跳过训练
            if len(unique_classes) < 2:
                self.logger.warning(f"{target_name} 只有一个类别，跳过分类训练")
                results['error'] = "只有一个类别，无法进行分类"
                return results
            
            # 数据分割 - 使用stratify确保每个类别都有足够的样本
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                # 如果stratify失败，使用简单分割
                self.logger.warning(f"分层采样失败，使用简单随机分割: {str(e)}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # 定义分类模型
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'MLP': MLPClassifier(random_state=42, max_iter=1000)
            }
            
            # 训练和评估每个模型
            for name, model in models.items():
                try:
                    self.logger.info(f"训练 {name} 分类模型...")
                    
                    # 训练模型
                    model.fit(X_train, y_train)
                    
                    # 预测
                    y_pred = model.predict(X_test)
                    y_pred_proba = None
                    
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_test)
                    except Exception:
                        pass
                    
                    # 计算评估指标
                    model_results = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    }
                    
                    # 添加分类报告
                    try:
                        model_results['classification_report'] = classification_report(
                            y_test, y_pred, output_dict=True, zero_division=0
                        )
                    except Exception as e:
                        self.logger.warning(f"生成分类报告失败: {str(e)}")
                    
                    # 添加混淆矩阵
                    try:
                        cm = confusion_matrix(y_test, y_pred)
                        model_results['confusion_matrix'] = cm.tolist()
                    except Exception as e:
                        self.logger.warning(f"生成混淆矩阵失败: {str(e)}")
                    
                    results['models'][name] = model_results
                    self.logger.info(f"{name} 分类完成，准确率: {model_results['accuracy']:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"{name} 分类训练失败: {str(e)}")
                    results['models'][name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"分类模型训练失败: {str(e)}")
            results['error'] = str(e)
            return results 