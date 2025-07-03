"""
DMM风险预测器
基于机器学习模型预测代码质量风险
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# DataPreprocessor will be imported when needed


class RiskPredictor:
    """提交风险预测器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        plt.rcParams['font.family'] = 'Songti SC'
        
    def predict_dmm_risk(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        预测DMM风险分数
        
        Args:
            data: 包含提交数据的DataFrame
            
        Returns:
            DMM风险预测结果
        """
        self.logger.info("开始DMM风险预测分析...")
        
        # 数据预处理
        from utils.data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(data.copy())
        
        results = {
            'dmm_regression': {},
            'dmm_classification': {},
            'feature_importance': {}
        }
        
        # 1. DMM回归预测（预测具体分数）
        dmm_regression_results = self._predict_dmm_regression(processed_data)
        results['dmm_regression'] = dmm_regression_results
        
        # 2. DMM分类预测（预测风险等级）
        dmm_classification_results = self._predict_dmm_classification(processed_data)
        results['dmm_classification'] = dmm_classification_results
        
        # 3. 生成风险预测可视化
        self._plot_dmm_predictions(dmm_regression_results, dmm_classification_results)
        
        return results
    
    def _predict_dmm_regression(self, data: pd.DataFrame) -> Dict[str, Any]:
        """DMM分数回归预测"""
        self.logger.info("进行DMM分数回归预测...")
        
        # 检查DMM字段是否存在
        dmm_fields = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        missing_fields = [field for field in dmm_fields if field not in data.columns]
        
        if missing_fields:
            self.logger.warning(f"缺少DMM字段: {missing_fields}")
            return {'error': f'缺少DMM字段: {missing_fields}'}
        
        # 检查DMM字段是否有有效数据
        dmm_data_counts = {}
        for field in dmm_fields:
            non_null_count = data[field].notna().sum()
            non_zero_count = (data[field] != 0).sum()
            dmm_data_counts[field] = {'non_null': non_null_count, 'non_zero': non_zero_count}
            self.logger.info(f"{field}: {non_null_count} 个非空值, {non_zero_count} 个非零值")
        
        # 检查是否有足够的有效DMM数据
        total_valid_dmm = sum(1 for field in dmm_fields if dmm_data_counts[field]['non_null'] > 0)
        
        if total_valid_dmm == 0:
            self.logger.warning("所有DMM字段都为空，无法进行DMM预测")
            return {
                'error': '所有DMM字段都为空',
                'suggestion': '该项目可能不支持DMM指标，建议使用其他分析功能',
                'dmm_data_summary': dmm_data_counts
            }
        
        # 计算DMM综合分数
        valid_dmm_fields = [field for field in dmm_fields if dmm_data_counts[field]['non_null'] > 0]
        
        if len(valid_dmm_fields) < 3:
            self.logger.warning(f"只有 {len(valid_dmm_fields)} 个DMM字段有数据: {valid_dmm_fields}")
            # 使用可用的字段计算平均值
            data['dmm_score'] = data[valid_dmm_fields].mean(axis=1)
        else:
            data['dmm_score'] = (data['dmm_unit_size'] + data['dmm_unit_complexity'] + data['dmm_unit_interfacing']) / 3
        
        # 检查计算出的DMM分数
        valid_dmm_scores = data['dmm_score'].notna().sum()
        if valid_dmm_scores < 10:
            return {
                'error': f'有效DMM分数不足: 只有 {valid_dmm_scores} 个有效值',
                'suggestion': '需要至少10个有效的DMM分数才能进行预测',
                'dmm_data_summary': dmm_data_counts
            }
        
        # 准备特征和目标变量
        X, y = self._prepare_dmm_data(data, target='dmm_score')
        
        if len(X) < 10:
            return {
                'error': f'预处理后数据量不足: {len(X)} 行',
                'suggestion': '需要更多的有效数据进行预测',
                'dmm_data_summary': dmm_data_counts
            }
        
        # 使用5折交叉验证
        results = self._train_regression_models_cv(X, y, 'DMM分数')
        results['dmm_data_summary'] = dmm_data_counts
        
        return results
    
    def _predict_dmm_classification(self, data: pd.DataFrame) -> Dict[str, Any]:
        """DMM风险等级分类预测"""
        self.logger.info("进行DMM风险等级分类预测...")
        
        if 'dmm_score' not in data.columns:
            return {'error': '需要先计算DMM分数'}
        
        # 将DMM分数转换为风险等级（参考您的代码逻辑）
        data['dmm_risk_level'] = self._categorize_dmm_risk(data['dmm_score'])
        
        # 准备特征和目标变量
        X, y = self._prepare_dmm_data(data, target='dmm_risk_level')
        
        if len(X) < 10:
            return {'error': '数据量不足'}
        
        # 使用5折交叉验证
        results = self._train_classification_models_cv(X, y, 'DMM风险等级')
        
        return results
    
    def _categorize_dmm_risk(self, dmm_scores: pd.Series) -> pd.Series:
        """将DMM分数转换为风险等级"""
        # 使用分位数划分风险等级
        q25 = dmm_scores.quantile(0.25)
        q75 = dmm_scores.quantile(0.75)
        
        def classify_risk(score):
            if score <= q25:
                return 'low'
            elif score <= q75:
                return 'medium'
            else:
                return 'high'
        
        return dmm_scores.apply(classify_risk)
    
    def _prepare_dmm_data(self, data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """准备DMM预测数据"""
        # 排除目标变量和DMM原始字段
        exclude_cols = [target, 'dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
        if target != 'dmm_score':
            exclude_cols.append('dmm_score')
        
        # 选择数值特征
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        X = data[feature_cols].select_dtypes(include=[np.number])
        y = data[target]
        
        # 移除缺失值
        valid_indices = y.notna() & X.notna().all(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def _train_regression_models_cv(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict[str, Any]:
        """使用交叉验证训练回归模型"""
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': SVR(kernel='rbf'),
            'MLP': MLPRegressor(hidden_layer_sizes=(64, 128, 64), random_state=42, max_iter=1000)
        }
        
        results = {'models': {}, 'feature_importance': {}}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            self.logger.info(f"训练{model_name}回归模型...")
            
            fold_results = []
            feature_importances = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # 标准化（对SVM和MLP）
                if model_name in ['SVM', 'MLP']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # 评估
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                fold_results.append({
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                })
                
                # 特征重要性
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
            
            # 计算平均结果
            avg_results = {}
            for metric in ['mse', 'mae', 'r2', 'rmse']:
                values = [fold[metric] for fold in fold_results]
                avg_results[f'avg_{metric}'] = np.mean(values)
                avg_results[f'std_{metric}'] = np.std(values)
            
            results['models'][model_name] = avg_results
            
            # 平均特征重要性
            if feature_importances:
                avg_importance = np.mean(feature_importances, axis=0)
                feature_importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
                results['feature_importance'][model_name] = feature_importance_df.head(10).to_dict('records')
        
        return results
    
    def _train_classification_models_cv(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict[str, Any]:
        """使用交叉验证训练分类模型"""
        # 对标签进行编码
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True),
            'MLP': MLPClassifier(hidden_layer_sizes=(64, 128, 64), random_state=42, max_iter=1000)
        }
        
        results = {'models': {}, 'feature_importance': {}, 'label_mapping': dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            self.logger.info(f"训练{model_name}分类模型...")
            
            fold_results = []
            feature_importances = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
                
                # 标准化（对SVM和MLP）
                if model_name in ['SVM', 'MLP']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # 评估
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                fold_results.append({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                
                # 特征重要性
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
            
            # 计算平均结果
            avg_results = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                values = [fold[metric] for fold in fold_results]
                avg_results[f'avg_{metric}'] = np.mean(values)
                avg_results[f'std_{metric}'] = np.std(values)
            
            results['models'][model_name] = avg_results
            
            # 平均特征重要性
            if feature_importances:
                avg_importance = np.mean(feature_importances, axis=0)
                feature_importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
                results['feature_importance'][model_name] = feature_importance_df.head(10).to_dict('records')
        
        return results
    
    def _plot_dmm_predictions(self, regression_results: Dict, classification_results: Dict):
        """绘制DMM预测结果可视化"""
        # 设置字体以避免负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 回归模型性能对比
        if 'models' in regression_results:
            models = list(regression_results['models'].keys())
            r2_scores = [regression_results['models'][model]['avg_r2'] for model in models]
            
            axes[0, 0].bar(models, r2_scores, color='skyblue')
            axes[0, 0].set_title('DMM回归模型R²分数对比')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 分类模型性能对比
        if 'models' in classification_results:
            models = list(classification_results['models'].keys())
            f1_scores = [classification_results['models'][model]['avg_f1'] for model in models]
            
            axes[0, 1].bar(models, f1_scores, color='lightcoral')
            axes[0, 1].set_title('DMM分类模型F1分数对比')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 特征重要性（回归）
        if 'feature_importance' in regression_results and 'RandomForest' in regression_results['feature_importance']:
            importance_data = regression_results['feature_importance']['RandomForest']
            features = [item['feature'] for item in importance_data]
            importances = [item['importance'] for item in importance_data]
            
            axes[1, 0].barh(features, importances, color='lightgreen')
            axes[1, 0].set_title('DMM回归特征重要性 (RandomForest)')
            axes[1, 0].set_xlabel('Importance')
        
        # 4. 特征重要性（分类）
        if 'feature_importance' in classification_results and 'RandomForest' in classification_results['feature_importance']:
            importance_data = classification_results['feature_importance']['RandomForest']
            features = [item['feature'] for item in importance_data]
            importances = [item['importance'] for item in importance_data]
            
            axes[1, 1].barh(features, importances, color='gold')
            axes[1, 1].set_title('DMM分类特征重要性 (RandomForest)')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('dmm_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show() 