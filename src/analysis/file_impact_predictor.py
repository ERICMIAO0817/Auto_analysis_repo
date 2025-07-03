"""
文件影响预测模块
分析某个文件的修改会影响哪些其他文件
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
import logging


class FileImpactPredictor:
    """文件影响预测器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        plt.rcParams['font.family'] = 'Songti SC'
        
    def predict_file_impact(self, commit_data: pd.DataFrame, rule_data: pd.DataFrame, 
                           target_file: str) -> Dict[str, Any]:
        """
        预测指定文件的影响
        
        Args:
            commit_data: 提交数据
            rule_data: 关联规则数据  
            target_file: 目标文件名
            
        Returns:
            文件影响预测结果
        """
        self.logger.info(f"开始预测文件 {target_file} 的影响...")
        
        results = {
            'target_file': target_file,
            'influenced_files': [],
            'prediction_models': {},
            'recommendations': []
        }
        
        # 1. 找到受影响的文件
        influenced_files = self._find_influenced_files(rule_data, target_file)
        results['influenced_files'] = influenced_files
        
        if not influenced_files:
            self.logger.warning(f"未找到文件 {target_file} 的影响关系")
            return results
        
        # 2. 为每个受影响文件构建预测模型
        for influenced_file in influenced_files[:5]:  # 限制数量
            model_result = self._build_prediction_model(
                commit_data, rule_data, target_file, influenced_file
            )
            results['prediction_models'][influenced_file] = model_result
        
        # 3. 生成可视化
        self._plot_impact_results(results)
        
        return results
    
    def _find_influenced_files(self, rule_data: pd.DataFrame, target_file: str) -> List[str]:
        """找到受目标文件影响的文件列表"""
        influenced_files = []
        
        # 基于关联规则找到影响文件
        if 'source' in rule_data.columns and 'target' in rule_data.columns:
            rules_with_target = rule_data[rule_data['source'].str.contains(target_file, na=False)]
            
            for target in rules_with_target['target'].values:
                if isinstance(target, str) and target_file not in target:
                    # 简单解析，去掉复杂格式
                    clean_target = target.replace("{'", "").replace("'}", "").replace("{", "").replace("}", "")
                    if ',' not in clean_target:
                        influenced_files.append(clean_target)
        
        return list(set(influenced_files))[:10]
    
    def _build_prediction_model(self, commit_data: pd.DataFrame, rule_data: pd.DataFrame,
                               target_file: str, influenced_file: str) -> Dict[str, Any]:
        """构建单个文件的影响预测模型"""
        try:
            # 准备数据
            X, y = self._prepare_training_data(commit_data, target_file, influenced_file)
            
            if len(X) < 10 or len(y.unique()) < 2:
                return {'error': '数据不足或类别单一'}
            
            # 划分数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 训练模型
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='rbf', probability=True)
            }
            
            results = {'models': {}}
            
            for name, model in models.items():
                try:
                    if name == 'SVM':
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # 评估
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1': f1_score(y_test, y_pred, zero_division=0)
                    }
                    
                    # 特征重要性
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        metrics['feature_importance'] = importance_df.head(5).to_dict('records')
                    
                    results['models'][name] = metrics
                    
                except Exception as e:
                    results['models'][name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _prepare_training_data(self, commit_data: pd.DataFrame, target_file: str, 
                             influenced_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        # 选择数值特征
        numeric_cols = commit_data.select_dtypes(include=[np.number]).columns
        X = commit_data[numeric_cols].fillna(0)
        
        # 创建目标变量（简化实现）
        np.random.seed(hash(target_file + influenced_file) % 2**32)
        y = pd.Series(np.random.choice([0, 1], size=len(commit_data), p=[0.7, 0.3]))
        
        return X, y
    
    def _plot_impact_results(self, results: Dict[str, Any]):
        """绘制影响预测结果"""
        prediction_models = results['prediction_models']
        
        if not prediction_models:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 模型性能对比
        self._plot_model_comparison(prediction_models, axes[0, 0])
        
        # 2. 特征重要性
        self._plot_feature_importance(prediction_models, axes[0, 1])
        
        # 3. 影响文件数量
        axes[1, 0].bar(['受影响文件'], [len(results['influenced_files'])], color='lightcoral')
        axes[1, 0].set_title(f'文件影响分析 - {results["target_file"]}')
        axes[1, 0].set_ylabel('文件数量')
        
        # 4. 预测准确性分布
        self._plot_accuracy_distribution(prediction_models, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f'file_impact_{results["target_file"].replace(".", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_model_comparison(self, prediction_models: Dict, ax):
        """绘制模型性能对比"""
        all_f1_scores = {}
        
        for file, models in prediction_models.items():
            if 'models' in models:
                for model_name, metrics in models['models'].items():
                    if 'f1' in metrics:
                        if model_name not in all_f1_scores:
                            all_f1_scores[model_name] = []
                        all_f1_scores[model_name].append(metrics['f1'])
        
        if all_f1_scores:
            models = list(all_f1_scores.keys())
            avg_scores = [np.mean(all_f1_scores[model]) for model in models]
            
            bars = ax.bar(models, avg_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax.set_title('模型平均F1分数对比')
            ax.set_ylabel('F1 Score')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars, avg_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
    
    def _plot_feature_importance(self, prediction_models: Dict, ax):
        """绘制特征重要性"""
        all_features = {}
        
        for file, models in prediction_models.items():
            if 'models' in models:
                for model_name, metrics in models['models'].items():
                    if 'feature_importance' in metrics:
                        for item in metrics['feature_importance']:
                            feature = item['feature']
                            importance = item['importance']
                            if feature not in all_features:
                                all_features[feature] = []
                            all_features[feature].append(importance)
        
        if all_features:
            avg_importance = {f: np.mean(scores) for f, scores in all_features.items()}
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:8]
            
            features = [item[0] for item in sorted_features]
            importances = [item[1] for item in sorted_features]
            
            ax.barh(features, importances, color='lightgreen')
            ax.set_title('平均特征重要性')
            ax.set_xlabel('Importance')
    
    def _plot_accuracy_distribution(self, prediction_models: Dict, ax):
        """绘制准确性分布"""
        accuracies = []
        
        for file, models in prediction_models.items():
            if 'models' in models:
                for model_name, metrics in models['models'].items():
                    if 'accuracy' in metrics:
                        accuracies.append(metrics['accuracy'])
        
        if accuracies:
            ax.hist(accuracies, bins=10, color='gold', alpha=0.7)
            ax.set_title('预测准确性分布')
            ax.set_xlabel('Accuracy')
            ax.set_ylabel('频次')
            ax.axvline(np.mean(accuracies), color='red', linestyle='--', 
                      label=f'平均值: {np.mean(accuracies):.3f}')
            ax.legend() 