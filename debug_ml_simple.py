#!/usr/bin/env python3
"""
简单调试机器学习结果
"""
import sys
sys.path.append('src')
from analysis.prediction_models import PredictionModels
from core.data_processor import DataProcessor
import pandas as pd

# 加载数据
df = pd.read_csv('data/raw/commits_knlp.csv')
processor = DataProcessor()
processed_data = processor.preprocess_data(df)

# 运行ML分析
ml_analyzer = PredictionModels()
ml_results = ml_analyzer.run_all_predictions(processed_data)

print("=== 机器学习结果调试 ===")
for task_name, task_results in ml_results.items():
    print(f"\n任务: {task_name}")
    print(f"类型: {type(task_results)}")
    
    if isinstance(task_results, dict):
        print("键:", list(task_results.keys()))
        
        # 检查是否有models键
        if 'models' in task_results:
            print("  有models键，内容:")
            models = task_results['models']
            for model_name, model_data in models.items():
                print(f"    {model_name}: {type(model_data)}")
                if isinstance(model_data, dict):
                    print(f"      键: {list(model_data.keys())}")
                    # 显示关键指标
                    accuracy = model_data.get('accuracy', 'N/A')
                    r2 = model_data.get('r2_score', 'N/A')
                    precision = model_data.get('precision', 'N/A')
                    recall = model_data.get('recall', 'N/A')
                    f1 = model_data.get('f1', 'N/A')
                    print(f"      accuracy: {accuracy}, r2: {r2}, precision: {precision}, recall: {recall}, f1: {f1}")
        else:
            print("  没有models键，直接检查模型结果:")
            for key, value in task_results.items():
                if isinstance(value, dict) and key not in ['class_distribution', 'feature_importance', 'confusion_matrix', 'analysis']:
                    print(f"    {key}: {type(value)}")
                    if 'r2_score' in value or 'accuracy' in value:
                        accuracy = value.get('accuracy', 'N/A')
                        r2 = value.get('r2_score', 'N/A')
                        print(f"      accuracy: {accuracy}, r2: {r2}")

print("\n=== 调试完成 ===") 