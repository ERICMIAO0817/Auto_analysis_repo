#!/usr/bin/env python3
"""
正确的机器学习结果调试
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.analyzer import GitAnalyzer
import pandas as pd

def debug_ml_results():
    """调试机器学习结果"""
    print("=== 开始调试机器学习结果 ===")
    
    # 加载数据
    df = pd.read_csv('data/raw/commits_knlp.csv')
    print(f"数据加载成功: {len(df)} 行, {len(df.columns)} 列")
    
    # 创建分析器
    analyzer = GitAnalyzer()
    
    # 运行分析
    results = analyzer.comprehensive_analysis(df, project_name="knlp_debug")
    
    print("\n=== 分析结果结构 ===")
    print("主要键:", list(results.keys()))
    
    if 'ml_analysis' in results:
        ml_data = results['ml_analysis']
        print(f"\nML分析类型: {type(ml_data)}")
        print("ML分析键:", list(ml_data.keys()))
        
        for task_name, task_results in ml_data.items():
            print(f"\n任务: {task_name}")
            print(f"  类型: {type(task_results)}")
            
            if isinstance(task_results, dict):
                print(f"  键: {list(task_results.keys())}")
                
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
                            if isinstance(value, dict):
                                print(f"      键: {list(value.keys())}")
                                if 'r2_score' in value or 'accuracy' in value:
                                    accuracy = value.get('accuracy', 'N/A')
                                    r2 = value.get('r2_score', 'N/A')
                                    print(f"      accuracy: {accuracy}, r2: {r2}")
    else:
        print("没有找到ml_analysis键")
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    debug_ml_results() 