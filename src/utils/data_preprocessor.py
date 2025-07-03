"""
数据预处理器
清洗和转换原始数据，为分析做准备
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gensim.models import Word2Vec
from datetime import datetime
import logging
from typing import Dict, Any, Tuple


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            data: 原始数据DataFrame
            
        Returns:
            预处理后的DataFrame
        """
        self.logger.info("开始数据预处理...")
        
        # 复制数据避免修改原始数据
        processed_data = data.copy()
        
        # 1. 处理时间特征
        processed_data = self._process_time_features(processed_data)
        
        # 2. 处理数值特征
        processed_data = self._process_numeric_features(processed_data)
        
        # 3. 处理文本特征
        processed_data = self._process_text_features(processed_data)
        
        # 4. 处理分类特征
        processed_data = self._process_categorical_features(processed_data)
        
        # 5. 清理和删除不需要的列
        processed_data = self._cleanup_data(processed_data)
        
        # 6. 标准化数值特征
        processed_data = self._standardize_features(processed_data)
        
        self.logger.info(f"数据预处理完成，最终特征数: {processed_data.shape[1]}")
        return processed_data
    
    def _process_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理时间特征"""
        self.logger.info("处理时间特征...")
        
        # 检查是否有时间字段
        time_columns = ['author_date', 'committer_date']
        available_time_columns = [col for col in time_columns if col in data.columns]
        
        if not available_time_columns:
            self.logger.info("未找到时间字段，跳过时间特征处理")
            return data
        
        try:
            for time_col in available_time_columns:
                if time_col in data.columns:
                    # 转换为datetime，指定utc=True以避免警告
                    data[time_col] = pd.to_datetime(data[time_col], errors='coerce', utc=True)
                    
                    # 提取时间特征
                    data[f'{time_col}_year'] = data[time_col].dt.year
                    data[f'{time_col}_month'] = data[time_col].dt.month
                    data[f'{time_col}_day'] = data[time_col].dt.day
                    data[f'{time_col}_hour'] = data[time_col].dt.hour
                    data[f'{time_col}_weekday'] = data[time_col].dt.weekday
                    
                    # 填充可能的NaN值
                    for feature in [f'{time_col}_year', f'{time_col}_month', f'{time_col}_day', 
                                  f'{time_col}_hour', f'{time_col}_weekday']:
                        if feature in data.columns:
                            data[feature] = data[feature].fillna(0).astype(int)
                    
                    # 删除原始时间列
                    data = data.drop(columns=[time_col])
            
            return data
            
        except Exception as e:
            self.logger.warning(f"时间特征处理失败: {str(e)}")
            return data
    
    def _process_numeric_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理数值特征"""
        self.logger.info("处理数值特征...")
        
        # 数值型特征列表
        numeric_cols = [
            'deletions', 'insertions', 'lines', 'files',
            'dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing'
        ]
        
        # 填充缺失值和处理无穷大值
        for col in numeric_cols:
            if col in data.columns:
                # 替换无穷大值为NaN，然后填充为0
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                data[col] = data[col].fillna(0)
                
                # 确保数据类型为数值型
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
                
                # 处理异常值（使用IQR方法）
                if data[col].std() > 0:  # 只有当标准差大于0时才进行异常值处理
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:  # 避免IQR为0的情况
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 计算衍生特征
        if 'insertions' in data.columns and 'deletions' in data.columns:
            data['net_lines'] = data['insertions'] - data['deletions']
            # 避免除零错误，并处理可能的无穷大值
            data['churn_ratio'] = data['deletions'] / (data['insertions'] + 1)
            data['churn_ratio'] = data['churn_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # 提取parents特征
        if 'parents' in data.columns:
            data['parent_count'] = data['parents'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) and str(x) != '' else 0
            )
        
        return data
    
    def _process_text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理文本特征"""
        self.logger.info("处理文本特征...")
        
        if 'msg' not in data.columns:
            return data
        
        # 基本文本特征
        data['msg_length'] = data['msg'].apply(lambda x: len(str(x)))
        data['msg_word_count'] = data['msg'].apply(lambda x: len(str(x).split()))
        
        # Word2Vec特征
        try:
            # 预处理消息文本
            def preprocess_msg(msg):
                return str(msg).lower().split()
            
            data['msg_tokens'] = data['msg'].apply(preprocess_msg)
            
            # 训练Word2Vec模型
            w2v_model = Word2Vec(
                sentences=data['msg_tokens'],
                vector_size=50,  # 减少维度以提高效率
                window=5,
                min_count=1,
                workers=4
            )
            
            # 获取Word2Vec向量
            def get_w2v_vector(tokens):
                vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
                if vectors:
                    return np.mean(vectors, axis=0)
                else:
                    return np.zeros(50)
            
            data['msg_w2v'] = data['msg_tokens'].apply(get_w2v_vector)
            
            # 展开Word2Vec特征
            w2v_features = pd.DataFrame(
                data['msg_w2v'].tolist(),
                index=data.index,
                columns=[f'w2v_{i}' for i in range(50)]
            )
            data = pd.concat([data, w2v_features], axis=1)
            
            # 删除临时列
            data = data.drop(columns=['msg_w2v', 'msg_tokens'])
            
        except Exception as e:
            self.logger.warning(f"Word2Vec处理失败: {str(e)}")
        
        return data
    
    def _process_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理分类特征"""
        self.logger.info("处理分类特征...")
        
        categorical_cols = ['author', 'committer', 'branches', 'main_file_type', 'main_change_type']
        
        for col in categorical_cols:
            if col in data.columns:
                # 填充缺失值
                data[col] = data[col].fillna('unknown')
                
                # 标签编码
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
        
        # 处理布尔特征
        bool_cols = ['in_main_branch', 'merge']
        for col in bool_cols:
            if col in data.columns:
                # 先填充缺失值，然后转换为整数
                data[col] = data[col].fillna(False)
                data[col] = data[col].astype(bool).astype(int)
        
        return data
    
    def _cleanup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清理数据，删除不需要的列"""
        self.logger.info("清理数据...")
        
        # 要删除的列（保留modified_files用于关联规则挖掘）
        drop_cols = [
            'hash', 'msg', 'author_date', 'author_timezone',
            'committer_date', 'committer_zone', 'project_path',
            'parents', 'project_name'
        ]
        
        # 只删除存在的列
        existing_drop_cols = [col for col in drop_cols if col in data.columns]
        data = data.drop(columns=existing_drop_cols, errors='ignore')
        
        return data
    
    def _standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化数值特征"""
        self.logger.info("标准化特征...")
        
        # 需要标准化的数值特征
        numeric_features = [
            'deletions', 'insertions', 'lines', 'files',
            'dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing',
            'time_diff', 'msg_length', 'msg_word_count',
            'author_date_year', 'author_date_month', 'author_date_day', 'author_date_hour',
            'committer_date_year', 'committer_date_month', 'committer_date_day', 'committer_date_hour',
            'net_lines', 'churn_ratio', 'parent_count'
        ]
        
        # 只标准化存在的特征
        existing_numeric_features = [col for col in numeric_features if col in data.columns]
        
        if existing_numeric_features:
            # 确保所有特征都没有NaN或无穷大值
            for col in existing_numeric_features:
                data[col] = data[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # 进行标准化
            try:
                data[existing_numeric_features] = self.scaler.fit_transform(data[existing_numeric_features])
            except Exception as e:
                self.logger.warning(f"标准化失败: {str(e)}")
                # 如果标准化失败，至少确保数据是数值型的
                for col in existing_numeric_features:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
        
        # 标准化Word2Vec特征
        w2v_cols = [col for col in data.columns if col.startswith('w2v_')]
        if w2v_cols:
            try:
                # 确保Word2Vec特征没有NaN值
                for col in w2v_cols:
                    data[col] = data[col].fillna(0)
                data[w2v_cols] = StandardScaler().fit_transform(data[w2v_cols])
            except Exception as e:
                self.logger.warning(f"Word2Vec特征标准化失败: {str(e)}")
        
        return data 