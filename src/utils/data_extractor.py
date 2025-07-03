"""
数据提取器
从Git仓库中提取提交数据
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
from typing import Dict, Any
import logging
from tqdm import tqdm

from pydriller import Repository


class DataExtractor:
    """Git仓库数据提取器"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def extract_repository_data(self, repo_url: str) -> pd.DataFrame:
        """
        从Git仓库中提取数据
        
        Args:
            repo_url: 仓库URL
            
        Returns:
            包含提交数据的DataFrame
        """
        self.logger.info(f"开始从仓库提取数据: {repo_url}")
        
        # 定义CSV文件的列标题
        columns = [
            'hash', 'msg', 'author', 'committer', 'author_date', 'author_timezone',
            'committer_date', 'committer_zone', 'branches', 'in_main_branch', 'merge',
            'modified_files', 'parents', 'project_name', 'project_path', 'deletions',
            'insertions', 'lines', 'files', 'dmm_unit_size', 'dmm_unit_complexity',
            'dmm_unit_interfacing', 'main_file_type', 'main_change_type'
        ]
        
        data_rows = []
        
        try:
            # 遍历仓库中的提交
            skipped_commits = 0
            
            for commit in tqdm(Repository(repo_url).traverse_commits(), desc="提取提交数据"):
                try:
                    # 先检查DMM数据是否可用（按照你原来的方式）
                    if commit.dmm_unit_size is None:
                        skipped_commits += 1
                        continue
                    
                    # 分析文件类型和变更类型（按照你原来的逻辑）
                    main_file_type = None
                    main_change_type = None
                    ADD_NUM = 0
                    DELETE_NUM = 0
                    MODIFY_NUM = 0
                    RENAME_NUM = 0
                    extension_counter = Counter()
                    
                    # 安全地访问 modified_files
                    try:
                        for m in commit.modified_files:
                            _, ext = os.path.splitext(m.filename)
                            extension_counter[ext] += 1

                            if m.change_type.name == 'ADD':
                                ADD_NUM += 1
                            elif m.change_type.name == 'DELETE':
                                DELETE_NUM += 1
                            elif m.change_type.name == 'MODIFY':
                                MODIFY_NUM += 1
                            elif m.change_type.name == 'RENAME':
                                RENAME_NUM += 1
                        
                        # 确定主要文件类型
                        most_common = extension_counter.most_common(1)
                        if most_common:
                            most_common_extension, count = most_common[0]
                            main_file_type = most_common_extension
                        
                        # 确定主要变更类型
                        if ADD_NUM >= DELETE_NUM and ADD_NUM >= MODIFY_NUM and ADD_NUM >= RENAME_NUM:
                            main_change_type = 'ADD'
                        elif DELETE_NUM >= ADD_NUM and DELETE_NUM >= MODIFY_NUM and DELETE_NUM >= RENAME_NUM:
                            main_change_type = 'DELETE'
                        elif MODIFY_NUM >= ADD_NUM and MODIFY_NUM >= DELETE_NUM and MODIFY_NUM >= RENAME_NUM:
                            main_change_type = 'MODIFY'
                        elif RENAME_NUM >= ADD_NUM and RENAME_NUM >= DELETE_NUM and RENAME_NUM >= MODIFY_NUM:
                            main_change_type = 'RENAME'
                        
                        # 获取修改文件列表
                        modified_files_str = str([f.filename for f in commit.modified_files])
                        print(commit.hash)

                    except Exception as e:
                        # 如果访问 modified_files 失败，使用默认值
                        self.logger.debug(f"无法访问 modified_files for commit {commit.hash[:8]}: {str(e)}")
                        main_file_type = ''
                        main_change_type = 'UNKNOWN'
                        modified_files_str = '[]'
                    # 构建数据行（按照你原来的方式）
                    row = [
                        commit.hash,
                        commit.msg if commit.msg else '',
                        commit.author.name if commit.author and commit.author.name else '',
                        commit.committer.name if commit.committer and commit.committer.name else '',
                        commit.author_date,
                        commit.author_timezone,
                        commit.committer_date,
                        commit.committer_timezone,
                        str(commit.branches) if commit.branches else '',
                        commit.in_main_branch,
                        commit.merge,
                        modified_files_str,
                        str(commit.parents) if commit.parents else '',
                        commit.project_name,
                        commit.project_path,
                        commit.deletions,
                        commit.insertions,
                        commit.lines,
                        commit.files,
                        commit.dmm_unit_size,
                        commit.dmm_unit_complexity,
                        commit.dmm_unit_interfacing,
                        main_file_type,
                        main_change_type
                    ]
                    
                    data_rows.append(row)
                    
                except Exception as e:
                    # 记录错误但继续处理其他提交
                    self.logger.warning(f"跳过提交 {commit.hash[:8]}，原因: {str(e)}")
                    skipped_commits += 1
                    continue
            
            # 创建DataFrame
            df = pd.DataFrame(data_rows, columns=columns)
            
            self.logger.info(f"成功提取 {len(df)} 条有效提交记录，跳过 {skipped_commits} 条有问题的提交")
            return df
            
        except Exception as e:
            self.logger.error(f"提取数据时出错: {str(e)}")
            raise
    
    def _analyze_commit_changes(self, commit) -> tuple:
        """
        分析提交的文件变更
        
        Args:
            commit: PyDriller提交对象
            
        Returns:
            (主要文件类型, 主要变更类型)
        """
        # 统计文件扩展名
        extension_counter = Counter()
        
        # 统计变更类型
        add_num = 0
        delete_num = 0
        modify_num = 0
        rename_num = 0
        
        for modified_file in commit.modified_files:
            # 文件扩展名统计
            _, ext = os.path.splitext(modified_file.filename)
            extension_counter[ext] += 1
            
            # 变更类型统计
            change_type = modified_file.change_type.name
            if change_type == 'ADD':
                add_num += 1
            elif change_type == 'DELETE':
                delete_num += 1
            elif change_type == 'MODIFY':
                modify_num += 1
            elif change_type == 'RENAME':
                rename_num += 1
        
        # 确定主要文件类型
        main_file_type = ''
        if extension_counter:
            most_common = extension_counter.most_common(1)
            if most_common:
                main_file_type = most_common[0][0]
        
        # 确定主要变更类型
        change_counts = {
            'ADD': add_num,
            'DELETE': delete_num,
            'MODIFY': modify_num,
            'RENAME': rename_num
        }
        
        main_change_type = max(change_counts, key=change_counts.get)
        
        return main_file_type, main_change_type