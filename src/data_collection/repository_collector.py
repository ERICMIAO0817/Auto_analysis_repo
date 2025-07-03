from pydriller import Repository
import csv
import os
import pandas as pd
from collections import Counter
from datetime import datetime
import logging
from typing import Optional, List, Dict, Any

class RepositoryCollector:
    """GitHub仓库数据收集器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 定义CSV文件的列标题
        self.csv_columns = [
            'hash', 'msg', 'author', 'committer', 'author_date', 'author_timezone',
            'committer_date', 'committer_zone', 'branches', 'in_main_branch', 'merge',
            'modified_files', 'parents', 'project_name', 'project_path', 'deletions',
            'insertions', 'lines', 'files', 'dmm_unit_size', 'dmm_unit_complexity',
            'dmm_unit_interfacing', 'main_file_type', 'main_change_type'
        ]
    
    def check_existing_data(self, output_file: str) -> Optional[str]:
        """
        检查现有数据文件，返回最后一个commit的hash
        
        Args:
            output_file: CSV文件路径
            
        Returns:
            最后一个commit的hash，如果文件不存在或为空则返回None
        """
        if not os.path.exists(output_file):
            self.logger.info(f"文件不存在，将创建新文件: {output_file}")
            return None
        
        try:
            # 读取现有文件的最后几行
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) <= 1:  # 只有标题行或空文件
                self.logger.info(f"文件为空或只有标题行: {output_file}")
                return None
            
            # 获取最后一行的commit hash
            last_line = lines[-1].strip()
            if last_line:
                last_commit_hash = last_line.split(',')[0].strip('"')
                self.logger.info(f"发现现有数据，最后一个commit: {last_commit_hash}")
                return last_commit_hash
            
        except Exception as e:
            self.logger.warning(f"读取现有文件时出错: {str(e)}")
            return None
        
        return None
    
    def get_commit_count_from_file(self, output_file: str) -> int:
        """
        获取现有文件中的commit数量
        
        Args:
            output_file: CSV文件路径
            
        Returns:
            commit数量（不包括标题行）
        """
        if not os.path.exists(output_file):
            return 0
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return max(0, len(lines) - 1)  # 减去标题行
        except Exception as e:
            self.logger.warning(f"计算文件行数时出错: {str(e)}")
            return 0

    def collect_repository_data(self, 
                              repo_url: str, 
                              output_file: str,
                              since: Optional[datetime] = None,
                              to: Optional[datetime] = None,
                              only_in_branch: Optional[str] = None,
                              only_modifications_with_file_types: Optional[List[str]] = None,
                              skip_whitespaces: bool = True,
                              skip_empty_dmm: bool = True,
                              resume: bool = True) -> str:
        """
        收集GitHub仓库的commit数据并保存为CSV
        
        Args:
            repo_url: GitHub仓库URL
            output_file: 输出CSV文件路径
            since: 开始时间
            to: 结束时间
            only_in_branch: 只分析指定分支
            only_modifications_with_file_types: 只分析指定文件类型
            skip_whitespaces: 跳过只有空白字符变更的提交
            skip_empty_dmm: 跳过DMM值为空的提交
            resume: 是否启用断点续传
            
        Returns:
            输出文件路径
        """
        self.logger.info(f"开始收集仓库数据: {repo_url}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 检查是否需要断点续传
        last_commit_hash = None
        existing_count = 0
        file_mode = 'w'
        write_header = True
        
        if resume:
            last_commit_hash = self.check_existing_data(output_file)
            if last_commit_hash:
                existing_count = self.get_commit_count_from_file(output_file)
                file_mode = 'a'  # 追加模式
                write_header = False
                self.logger.info(f"启用断点续传，已有 {existing_count} 个commit，从 {last_commit_hash} 之后继续")
        
        commit_count = existing_count
        skipped_count = 0
        new_commits = 0
        found_resume_point = last_commit_hash is None  # 如果没有断点，直接开始
        
        try:
            with open(output_file, mode=file_mode, newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                
                # 写入标题行（仅在新文件时）
                if write_header:
                    writer.writerow(self.csv_columns)
                
                # 配置Repository参数
                repo_kwargs = {
                    'path_to_repo': repo_url,
                    'since': since,
                    'to': to,
                    'only_in_branch': only_in_branch,
                    'only_modifications_with_file_types': only_modifications_with_file_types,
                    'skip_whitespaces': skip_whitespaces
                }
                
                # 移除None值参数
                repo_kwargs = {k: v for k, v in repo_kwargs.items() if v is not None}
                
                # 遍历仓库中的提交
                for commit in Repository(**repo_kwargs).traverse_commits():
                    # 如果需要断点续传，跳过已处理的commit
                    if not found_resume_point:
                        if commit.hash == last_commit_hash:
                            found_resume_point = True
                            self.logger.info(f"找到断点位置: {commit.hash}")
                        continue
                    
                    self.logger.debug(f"处理提交: {commit.hash}")
                    
                    # 跳过DMM值为空的提交
                    if skip_empty_dmm and commit.dmm_unit_size is None:
                        skipped_count += 1
                        continue
                    
                    # 分析文件类型和变更类型
                    main_file_type, main_change_type = self._analyze_file_changes(commit)
                    
                    # 构建CSV行数据
                    row = self._build_csv_row(commit, main_file_type, main_change_type)
                    
                    # 写入CSV文件
                    writer.writerow(row)
                    commit_count += 1
                    new_commits += 1
                    
                    if commit_count % 100 == 0:
                        self.logger.info(f"已处理 {commit_count} 个提交（新增 {new_commits} 个）")
        
        except Exception as e:
            self.logger.error(f"收集数据时发生错误: {str(e)}")
            raise
        
        if resume and existing_count > 0:
            self.logger.info(f"断点续传完成! 原有 {existing_count} 个提交，新增 {new_commits} 个提交，总共 {commit_count} 个提交，跳过 {skipped_count} 个提交")
        else:
            self.logger.info(f"数据收集完成! 总共处理 {commit_count} 个提交，跳过 {skipped_count} 个提交")
        
        self.logger.info(f"数据已保存到: {output_file}")
        
        return output_file
    
    def _analyze_file_changes(self, commit) -> tuple:
        """分析提交中的文件变更类型"""
        main_file_type = None
        main_change_type = None
        
        # 统计各种变更类型的数量
        change_counts = {
            'ADD': 0,
            'DELETE': 0,
            'MODIFY': 0,
            'RENAME': 0
        }
        
        # 统计文件扩展名
        extension_counter = Counter()
        
        for modified_file in commit.modified_files:
            # 统计文件扩展名
            _, ext = os.path.splitext(modified_file.filename)
            if ext:  # 只统计有扩展名的文件
                extension_counter[ext] += 1
            
            # 统计变更类型
            change_type = modified_file.change_type.name
            if change_type in change_counts:
                change_counts[change_type] += 1
        
        # 确定主要文件类型
        if extension_counter:
            most_common_ext = extension_counter.most_common(1)[0]
            main_file_type = most_common_ext[0]
        
        # 确定主要变更类型
        if any(change_counts.values()):
            main_change_type = max(change_counts, key=change_counts.get)
        
        return main_file_type, main_change_type
    
    def _build_csv_row(self, commit, main_file_type: str, main_change_type: str) -> List[str]:
        """构建CSV行数据"""
        return [
            commit.hash,
            commit.msg if commit.msg else '',
            commit.author.name if commit.author and commit.author.name else '',
            commit.committer.name if commit.committer and commit.committer.name else '',
            str(commit.author_date) if commit.author_date else '',
            str(commit.author_timezone) if commit.author_timezone else '',
            str(commit.committer_date) if commit.committer_date else '',
            str(commit.committer_timezone) if commit.committer_timezone else '',
            str(commit.branches) if commit.branches else '',
            str(commit.in_main_branch) if hasattr(commit, 'in_main_branch') else '',
            str(commit.merge) if hasattr(commit, 'merge') and commit.merge else '',
            str(len(commit.modified_files)) if commit.modified_files else '0',
            str(len(commit.parents)) if commit.parents else '0',
            str(commit.project_name) if hasattr(commit, 'project_name') and commit.project_name else '',
            str(commit.project_path) if hasattr(commit, 'project_path') and commit.project_path else '',
            str(commit.deletions) if hasattr(commit, 'deletions') and commit.deletions is not None else '0',
            str(commit.insertions) if hasattr(commit, 'insertions') and commit.insertions is not None else '0',
            str(commit.lines) if hasattr(commit, 'lines') and commit.lines is not None else '0',
            str(commit.files) if hasattr(commit, 'files') and commit.files is not None else '0',
            str(commit.dmm_unit_size) if hasattr(commit, 'dmm_unit_size') and commit.dmm_unit_size is not None else '',
            str(commit.dmm_unit_complexity) if hasattr(commit, 'dmm_unit_complexity') and commit.dmm_unit_complexity is not None else '',
            str(commit.dmm_unit_interfacing) if hasattr(commit, 'dmm_unit_interfacing') and commit.dmm_unit_interfacing is not None else '',
            str(main_file_type) if main_file_type else '',
            str(main_change_type) if main_change_type else ''
        ]
    
    def collect_multiple_repositories(self, 
                                    repo_configs: List[Dict[str, Any]], 
                                    output_dir: str = "data/raw") -> List[str]:
        """
        批量收集多个仓库的数据
        
        Args:
            repo_configs: 仓库配置列表，每个配置包含repo_url和其他参数
            output_dir: 输出目录
            
        Returns:
            生成的CSV文件路径列表
        """
        output_files = []
        
        for i, config in enumerate(repo_configs):
            repo_url = config['repo_url']
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            output_file = os.path.join(output_dir, f"commits_{repo_name}.csv")
            
            self.logger.info(f"处理仓库 {i+1}/{len(repo_configs)}: {repo_name}")
            
            try:
                # 移除repo_url，传递其他参数
                collect_config = {k: v for k, v in config.items() if k != 'repo_url'}
                result_file = self.collect_repository_data(repo_url, output_file, **collect_config)
                output_files.append(result_file)
            except Exception as e:
                self.logger.error(f"处理仓库 {repo_name} 时发生错误: {str(e)}")
                continue
        
        return output_files

def main():
    """命令行接口示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='收集GitHub仓库commit数据')
    parser.add_argument('repo_url', help='GitHub仓库URL')
    parser.add_argument('-o', '--output', default='commits.csv', help='输出CSV文件路径')
    parser.add_argument('--since', help='开始时间 (YYYY-MM-DD)')
    parser.add_argument('--to', help='结束时间 (YYYY-MM-DD)')
    parser.add_argument('--branch', help='指定分支')
    parser.add_argument('--file-types', nargs='+', help='指定文件类型 (如 .py .java)')
    parser.add_argument('--skip-empty-dmm', action='store_true', help='跳过DMM值为空的提交')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析时间参数
    since = datetime.strptime(args.since, '%Y-%m-%d') if args.since else None
    to = datetime.strptime(args.to, '%Y-%m-%d') if args.to else None
    
    # 创建收集器并执行
    collector = RepositoryCollector()
    collector.collect_repository_data(
        repo_url=args.repo_url,
        output_file=args.output,
        since=since,
        to=to,
        only_in_branch=args.branch,
        only_modifications_with_file_types=args.file_types,
        skip_empty_dmm=args.skip_empty_dmm
    )

if __name__ == "__main__":
    main() 