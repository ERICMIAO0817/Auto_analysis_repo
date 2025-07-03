#!/usr/bin/env python3
"""
批量收集多个GitHub仓库数据的工具

使用示例:
python batch_collect_repositories.py --config repo_config.json
python batch_collect_repositories.py --repos https://github.com/apache/flink https://github.com/apache/spark
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.repository_collector import RepositoryCollector

def setup_logging(verbose: bool = False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('batch_repository_collection.log')
        ]
    )

def load_config_file(config_file: str) -> List[Dict[str, Any]]:
    """从JSON配置文件加载仓库配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if isinstance(config, dict) and 'repositories' in config:
            return config['repositories']
        elif isinstance(config, list):
            return config
        else:
            raise ValueError("配置文件格式错误，应包含 'repositories' 数组或直接为数组")
    
    except Exception as e:
        raise ValueError(f"无法加载配置文件 {config_file}: {e}")

def create_sample_config():
    """创建示例配置文件"""
    sample_config = {
        "repositories": [
            {
                "repo_url": "https://github.com/BeyondDimension/SteamTools",
                "since": "2023-01-01",
                "to": "2023-12-31",
                "only_in_branch": "main",
                "skip_empty_dmm": True
            },
            {
                "repo_url": "https://github.com/apache/flink",
                "only_modifications_with_file_types": [".java", ".xml"],
                "skip_empty_dmm": True
            },
            {
                "repo_url": "https://github.com/apache/spark",
                "since": "2023-06-01",
                "only_modifications_with_file_types": [".scala", ".java"],
                "skip_empty_dmm": False
            }
        ]
    }
    
    config_file = "repo_config_sample.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    return config_file

def parse_repo_configs_from_urls(repo_urls: List[str], 
                                since: str = None, 
                                to: str = None,
                                branch: str = None,
                                file_types: List[str] = None,
                                skip_empty_dmm: bool = True) -> List[Dict[str, Any]]:
    """从URL列表创建仓库配置"""
    configs = []
    
    for repo_url in repo_urls:
        config = {"repo_url": repo_url}
        
        if since:
            config["since"] = since
        if to:
            config["to"] = to
        if branch:
            config["only_in_branch"] = branch
        if file_types:
            config["only_modifications_with_file_types"] = file_types
        
        config["skip_empty_dmm"] = skip_empty_dmm
        configs.append(config)
    
    return configs

def process_config_dates(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """处理配置中的日期字符串，转换为datetime对象"""
    processed_configs = []
    
    for config in configs:
        processed_config = config.copy()
        
        # 处理since日期
        if 'since' in processed_config and isinstance(processed_config['since'], str):
            try:
                processed_config['since'] = datetime.strptime(processed_config['since'], '%Y-%m-%d')
            except ValueError as e:
                logging.warning(f"无效的since日期格式 '{processed_config['since']}': {e}")
                del processed_config['since']
        
        # 处理to日期
        if 'to' in processed_config and isinstance(processed_config['to'], str):
            try:
                processed_config['to'] = datetime.strptime(processed_config['to'], '%Y-%m-%d')
            except ValueError as e:
                logging.warning(f"无效的to日期格式 '{processed_config['to']}': {e}")
                del processed_config['to']
        
        processed_configs.append(processed_config)
    
    return processed_configs

def main():
    parser = argparse.ArgumentParser(
        description='批量收集多个GitHub仓库commit数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --config repo_config.json
  %(prog)s --repos https://github.com/apache/flink https://github.com/apache/spark
  %(prog)s --create-sample-config
        """
    )
    
    # 配置文件选项
    parser.add_argument('--config', help='JSON配置文件路径')
    
    # 直接指定仓库选项
    parser.add_argument('--repos', nargs='+', help='仓库URL列表')
    parser.add_argument('--since', help='开始时间 (格式: YYYY-MM-DD)')
    parser.add_argument('--to', help='结束时间 (格式: YYYY-MM-DD)')
    parser.add_argument('--branch', help='指定分析的分支')
    parser.add_argument('--file-types', nargs='+', help='指定文件类型 (如: .py .java .cpp)')
    parser.add_argument('--skip-empty-dmm', action='store_true', default=True,
                       help='跳过DMM值为空的提交 (默认: True)')
    parser.add_argument('--include-empty-dmm', action='store_true',
                       help='包含DMM值为空的提交')
    
    # 输出选项
    parser.add_argument('-o', '--output-dir', default='data/raw',
                       help='输出目录 (默认: data/raw)')
    
    # 断点续传选项
    parser.add_argument('--resume', action='store_true', default=True,
                       help='启用断点续传 (默认: True)')
    parser.add_argument('--no-resume', action='store_true',
                       help='禁用断点续传，重新开始收集')
    
    # 其他选项
    parser.add_argument('--create-sample-config', action='store_true',
                       help='创建示例配置文件')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # 创建示例配置文件
        if args.create_sample_config:
            config_file = create_sample_config()
            logger.info(f"示例配置文件已创建: {config_file}")
            return 0
        
        # 获取仓库配置
        repo_configs = []
        
        if args.config:
            # 从配置文件加载
            repo_configs = load_config_file(args.config)
            logger.info(f"从配置文件加载了 {len(repo_configs)} 个仓库配置")
        
        elif args.repos:
            # 从命令行参数创建配置
            skip_empty_dmm = args.skip_empty_dmm and not args.include_empty_dmm
            repo_configs = parse_repo_configs_from_urls(
                args.repos, args.since, args.to, args.branch, 
                args.file_types, skip_empty_dmm
            )
            logger.info(f"从命令行参数创建了 {len(repo_configs)} 个仓库配置")
        
        else:
            logger.error("请指定 --config 或 --repos 参数")
            return 1
        
        if not repo_configs:
            logger.error("没有找到要处理的仓库配置")
            return 1
        
        # 处理配置中的日期
        repo_configs = process_config_dates(repo_configs)
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 创建收集器并批量处理
        collector = RepositoryCollector()
        
        logger.info(f"开始批量收集 {len(repo_configs)} 个仓库的数据")
        logger.info(f"输出目录: {args.output_dir}")
        
        # 为每个配置添加resume参数
        resume_enabled = args.resume and not args.no_resume
        for config in repo_configs:
            if 'resume' not in config:
                config['resume'] = resume_enabled
        
        result_files = collector.collect_multiple_repositories(
            repo_configs, args.output_dir
        )
        
        # 显示结果统计
        logger.info(f"批量收集完成!")
        logger.info(f"成功处理 {len(result_files)} 个仓库")
        
        total_size = 0
        total_commits = 0
        
        for result_file in result_files:
            if os.path.exists(result_file):
                file_size = os.path.getsize(result_file) / 1024 / 1024  # MB
                total_size += file_size
                
                try:
                    import pandas as pd
                    df = pd.read_csv(result_file)
                    commit_count = len(df)
                    total_commits += commit_count
                    
                    repo_name = os.path.basename(result_file).replace('commits_', '').replace('.csv', '')
                    logger.info(f"  - {repo_name}: {commit_count} commits, {file_size:.2f} MB")
                    
                except Exception as e:
                    logger.warning(f"无法读取文件 {result_file}: {e}")
        
        logger.info(f"总计: {total_commits} commits, {total_size:.2f} MB")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"批量收集数据时发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 