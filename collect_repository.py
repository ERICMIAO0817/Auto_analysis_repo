#!/usr/bin/env python3
"""
GitHub仓库数据收集工具

使用示例:
python collect_repository.py https://github.com/BeyondDimension/SteamTools -o data/raw/commits_steamtools.csv
python collect_repository.py https://github.com/apache/flink --since 2023-01-01 --to 2023-12-31
"""

import sys
import os
import argparse
import logging
from datetime import datetime

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
            logging.FileHandler('repository_collection.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(
        description='收集GitHub仓库commit数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s https://github.com/BeyondDimension/SteamTools
  %(prog)s https://github.com/apache/flink -o data/raw/commits_flink.csv
  %(prog)s https://github.com/apache/flink --since 2023-01-01 --to 2023-12-31
  %(prog)s https://github.com/apache/flink --branch main --file-types .java .xml
        """
    )
    
    parser.add_argument('repo_url', help='GitHub仓库URL')
    parser.add_argument('-o', '--output', help='输出CSV文件路径 (默认: data/raw/commits_<repo_name>.csv)')
    parser.add_argument('--since', help='开始时间 (格式: YYYY-MM-DD)')
    parser.add_argument('--to', help='结束时间 (格式: YYYY-MM-DD)')
    parser.add_argument('--branch', help='指定分析的分支')
    parser.add_argument('--file-types', nargs='+', help='指定文件类型 (如: .py .java .cpp)')
    parser.add_argument('--skip-empty-dmm', action='store_true', 
                       help='跳过DMM值为空的提交 (默认: True)', default=True)
    parser.add_argument('--include-empty-dmm', action='store_true', 
                       help='包含DMM值为空的提交')
    parser.add_argument('--no-skip-whitespaces', action='store_true', 
                       help='不跳过只有空白字符变更的提交')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='启用断点续传 (默认: True)')
    parser.add_argument('--no-resume', action='store_true',
                       help='禁用断点续传，重新开始收集')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # 解析时间参数
        since = None
        to = None
        if args.since:
            try:
                since = datetime.strptime(args.since, '%Y-%m-%d')
            except ValueError:
                logger.error(f"无效的开始时间格式: {args.since}，请使用 YYYY-MM-DD 格式")
                return 1
        
        if args.to:
            try:
                to = datetime.strptime(args.to, '%Y-%m-%d')
            except ValueError:
                logger.error(f"无效的结束时间格式: {args.to}，请使用 YYYY-MM-DD 格式")
                return 1
        
        # 确定输出文件路径
        if args.output:
            output_file = args.output
        else:
            repo_name = args.repo_url.split('/')[-1].replace('.git', '')
            output_file = f"data/raw/commits_{repo_name}.csv"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 创建收集器
        collector = RepositoryCollector()
        
        # 收集数据
        logger.info(f"开始收集仓库数据: {args.repo_url}")
        logger.info(f"输出文件: {output_file}")
        
        result_file = collector.collect_repository_data(
            repo_url=args.repo_url,
            output_file=output_file,
            since=since,
            to=to,
            only_in_branch=args.branch,
            only_modifications_with_file_types=args.file_types,
            skip_whitespaces=not args.no_skip_whitespaces,
            skip_empty_dmm=args.skip_empty_dmm and not args.include_empty_dmm,
            resume=args.resume and not args.no_resume
        )
        
        logger.info(f"数据收集完成! 文件已保存到: {result_file}")
        
        # 显示文件信息
        if os.path.exists(result_file):
            import pandas as pd
            try:
                df = pd.read_csv(result_file)
                logger.info(f"CSV文件信息:")
                logger.info(f"  - 总行数: {len(df)}")
                logger.info(f"  - 总列数: {len(df.columns)}")
                logger.info(f"  - 文件大小: {os.path.getsize(result_file) / 1024 / 1024:.2f} MB")
                
                # 显示DMM统计信息
                dmm_cols = ['dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing']
                for col in dmm_cols:
                    if col in df.columns:
                        non_empty = df[col].notna().sum()
                        logger.info(f"  - {col}: {non_empty}/{len(df)} 个非空值")
                
            except Exception as e:
                logger.warning(f"无法读取生成的CSV文件进行统计: {e}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"收集数据时发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 