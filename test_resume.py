#!/usr/bin/env python3
"""
测试断点续传功能

这个脚本用于测试数据收集器的断点续传功能
"""

import sys
import os
import logging

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.repository_collector import RepositoryCollector

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_resume_functionality():
    """测试断点续传功能"""
    logger = logging.getLogger(__name__)
    
    # 测试参数
    repo_url = "https://github.com/BeyondDimension/SteamTools"
    output_file = "data/raw/commits_SteamTools.csv"
    
    logger.info("开始测试断点续传功能...")
    
    # 创建收集器
    collector = RepositoryCollector()
    
    # 检查现有数据
    last_commit = collector.check_existing_data(output_file)
    if last_commit:
        logger.info(f"发现现有数据，最后一个commit: {last_commit}")
        
        existing_count = collector.get_commit_count_from_file(output_file)
        logger.info(f"现有commit数量: {existing_count}")
        
        # 测试断点续传
        logger.info("开始断点续传测试...")
        try:
            result_file = collector.collect_repository_data(
                repo_url=repo_url,
                output_file=output_file,
                resume=True
            )
            
            # 检查结果
            new_count = collector.get_commit_count_from_file(result_file)
            logger.info(f"断点续传后commit数量: {new_count}")
            logger.info(f"新增commit数量: {new_count - existing_count}")
            
        except Exception as e:
            logger.error(f"断点续传测试失败: {str(e)}")
            return False
    else:
        logger.info("没有发现现有数据，将进行全新收集")
        
        try:
            result_file = collector.collect_repository_data(
                repo_url=repo_url,
                output_file=output_file,
                resume=True
            )
            
            final_count = collector.get_commit_count_from_file(result_file)
            logger.info(f"全新收集完成，总commit数量: {final_count}")
            
        except Exception as e:
            logger.error(f"全新收集失败: {str(e)}")
            return False
    
    logger.info("断点续传功能测试完成!")
    return True

def main():
    """主函数"""
    setup_logging()
    
    try:
        success = test_resume_functionality()
        if success:
            print("✅ 断点续传功能测试通过!")
            return 0
        else:
            print("❌ 断点续传功能测试失败!")
            return 1
            
    except KeyboardInterrupt:
        print("用户中断测试")
        return 1
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 