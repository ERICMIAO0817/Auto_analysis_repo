"""
日志配置工具
"""

import logging
import sys
from pathlib import Path


def setup_logger(verbose=False, log_file=None):
    """
    设置日志记录器
    
    Args:
        verbose: 是否详细输出
        log_file: 日志文件路径
        
    Returns:
        配置好的logger
    """
    # 创建logger
    logger = logging.getLogger('GitAnalytics')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 