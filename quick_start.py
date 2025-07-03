#!/usr/bin/env python3
"""
GitAnalytics 快速开始脚本

最简单的使用方式 - 一键从GitHub仓库收集数据并分析
"""

import sys
import os

def main():
    """快速开始主函数"""
    
    print("🚀 GitAnalytics 快速开始")
    print("="*50)
    
    # 获取用户输入
    repo_url = input("请输入GitHub仓库URL (例如: https://github.com/user/repo): ").strip()
    
    if not repo_url:
        print("❌ 请提供有效的GitHub仓库URL")
        return 1
    
    # 提取项目名称
    project_name = repo_url.split('/')[-1].replace('.git', '')
    
    print(f"📊 开始分析项目: {project_name}")
    print(f"🔗 仓库地址: {repo_url}")
    
    # 构建命令
    cmd = f'python analyze_csv_improved.py --repo-url "{repo_url}" --project-name "{project_name}" -v'
    
    print(f"\n🔧 执行命令:")
    print(f"   {cmd}")
    print("\n⏳ 正在收集数据和分析，请稍候...")
    
    # 执行分析
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("\n✅ 分析完成!")
        print("📁 结果文件已保存到当前目录")
        print("📊 可视化图表已生成")
    else:
        print("\n❌ 分析失败，请检查错误信息")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1) 