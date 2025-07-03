# GitAnalytics 使用指南

## 🚀 快速开始

### 1. 最简单的方式
```bash
python quick_start.py
```
然后输入GitHub仓库URL即可！

### 2. 命令行使用

#### 分析GitHub仓库
```bash
# 直接分析GitHub仓库 (支持断点续传)
python analyze_csv_improved.py --repo-url https://github.com/user/repo --project-name myproject

# 指定时间范围
python analyze_csv_improved.py --repo-url https://github.com/user/repo --project-name myproject --since 2023-01-01 --to 2023-12-31

# 指定输出目录
python analyze_csv_improved.py --repo-url https://github.com/user/repo --project-name myproject --output results/

# 强制重新开始收集 (忽略现有数据)
python analyze_csv_improved.py --repo-url https://github.com/user/repo --project-name myproject --no-resume
```

#### 只收集数据
```bash
# 收集单个仓库数据 (支持断点续传)
python collect_repository.py https://github.com/user/repo -o data/repo_data.csv

# 强制重新开始收集
python collect_repository.py https://github.com/user/repo -o data/repo_data.csv --no-resume

# 批量收集多个仓库 (支持断点续传)
python batch_collect_repositories.py --create-sample-config
python batch_collect_repositories.py --config repo_config_sample.json
```

#### 分析现有CSV文件
```bash
python analyze_csv_improved.py data/repo_data.csv --project-name myproject
```

### 3. 编程接口

```python
from src.data_collection.repository_collector import RepositoryCollector
from src.core.analyzer import GitAnalyzer

# 收集数据
collector = RepositoryCollector()
csv_file = collector.collect_repository_data(
    repo_url="https://github.com/user/repo",
    output_file="data/commits.csv"
)

# 分析数据
analyzer = GitAnalyzer()
results = analyzer.analyze_csv_data(csv_file, "project_name")
```

## 📊 分析功能

- **DMM风险预测**: 预测代码质量风险
- **文件影响分析**: 分析文件修改的影响范围
- **聚类分析**: 基于提交特征的智能聚类
- **关联规则挖掘**: 发现文件间的协同变更模式
- **可视化报告**: 自动生成图表和报告

## 📁 输出文件

分析完成后会生成：
- CSV数据文件
- 可视化图表 (PNG格式)
- 分析报告
- 模型性能对比图

## 🔧 配置选项

### 数据收集选项
- `--since` / `--to`: 时间范围过滤
- `--branch`: 指定分支
- `--file-types`: 文件类型过滤
- `--skip-empty-dmm`: 跳过DMM值为空的提交
- `--resume` / `--no-resume`: 断点续传控制

### 分析选项
- `--output`: 输出目录
- `--verbose`: 详细输出
- `--project-name`: 项目名称

## 📖 详细文档

- [数据收集功能详细说明](DATA_COLLECTION_README.md)
- [功能改进总结](IMPROVEMENTS_SUMMARY.md)
- [示例代码](examples/collect_and_analyze_example.py)

## ❓ 常见问题

**Q: 如何分析私有仓库？**
A: 确保您有访问权限，PyDriller会使用您的Git凭据。

**Q: 分析大型仓库很慢怎么办？**
A: 使用时间范围过滤 `--since` 和 `--to` 参数。

**Q: 如何批量分析多个仓库？**
A: 使用 `batch_collect_repositories.py` 工具。

**Q: 生成的图表在哪里？**
A: 默认保存在当前目录或指定的输出目录中。

**Q: 如何使用断点续传功能？**
A: 默认启用断点续传，程序会自动检测`data/raw`目录中的现有数据并继续收集。使用`--no-resume`可以强制重新开始。

**Q: 网络中断后如何继续收集？**
A: 直接重新运行相同的命令，程序会自动从中断点继续收集数据。 