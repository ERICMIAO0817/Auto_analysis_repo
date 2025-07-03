# GitAnalytics

一个基于PyDriller的Git仓库智能分析工具，提供全面的代码仓库分析和可视化功能。支持直接从GitHub仓库收集数据，无需手动下载。

## 功能特性

### 🚀 数据收集
- **GitHub仓库直接收集**: 支持从任何GitHub仓库URL直接收集commit数据
- **🔄 智能断点续传**: 自动检测已有数据，从中断点继续收集，节省时间
- **智能过滤**: 支持时间范围、分支、文件类型等多维度过滤
- **批量处理**: 支持同时收集多个仓库的数据
- **DMM质量指标**: 自动提取代码可维护性指标

### 🔍 核心分析功能
- **提交分析**: 提取提交历史、作者信息、时间模式等
- **代码质量评估**: 基于DMM模型的代码可维护性分析
- **文件关联挖掘**: 使用关联规则算法发现文件间的协同变更模式
- **影响预测**: 预测代码变更可能影响的文件
- **聚类分析**: 基于提交特征的智能聚类
- **风险预测**: DMM风险回归和分类预测

### 📊 可视化展示
- 交互式网络图展示文件关联关系
- 代码质量趋势图
- 开发者贡献分析图
- 特征重要性分析图

### 🤖 机器学习模型
- 随机森林
- XGBoost
- 支持向量机
- 神经网络
- 决策树

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 🚀 超级快速开始

```bash
# 一键分析任何GitHub仓库
python quick_start.py
```

然后输入GitHub仓库URL即可自动收集数据并分析！

### 方式一：直接从GitHub仓库分析 (推荐)

```bash
# 从GitHub仓库直接收集数据并分析 (支持断点续传)
python analyze_csv_improved.py --repo-url https://github.com/BeyondDimension/SteamTools --project-name steamtools

# 指定时间范围和输出目录
python analyze_csv_improved.py \
  --repo-url https://github.com/apache/flink \
  --project-name flink \
  --since 2023-01-01 --to 2023-12-31 \
  --output results/flink_analysis

# 强制重新开始收集 (忽略现有数据)
python analyze_csv_improved.py --repo-url https://github.com/BeyondDimension/SteamTools --project-name steamtools --no-resume
```

### 方式二：先收集数据，再分析

```bash
# 1. 收集GitHub仓库数据 (支持断点续传)
python collect_repository.py https://github.com/BeyondDimension/SteamTools -o data/steamtools.csv

# 2. 分析收集的数据
python analyze_csv_improved.py data/steamtools.csv --project-name steamtools
```

### 方式三：批量处理多个仓库

```bash
# 创建配置文件
python batch_collect_repositories.py --create-sample-config

# 批量收集多个仓库
python batch_collect_repositories.py --config repo_config_sample.json
```

### 编程接口使用

```python
from src.core.analyzer import GitAnalyzer
from src.data_collection.repository_collector import RepositoryCollector

# 收集数据
collector = RepositoryCollector()
csv_file = collector.collect_repository_data(
    repo_url="https://github.com/user/repo",
    output_file="data/repo_data.csv"
)

# 分析数据
analyzer = GitAnalyzer()
results = analyzer.analyze_csv_data(csv_file, "project_name")
```

## 项目结构

```
GitAnalytics/
├── src/                          # 源代码目录
│   ├── core/                     # 核心分析模块
│   │   └── analyzer.py           # 主分析器
│   ├── analysis/                 # 各种分析算法
│   │   ├── risk_predictor.py     # DMM风险预测
│   │   ├── file_impact_predictor.py # 文件影响预测
│   │   ├── clustering_analyzer.py # 聚类分析
│   │   ├── association_miner.py  # 关联规则挖掘
│   │   ├── quality_analyzer.py   # 代码质量分析
│   │   └── prediction_models.py  # 预测模型
│   ├── data_collection/          # 数据收集模块
│   │   └── repository_collector.py # GitHub仓库数据收集
│   ├── visualization/            # 可视化模块
│   │   └── report_generator.py   # 报告生成器
│   └── utils/                    # 工具函数
│       ├── data_extractor.py     # 数据提取
│       ├── data_preprocessor.py  # 数据预处理
│       └── logger.py             # 日志工具
├── data/                         # 数据存储目录
├── examples/                     # 示例代码
│   └── collect_and_analyze_example.py
├── collect_repository.py         # 单仓库数据收集工具
├── batch_collect_repositories.py # 批量数据收集工具
├── analyze_csv_improved.py       # 综合分析工具
├── quick_start.py                # 快速开始工具
├── repo_config_sample.json       # 示例配置文件
├── DATA_COLLECTION_README.md     # 数据收集功能说明
├── IMPROVEMENTS_SUMMARY.md       # 功能改进总结
├── requirements.txt              # 依赖文件
├── setup.py                      # 安装脚本
└── LICENSE                       # 许可证
```

## 分析报告示例

分析完成后，系统会生成包含以下内容的综合报告：

1. **仓库概览**: 基本统计信息
2. **代码质量分析**: DMM指标和趋势
3. **文件关联网络**: 协同变更的文件关系
4. **开发者分析**: 贡献模式和协作关系
5. **预测模型结果**: 各种预测任务的性能

## 技术栈

- **数据挖掘**: PyDriller, Pandas, NumPy
- **机器学习**: Scikit-learn, XGBoost, Imbalanced-learn
- **关联规则**: MLxtend
- **可视化**: Matplotlib, Seaborn, NetworkX, Plotly
- **自然语言处理**: Gensim (Word2Vec)

## 详细文档

- [使用指南](USAGE_GUIDE.md) - 快速上手指南
- [数据收集功能详细说明](DATA_COLLECTION_README.md) - 如何从GitHub仓库收集数据
- [分析功能改进总结](IMPROVEMENTS_SUMMARY.md) - 最新功能改进说明

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 作者

基于PyDriller框架开发的Git仓库分析工具 