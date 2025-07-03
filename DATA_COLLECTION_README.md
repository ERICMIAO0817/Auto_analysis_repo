# GitHub仓库数据收集功能

GitAnalytics现在支持直接从GitHub仓库收集commit数据，无需手动下载和处理数据文件。

## 功能特性

- 🚀 直接从GitHub仓库URL收集commit数据
- 🔄 **断点续传**: 自动检测已有数据，从中断点继续收集
- 📊 自动提取DMM质量指标 (dmm_unit_size, dmm_unit_complexity, dmm_unit_interfacing)
- 🔍 智能分析文件类型和变更类型
- ⏰ 支持时间范围过滤
- 🌿 支持指定分支分析
- 📁 支持文件类型过滤
- 🔄 批量处理多个仓库
- 📈 收集完成后自动进行综合分析

## 使用方法

### 1. 单个仓库数据收集

#### 基础用法
```bash
# 收集SteamTools仓库的所有commit数据
python collect_repository.py https://github.com/BeyondDimension/SteamTools

# 指定输出文件
python collect_repository.py https://github.com/BeyondDimension/SteamTools -o data/raw/steamtools.csv
```

#### 高级用法
```bash
# 指定时间范围
python collect_repository.py https://github.com/apache/flink --since 2023-01-01 --to 2023-12-31

# 指定分支和文件类型
python collect_repository.py https://github.com/apache/flink --branch main --file-types .java .xml

# 包含DMM值为空的提交
python collect_repository.py https://github.com/BeyondDimension/SteamTools --include-empty-dmm

# 详细输出模式
python collect_repository.py https://github.com/apache/spark -v

# 断点续传功能
python collect_repository.py https://github.com/BeyondDimension/SteamTools --resume
python collect_repository.py https://github.com/BeyondDimension/SteamTools --no-resume  # 重新开始
```

### 2. 批量收集多个仓库

#### 使用配置文件
```bash
# 创建示例配置文件
python batch_collect_repositories.py --create-sample-config

# 使用配置文件批量收集
python batch_collect_repositories.py --config repo_config_sample.json
```

#### 直接指定仓库列表
```bash
# 批量收集多个仓库
python batch_collect_repositories.py --repos \
  https://github.com/apache/flink \
  https://github.com/apache/spark \
  https://github.com/apache/kafka

# 为所有仓库指定相同的过滤条件
python batch_collect_repositories.py --repos \
  https://github.com/apache/flink \
  https://github.com/apache/spark \
  --since 2023-01-01 --file-types .java .scala
```

### 3. 一体化分析 (收集+分析)

```bash
# 从GitHub仓库直接收集数据并分析
python analyze_csv_improved.py --repo-url https://github.com/BeyondDimension/SteamTools --project-name steamtools

# 指定时间范围和输出目录
python analyze_csv_improved.py \
  --repo-url https://github.com/apache/flink \
  --project-name flink \
  --since 2023-01-01 --to 2023-12-31 \
  --output results/flink_analysis

# 指定文件类型和分支
python analyze_csv_improved.py \
  --repo-url https://github.com/apache/spark \
  --project-name spark \
  --branch main --file-types .scala .java \
  --output results/spark_analysis
```

## 配置文件格式

批量收集时可以使用JSON配置文件，格式如下：

```json
{
  "repositories": [
    {
      "repo_url": "https://github.com/BeyondDimension/SteamTools",
      "since": "2023-01-01",
      "to": "2023-12-31",
      "only_in_branch": "main",
      "skip_empty_dmm": true
    },
    {
      "repo_url": "https://github.com/apache/flink",
      "only_modifications_with_file_types": [".java", ".xml"],
      "skip_empty_dmm": true
    },
    {
      "repo_url": "https://github.com/apache/spark",
      "since": "2023-06-01",
      "only_modifications_with_file_types": [".scala", ".java"],
      "skip_empty_dmm": false
    }
  ]
}
```

## 断点续传功能

GitAnalytics支持智能断点续传，可以自动检测`data/raw`目录中的现有数据文件，并从中断的地方继续收集。

### 工作原理

1. **自动检测**: 程序启动时自动检查输出文件是否已存在
2. **获取断点**: 读取现有文件的最后一个commit hash作为断点
3. **继续收集**: 从断点之后的commit开始收集新数据
4. **追加模式**: 新数据追加到现有文件末尾

### 使用场景

- **大型仓库**: 避免重复下载已有数据
- **网络中断**: 网络中断后可以继续之前的进度
- **增量更新**: 定期更新仓库数据时只收集新的commit
- **节省时间**: 大幅减少重复收集的时间

### 示例

```bash
# 首次收集（假设收集了1000个commit后中断）
python collect_repository.py https://github.com/apache/flink

# 断点续传（自动从第1001个commit开始）
python collect_repository.py https://github.com/apache/flink --resume

# 强制重新开始（忽略现有数据）
python collect_repository.py https://github.com/apache/flink --no-resume
```

### 日志输出示例

```
2024-01-15 10:30:15 - INFO - 发现现有数据，最后一个commit: abc123def456
2024-01-15 10:30:15 - INFO - 启用断点续传，已有 1000 个commit，从 abc123def456 之后继续
2024-01-15 10:30:20 - INFO - 找到断点位置: abc123def456
2024-01-15 10:30:25 - INFO - 已处理 1100 个提交（新增 100 个）
2024-01-15 10:30:30 - INFO - 断点续传完成! 原有 1000 个提交，新增 150 个提交，总共 1150 个提交
```

## 输出数据格式

收集的CSV文件包含以下字段：

### 基础信息
- `hash`: 提交哈希值
- `msg`: 提交消息
- `author`: 作者姓名
- `committer`: 提交者姓名
- `author_date`: 作者时间
- `committer_date`: 提交时间

### 变更统计
- `deletions`: 删除行数
- `insertions`: 插入行数
- `lines`: 总变更行数
- `files`: 修改文件数

### DMM质量指标
- `dmm_unit_size`: DMM单元大小
- `dmm_unit_complexity`: DMM单元复杂度
- `dmm_unit_interfacing`: DMM单元接口复杂度

### 分析字段
- `main_file_type`: 主要文件类型 (如 .java, .py, .cs)
- `main_change_type`: 主要变更类型 (ADD, DELETE, MODIFY, RENAME)

## 性能优化建议

1. **大型仓库**: 对于大型仓库，建议使用时间范围过滤
2. **网络优化**: 使用本地Git仓库路径可以提高速度
3. **并行处理**: 批量收集时会自动并行处理多个仓库
4. **存储空间**: 大型项目的CSV文件可能很大，确保有足够存储空间

## 故障排除

### 常见问题

1. **网络连接问题**
   ```bash
   # 使用详细模式查看错误信息
   python collect_repository.py https://github.com/user/repo -v
   ```

2. **DMM值为空**
   ```bash
   # 包含DMM值为空的提交
   python collect_repository.py https://github.com/user/repo --include-empty-dmm
   ```

3. **内存不足**
   ```bash
   # 使用时间范围限制数据量
   python collect_repository.py https://github.com/user/repo --since 2023-01-01
   ```

### 日志文件

- 单个仓库收集: `repository_collection.log`
- 批量收集: `batch_repository_collection.log`
- 综合分析: 在输出目录中的日志文件

## 集成到现有工作流

### 1. 定期数据更新
```bash
#!/bin/bash
# 每日更新脚本
python collect_repository.py https://github.com/myorg/myproject --since $(date -d "1 day ago" +%Y-%m-%d)
```

### 2. CI/CD集成
```yaml
# GitHub Actions示例
- name: Collect Repository Data
  run: |
    python collect_repository.py ${{ github.repository_url }} --since 2023-01-01
    python analyze_csv_improved.py --repo-url ${{ github.repository_url }} --project-name ${{ github.event.repository.name }}
```

### 3. 多项目监控
```bash
# 监控多个项目的质量趋势
python batch_collect_repositories.py --config monitoring_repos.json
```

## 下一步

收集数据后，您可以：

1. 使用 `analyze_csv_improved.py` 进行综合分析
2. 运行DMM风险预测模型
3. 分析文件影响关系
4. 生成质量趋势报告
5. 建立持续监控系统

更多分析功能请参考主要的README文档。 