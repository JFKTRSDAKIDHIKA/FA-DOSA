# 🚀 DOSA深度学习加速器设计框架使用指南

## 📋 项目简介

**DOSA (Differentiable Model-Based One-Loop Search for DNN Accelerators)** 是一个用于深度学习加速器设计空间探索的现代化框架。

## 🎯 主要功能

- **网络级设计空间探索**: 自动搜索最优硬件配置
- **多种搜索策略**: 贝叶斯优化、梯度下降、随机搜索
- **智能预测器**: 分析模型、深度学习模型、混合模型
- **硬件架构生成**: 自动生成和优化硬件配置
- **可视化分析**: 丰富的图表和性能分析工具

## 🚀 快速开始

### 环境激活
```bash
conda activate dosa
cd /path/to/dosa
```

### 基本使用

#### 1. 简单搜索实验
```bash
python dosa_search.py \
    --workload bert \
    --arch_name gemmini \
    --dataset_path ./data/timeloop_dataset/dataset.csv \
    --predictor analytical
```

#### 2. 强制CPU模式（推荐）
```bash
python dosa_search.py \
    --workload bert \
    --arch_name gemmini \
    --dataset_path ./data/timeloop_dataset/dataset.csv \
    --predictor analytical \
    --use_cpu
```

#### 3. 使用深度学习预测器
```bash
python dosa_search.py \
    --workload resnet50 \
    --arch_name gemmini \
    --dataset_path ./data/timeloop_dataset/dataset.csv \
    --predictor dnn \
    --use_cpu
```

#### 4. 只生成可视化图表
```bash
python dosa_search.py \
    --workload bert \
    --dataset_path ./data/timeloop_dataset/dataset.csv \
    --plot_only \
    --use_cpu
```

## 📝 命令行参数详解

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--workload` | ✅ | - | 工作负载名称 (bert, resnet50, etc.) |
| `--arch_name` | ❌ | gemmini | 目标架构名称 |
| `--dataset_path` | ✅ | - | 数据集文件路径 |
| `--predictor` | ❌ | analytical | 预测器类型 (analytical/dnn/both) |
| `--output_dir` | ❌ | output_dir | 结果输出目录 |
| `--plot_only` | ❌ | False | 只生成图表，不运行搜索 |
| `--ordering` | ❌ | shuffle | 搜索顺序 (shuffle/sequential/random) |
| `--use_cpu` | ❌ | False | 强制使用CPU模式 |

## 💻 高级编程接口

### 使用SearchEngine类

```python
from dataset.dse.core import SearchEngine
import pathlib

# 创建搜索引擎
with SearchEngine(
    arch_name="gemmini",
    output_dir=pathlib.Path("./my_results"), 
    workload="bert",
    gpu_id=None,  # None表示CPU模式
    log_times=True
) as engine:
    
    # 运行贝叶斯优化搜索
    results = engine.search(
        strategy="bayesian", 
        n_calls=100,
        n_initial_points=10
    )
    
    # 运行梯度下降搜索
    gd_results = engine.search(
        strategy="gradient_descent",
        n_calls=50
    )
    
    # 运行随机搜索
    random_results = engine.search(
        strategy="random",
        n_calls=200
    )
    
    # 获取搜索摘要
    summary = engine.get_search_summary()
    print(f"搜索了 {summary['num_layers']} 个层")
    print(f"可用策略: {summary['available_strategies']}")
```

### 使用工具函数

```python
from dataset.common.utils import FileHandler, MathUtils, ProcessManager

# 文件操作
config = FileHandler.load_yaml("config.yaml")
results = FileHandler.load_json("results.json")
FileHandler.save_csv(data, "output.csv")

# 数学计算
factors = MathUtils.get_prime_factors(128)
correlation = MathUtils.get_correlation(data1, data2)

# 进程管理
result = ProcessManager.run_command("timeloop-mapper", timeout=300)
```

### 直接调用搜索函数

```python
from dataset.dse import mapping_driven_hw_search

results = mapping_driven_hw_search.search_network(
    arch_name="gemmini",
    output_dir="./results",
    workload="bert", 
    dataset_path="./data/timeloop_dataset/dataset.csv",
    predictor="analytical",
    plot_only=False,
    ordering="shuffle"
)

print(f"搜索状态: {results.get('status', 'unknown')}")
print(f"设备信息: {results.get('device_info', {})}")
```

## 🔧 常见工作负载

### 支持的工作负载
- `bert`: BERT语言模型
- `resnet50`: ResNet-50图像分类
- `mobilenet`: MobileNet轻量级网络
- `transformer`: Transformer模型

### 支持的架构
- `gemmini`: Gemmini加速器架构
- 其他架构可通过配置文件添加

## 📊 结果分析

### 输出文件结构
```
output_dir/
├── search_results_*.json    # 搜索结果
├── performance_logs/        # 性能日志
├── visualizations/          # 生成的图表
└── configs/                 # 使用的配置文件
```

### 结果解读
- **cycle**: 执行周期数
- **energy**: 能耗估算
- **area**: 硬件面积
- **efficiency**: 效率评分

## 🚨 常见问题解决

### CUDA兼容性问题
如果遇到CUDA错误，框架会自动回退到CPU模式：
```bash
# 手动强制CPU模式
python dosa_search.py --use_cpu [其他参数]
```

### 内存不足
```bash
# 减少搜索调用次数
python -c "
from dataset.dse.core import SearchEngine
engine = SearchEngine(...)
results = engine.search('bayesian', n_calls=50)  # 减少到50次
"
```

### 找不到工作负载
确保工作负载目录存在：
```bash
ls dataset/workloads/bert/  # 检查bert工作负载
ls dataset/workloads/       # 查看所有可用工作负载
```

## 🛠️ 扩展和定制

### 添加新的搜索策略
```python
from dataset.dse.core.search_strategies import BaseSearchStrategy

class MyCustomStrategy(BaseSearchStrategy):
    def search(self, n_calls=100, **kwargs):
        # 实现自定义搜索逻辑
        pass
```

### 添加新的预测器
```python
from dataset.dse.core.models import BasePredictor

class MyPredictor(BasePredictor):
    def predict(self, features):
        # 实现自定义预测逻辑
        pass
```

## 📈 性能优化建议

1. **使用CPU模式**: 除非确定GPU兼容，建议使用`--use_cpu`
2. **调整搜索次数**: 根据资源情况调整`n_calls`参数
3. **并行处理**: 框架内置并行优化，无需额外配置
4. **缓存结果**: 重复实验会自动使用缓存结果

## 🎯 最佳实践

1. **开始实验前先测试**:
   ```bash
   python dosa_test.py  # 运行框架测试
   ```

2. **使用合适的预测器**:
   - `analytical`: 快速，适合初步探索
   - `dnn`: 精确，适合最终优化
   - `both`: 综合，适合全面分析

3. **保存重要结果**:
   ```bash
   python dosa_search.py --output_dir ./important_results [其他参数]
   ```

4. **监控资源使用**:
   ```bash
   htop  # 监控CPU和内存使用
   ```

---

## 🎉 开始使用

现在你可以开始使用DOSA进行深度学习加速器设计了！从简单的命令开始：

```bash
conda activate dosa
python dosa_search.py --workload bert --dataset_path ./data/timeloop_dataset/dataset.csv --use_cpu
```

如有问题，检查日志输出或运行测试脚本 `python dosa_test.py`。 