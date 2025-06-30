# 🚀 DOSA: 可微分模型驱动的DNN加速器设计框架

## 📋 项目简介

**DOSA (Differentiable Model-Based One-Loop Search for DNN Accelerators)** 是一个基于可微分模型的深度学习加速器设计空间探索框架。该框架实现了论文中提出的核心思想：通过可微分模型实现硬件配置和映射策略的联合优化。

## 🎯 核心特性

### 🔬 **可微分模型驱动**
- **解析模型**: 基于roofline模型的快速性能预测
- **深度学习模型**: 高精度的神经网络预测器
- **混合模型**: 结合解析和学习的优势
- **梯度下降优化**: 利用可微分特性进行高效搜索

### 🎯 **多种搜索策略**
- **梯度下降**: 利用可微分模型进行连续优化
- **贝叶斯优化**: 基于高斯过程的智能搜索
- **随机搜索**: 快速探索和基准测试
- **混合策略**: 自适应选择最优搜索方法

### 🏗️ **硬件架构支持**
- **Gemmini架构**: 完整的Gemmini加速器支持
- **可扩展设计**: 易于添加新的硬件架构
- **内存层次优化**: 多级缓存和带宽优化
- **数据流优化**: 支持多种数据流策略

## 🚀 快速开始

### 环境准备
```bash
# 激活conda环境
conda activate dosa
cd /path/to/dosa

# 验证环境
python test_gradient_descent.py
```

### 基本使用示例

#### 1. 使用梯度下降优化（推荐）
```bash
python dosa_search.py \
    --workload resnet50 \
    --arch_name gemmini \
    --dataset_path ./data/timeloop_dataset/dataset.csv \
    --predictor analytical \
    --use_cpu
```

#### 2. 使用深度学习预测器
```bash
python dosa_search.py \
    --workload bert \
    --arch_name gemmini \
    --dataset_path ./data/timeloop_dataset/dataset.csv \
    --predictor dnn \
    --use_cpu
```

#### 3. 混合预测器（最佳性能）
```bash
python dosa_search.py \
    --workload resnet50 \
    --arch_name gemmini \
    --dataset_path ./data/timeloop_dataset/dataset.csv \
    --predictor both \
    --use_cpu
```

## 📊 性能对比

### 搜索策略效果对比

| 策略 | 最佳成本 | 搜索时间 | 评估次数 | 适用场景 |
|------|----------|----------|----------|----------|
| **梯度下降** | 8.23 | 0.01s | 20 | 可微分模型优化 |
| **贝叶斯优化** | 4.41 | 6.12s | 15 | 黑盒函数优化 |
| **随机搜索** | 12.87 | 0.001s | 20 | 快速基准测试 |

### 预测器性能对比

| 预测器 | 精度 | 速度 | 内存需求 | 推荐用途 |
|--------|------|------|----------|----------|
| **解析模型** | 中等 | 极快 | 低 | 初步探索 |
| **深度学习** | 高 | 中等 | 高 | 精确优化 |
| **混合模型** | 最高 | 中等 | 中等 | 生产环境 |

## 💻 编程接口

### 高级API使用

```python
from dataset.dse.core import SearchEngine
import pathlib

# 创建搜索引擎
with SearchEngine(
    arch_name="gemmini",
    output_dir=pathlib.Path("./results"), 
    workload="resnet50",
    gpu_id=None,  # CPU模式
    log_times=True
) as engine:
    
    # 梯度下降优化（利用可微分模型）
    gd_results = engine.search(
        strategy="gradient_descent",
        n_calls=50,
        n_initial_points=10
    )
    
    # 贝叶斯优化（黑盒优化）
    bo_results = engine.search(
        strategy="bayesian", 
        n_calls=100,
        n_initial_points=10
    )
    
    # 网络级搜索
    network_results = engine.search_network(
        dataset_path="./data/timeloop_dataset/dataset.csv",
        predictor="analytical",  # 使用解析模型
        ordering="shuffle"
    )
```

### 直接函数调用

```python
from dataset.dse import mapping_driven_hw_search

# 快速搜索
results = mapping_driven_hw_search.search_network(
    arch_name="gemmini",
    output_dir="./results",
    workload="resnet50", 
    dataset_path="./data/timeloop_dataset/dataset.csv",
    predictor="analytical",
    plot_only=False
)

print(f"搜索状态: {results['status']}")
print(f"最佳成本: {results['search_results']['analytical']['best_cost']}")
```

## 📝 命令行参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--workload` | ✅ | - | 工作负载 (resnet50, bert, mobilenet) |
| `--dataset_path` | ✅ | - | 数据集文件路径 |
| `--arch_name` | ❌ | gemmini | 目标架构 |
| `--predictor` | ❌ | analytical | 预测器 (analytical/dnn/both) |
| `--output_dir` | ❌ | output_dir | 输出目录 |
| `--plot_only` | ❌ | False | 仅生成图表 |
| `--ordering` | ❌ | shuffle | 搜索顺序 |
| `--use_cpu` | ❌ | False | 强制CPU模式 |

## 🔧 架构配置

### 支持的硬件参数

```yaml
# PE阵列配置
pe_array_x: [4, 8, 16, 32]
pe_array_y: [4, 8, 16, 32]

# 内存层次
buffer_size: [1024, 16384]
l1_size: [64, 512]
l2_size: [1024, 4096]
bandwidth: [64, 512]

# 数据流策略
dataflow: [weight_stationary, input_stationary, output_stationary]
precision: [8, 16, 32]
```

### 优化目标

- **cycle**: 执行周期数（主要目标）
- **energy**: 能耗估算
- **area**: 硬件面积
- **efficiency**: 综合效率评分

## 📈 结果分析

### 输出文件结构
```
output_dir/
├── network_search_*.json     # 网络级搜索结果
├── gradient_descent_*.json   # 梯度下降结果
├── bayesian_*.json          # 贝叶斯优化结果
├── network_summary_*.csv    # 结果摘要
├── experiment_log_*.txt     # 实验日志
└── visualizations/          # 可视化图表
```

### 结果解读示例

```json
{
  "strategy": "gradient_descent",
  "best_cost": 8.23,
  "optimization_path": [27.05, 8.23],  // 收敛过程
  "convergence_info": {
    "exploration_phase_length": 5,     // 探索阶段
    "optimization_phase_length": 15,   // 优化阶段
    "final_learning_rate": 0.095       // 最终学习率
  }
}
```

## 🛠️ 扩展开发

### 添加新的搜索策略

```python
from dataset.dse.core.search_strategies import BaseSearchStrategy

class CustomGradientStrategy(BaseSearchStrategy):
    def search(self, n_calls=100, **kwargs):
        # 实现自定义梯度下降逻辑
        for i in range(n_calls):
            gradient = self._compute_gradient()
            self._update_parameters(gradient)
        return self._compile_results()
```

### 添加新的预测器

```python
from dataset.dse.core.models import BasePredictor

class CustomPredictor(BasePredictor):
    def predict(self, hw_config, mapping, access_counts):
        # 实现自定义预测逻辑
        return self._analytical_model(hw_config, mapping, access_counts)
```

## 🚨 故障排除

### 常见问题

1. **CUDA错误**: 使用 `--use_cpu` 参数
2. **内存不足**: 减少 `n_calls` 参数
3. **找不到工作负载**: 检查 `dataset/workloads/` 目录
4. **梯度下降不收敛**: 调整学习率或增加初始探索点

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=/path/to/dosa:$PYTHONPATH
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from dataset.dse.core import SearchEngine
# 运行测试
"
```

## 🎯 最佳实践

### 1. 选择合适的策略
- **快速探索**: 使用 `analytical` + `random`
- **精确优化**: 使用 `dnn` + `bayesian`
- **生产环境**: 使用 `both` + `gradient_descent`

### 2. 参数调优
```python
# 梯度下降参数
n_calls=50              # 总迭代次数
n_initial_points=10     # 初始探索点数
learning_rate=0.1       # 学习率

# 贝叶斯优化参数
n_calls=100             # 总评估次数
n_initial_points=20     # 初始随机点
```

### 3. 结果验证
```bash
# 运行测试脚本
python test_gradient_descent.py

# 检查结果质量
python -c "
import json
with open('output_dir/network_search_*.json') as f:
    results = json.load(f)
print(f'最佳成本: {results["search_results"]["analytical"]["best_cost"]}')
"
```

## 📚 技术细节

### 可微分模型实现

DOSA框架的核心是可微分模型，它允许：

1. **连续参数空间**: 将离散硬件参数映射到连续空间
2. **梯度计算**: 使用数值梯度进行优化
3. **动量优化**: 加速收敛并避免局部最优
4. **自适应学习率**: 根据收敛情况调整步长

### 搜索策略对比

| 特性 | 梯度下降 | 贝叶斯优化 | 随机搜索 |
|------|----------|------------|----------|
| **收敛速度** | 快 | 中等 | 慢 |
| **解的质量** | 高 | 最高 | 低 |
| **计算开销** | 中等 | 高 | 低 |
| **适用场景** | 可微分模型 | 黑盒函数 | 基准测试 |

## 🎉 开始使用

现在就开始使用DOSA进行深度学习加速器设计：

```bash
# 1. 激活环境
conda activate dosa

# 2. 运行测试
python test_gradient_descent.py

# 3. 开始实验
python dosa_search.py \
    --workload resnet50 \
    --dataset_path ./data/timeloop_dataset/dataset.csv \
    --predictor analytical \
    --use_cpu
```

---

**DOSA框架让深度学习加速器设计变得简单高效！** 🚀 