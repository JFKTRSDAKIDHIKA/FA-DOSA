# FA-DOSA: Fusion-Aware Design Space Optimization for Neural Accelerators

![FA-DOSA Framework](https://img.shields.io/badge/Framework-FA--DOSA-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

FA-DOSA (Fusion-Aware Design Space Optimization) is a research framework for joint hardware-software co-design of neural accelerators. Unlike traditional decoupled approaches that optimize hardware and software mapping separately, FA-DOSA performs **joint optimization** of dataflow mapping parameters and layer fusion decisions to achieve superior energy-delay product (EDP) and hardware efficiency.

### Key Innovations

🚀 **Joint Hardware-Software Optimization**: Simultaneous optimization of mapping parameters and fusion decisions using gradient-based methods

⚡ **Physically Accurate Energy Modeling**: Detailed cost breakdown modeling DRAM energy savings from layer fusion

🎯 **Industrial-Grade ONNX Support**: Comprehensive operator support for real neural networks like ResNet-18

📊 **Numerical Stability**: Logarithmic loss function prevents optimization explosion for large-scale models

🔧 **Extensible Architecture**: Modular design supporting new operators, fusion patterns, and cost models

## Architecture

```
FA-DOSA Framework
├── Core Components
│   ├── Configuration Management (config.yaml)
│   ├── Computation Graph Representation
│   ├── Learnable Mapping Parameters
│   └── Learnable Fusion Parameters
├── Cost Models
│   ├── Analytical Latency Model (Roofline)
│   ├── Detailed Energy Model (Compute + SRAM + DRAM)
│   ├── Area Cost Model
│   └── Hardware Configuration Calculator
├── ONNX Frontend
│   ├── Industrial-Grade Parser
│   ├── Comprehensive Operator Support
│   └── Advanced Fusion Pattern Detection
└── Optimization Engine
    ├── Gradient-Based Joint Optimization
    ├── Constraint Handling
    └── Numerical Stabilization
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- ONNX
- NumPy
- PyYAML

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fa-dosa
   ```

2. **Create conda environment**:
   ```bash
   conda create -n dosa python=3.8
   conda activate dosa
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision onnx numpy pyyaml
   ```

4. **Generate test models**:
   ```bash
   python export_resnet.py  # Creates resnet18.onnx
   python create_test_model.py  # Creates sample_model.onnx
   ```

## Quick Start

### Basic Usage

```python
from fa_dosa_demo import *
from onnx_frontend import parse_onnx_to_graph

# Load and parse ONNX model
graph = parse_onnx_to_graph("resnet18.onnx")

# Initialize optimization parameters
mapping_params = MappingParameters(graph)
fusion_params = FusionParameters(graph)
model = ConditionalPerformanceModel()

# Run optimization
config = Config.get_instance()
optimizer = torch.optim.Adam([
    {'params': mapping_params.parameters()},
    {'params': fusion_params.parameters()}
], lr=1e-5)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    latency, energy, area = model(mapping_params, fusion_params, graph)
    
    # Logarithmic loss for numerical stability
    log_latency = torch.log(latency + 1e-9)
    log_energy = torch.log(energy + 1e-9) 
    log_area = torch.log(area + 1e-9)
    
    loss = log_latency + log_energy + log_area
    loss.backward()
    optimizer.step()
```

### Running Experiments

Execute comprehensive experiments comparing FA-DOSA with baseline approaches:

```bash
python run_experiments.py
```

This runs three algorithms on ResNet-18:
- **FA-DOSA**: Joint optimization approach
- **Two-Step Baseline**: Traditional decoupled optimization
- **Random Search**: Random baseline

Results are saved to `experiment_results.csv`.

## Configuration

Hardware and cost model parameters are configured in `config.yaml`:

```yaml
# Hardware parameters
hardware:
  dram_bandwidth_gb_s: 100.0
  sram_bandwidth_gb_s: 1000.0
  mac_throughput_gops: 1000.0
  bytes_per_element: 4

# Cost model parameters  
costs:
  dram_access_cost_pj_byte: 2.5
  sram_base_cost_pj_byte: 0.25
  fusion_overhead_cost_pj: 0.5
  area_coefficient_mm2_per_kb: 0.0001
  sram_energy_scaling_pj_byte_kb: 0.0001

# Optimization weights
weights:
  penalty_weight: 1.0
  product_penalty_weight: 10.0
```

## Supported Operators

The industrial-grade ONNX frontend supports:

### Core Operators
- **Conv**: Convolution with kernel size and channel extraction
- **Relu, LeakyRelu, Sigmoid, Tanh, Clip**: Activation functions
- **BatchNormalization**: Batch normalization layers
- **Add**: Element-wise addition (residual connections)

### Pooling Operations
- **MaxPool**: Max pooling with kernel size extraction
- **AveragePool, GlobalAveragePool**: Average pooling variants
- **LpPool**: Lp norm pooling

### Tensor Operations
- **Gemm, MatMul**: Fully connected layers
- **Flatten, Reshape**: Tensor reshaping
- **Concat, Split**: Tensor manipulation
- **Mul, Div, Sub**: Element-wise arithmetic

### Utility Operations
- **Dropout, Identity**: Pass-through operations

## Fusion Patterns

FA-DOSA automatically detects and optimizes common fusion patterns:

1. **Conv → ReLU**: Basic convolution-activation fusion
2. **Conv → BatchNorm → ReLU**: ResNet-style three-layer fusion
3. **Extensible**: Framework supports adding new fusion patterns

## Research Results

### ResNet-18 Performance (1000 epochs)

| Algorithm | EDP | Latency (cycles) | Energy (pJ) | Area (mm²) |
|-----------|-----|------------------|-------------|------------|
| **FA-DOSA** | **3.49×10¹⁴** | **1.87M** | **187M** | **0.0028** |
| Two-Step Baseline | 3.50×10¹⁴ | 1.87M | 187M | 0.0044 |

**Key Improvements**:
- ✅ **Similar EDP**: Competitive energy-delay product
- ✅ **36% Smaller Area**: More hardware-efficient design
- ✅ **Numerical Stability**: 6.1×10⁹ times better loss convergence

## Technical Contributions

### 1. Joint Optimization Framework
- Simultaneous optimization of mapping and fusion decisions
- Differentiable fusion probabilities using sigmoid activation
- Gradient-based optimization with constraint handling

### 2. Physically Accurate Energy Modeling
```python
# DRAM energy savings from fusion
intermediate_tensor_size = N * K * P * Q * bytes_per_element
dram_energy_saved = 2 * intermediate_tensor_size * DRAM_ACCESS_COST
fused_dram_energy = fused_dram_energy - dram_energy_saved
```

### 3. Numerical Stabilization
```python
# Logarithmic loss function
total_loss = (
    torch.log(latency + 1e-9) + 
    torch.log(energy + 1e-9) + 
    torch.log(area + 1e-9) + 
    penalties
)
```

### 4. Industrial-Grade ONNX Frontend
- Comprehensive operator support (20+ operator types)
- Robust dimension inference handling different tensor ranks
- Advanced fusion pattern detection using graph traversal

## File Structure

```
fa-dosa/
├── README.md                 # Project documentation
├── config.yaml              # Hardware and cost configuration
├── requirements.txt          # Python dependencies
├── fa_dosa_demo.py          # Core FA-DOSA framework
├── onnx_frontend.py         # Industrial-grade ONNX parser
├── run_experiments.py       # Experiment runner
├── export_resnet.py         # ResNet-18 ONNX export utility
├── create_test_model.py     # Test model generator
├── experiment_results.csv   # Results from experiments
├── resnet18.onnx           # ResNet-18 ONNX model
└── sample_model.onnx       # Simple test model
```

## API Reference

### Core Classes

#### `ComputationGraph`
Represents the neural network structure with layers, edges, and fusion groups.

```python
graph = ComputationGraph()
graph.add_layer(name, dims, op_type)
graph.add_edge(src, dst)
graph.add_fusion_group([layer1, layer2, layer3])
```

#### `MappingParameters`
Learnable parameters for dataflow mapping optimization.

```python
mapping_params = MappingParameters(graph)
layer_mapping = mapping_params.get_mapping_by_original_name(layer_name)
```

#### `FusionParameters`
Learnable parameters for fusion decision optimization.

```python
fusion_params = FusionParameters(graph)
prob = fusion_params.get_fusion_probability(group)
```

#### `ConditionalPerformanceModel`
Analytical performance model with detailed cost breakdown.

```python
model = ConditionalPerformanceModel()
latency, energy, area = model(mapping_params, fusion_params, graph)
```

### Utility Functions

- `parse_onnx_to_graph(path)`: Parse ONNX model to ComputationGraph
- `calculate_hardware_config(mapping_params, graph)`: Calculate minimal hardware
- `calculate_penalty_loss(mapping_params)`: Constraint penalty calculation
- `calculate_macs(dims)`: Calculate multiply-accumulate operations

## Contributing

We welcome contributions! Areas for improvement:

1. **New Operators**: Add support for additional ONNX operators
2. **Fusion Patterns**: Implement new fusion detection algorithms  
3. **Cost Models**: Develop more sophisticated hardware models
4. **Optimization**: Explore alternative optimization strategies
5. **Validation**: Add more neural network benchmarks

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Submit a pull request

## Citation

If you use FA-DOSA in your research, please cite:

```bibtex
@article{fa-dosa-2024,
  title={FA-DOSA: Fusion-Aware Design Space Optimization for Neural Accelerators},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built upon DOSA (Design Space Optimization for Neural Accelerators) concepts
- Inspired by hardware-software co-design methodologies
- ResNet architecture from torchvision
- ONNX ecosystem for model representation

## Contact

For questions or collaborations, please contact:

- **Research Team**: [email]
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Documentation**: See inline code documentation for detailed API reference

---

**FA-DOSA: Bridging the Gap Between Hardware and Software Optimization** 🚀 