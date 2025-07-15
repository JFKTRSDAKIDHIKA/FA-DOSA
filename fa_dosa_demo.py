"""
Enhanced FA-DOSA Performance and Energy Modeling Framework

This module implements a sophisticated and extensible cost modeling framework for 
hardware accelerator design space exploration, suitable for top-tier publications.

Expected config.yaml structure with new parameters:

```yaml
# Existing parameters (unchanged)
dimensions:
  loop_dims: ['N', 'K', 'C', 'P', 'Q', 'R', 'S']

hardware:
  num_memory_levels: 2
  bytes_per_element: 2
  dram_bandwidth_gb_s: 100.0
  sram_bandwidth_gb_s: 1000.0
  mac_throughput_gops: 256.0

costs:
  dram_access_cost_pj_byte: 640.0
  sram_base_cost_pj_byte: 5.0
  fusion_overhead_cost_pj: 100.0
  area_coefficient_mm2_per_kb: 0.1
  sram_energy_scaling_pj_byte_kb: 0.01

weights:
  penalty_weight: 1000.0
  product_penalty_weight: 500.0

# === NEW: Calibration factors for model tuning ===
calibration:
  latency_fudge_factor: 1.2        # Accounts for pipeline stalls, cache misses
  dram_energy_factor: 1.1          # Calibrate against real DRAM measurements
  sram_energy_factor: 0.9          # Fine-tune SRAM energy model
  compute_energy_factor: 1.0       # MAC operation energy calibration
  noc_energy_factor: 1.15          # Network-on-Chip energy adjustment

# === NEW: NoC (Network-on-Chip) energy parameters ===
noc:
  noc_energy_per_byte_pj: 0.05     # Energy cost per byte moved on NoC

# === NEW: Fusion control overhead parameters ===
fusion:
  fusion_control_overhead_cycles: 50  # Control overhead for coordinating fused ops

# === NEW: Compute energy parameters (made configurable) ===
compute:
  mac_energy_pj: 0.1               # Energy per MAC operation in picojoules

# === NEW: Hardware Design Space Exploration parameters ===
hardware_dse:
  clock_frequency_mhz: 1000.0      # Clock frequency in MHz
  macs_per_pe_per_cycle: 1.0       # MAC operations per PE per clock cycle

# === NEW: Area model coefficients ===
area_model:
  area_per_pe_mm2: 0.01            # Area per processing element in mm²
  area_per_kb_sram_mm2: 0.1        # Area per KB of SRAM in mm²
  area_base_mm2: 1.0               # Base area for control logic in mm²

# ------------------------------------------------------------------
# NEW: Weights for multi-objective loss (can be overridden in YAML)
# ------------------------------------------------------------------
latency_weight: 1.0   # Relative importance of latency
energy_weight: 1.0   # Relative importance of energy
area_weight: 1.0   # Relative importance of silicon area

# ------------------------------------------------------------------
# NEW: Penalty cycles for every fusion control step (fine-tunes latency)
# ------------------------------------------------------------------
fusion_latency_penalty_cycles: 30  # Extra cycles added per fused group
```

Key Improvements:
1. Modular cost calculation functions for better maintainability
2. Calibration factors for tuning against real simulators/hardware
3. NoC energy modeling for on-chip data movement
4. Fusion control overhead acknowledging that fusion isn't "free"
5. Learnable hardware parameters for joint hardware-software co-design
6. Sophisticated area modeling based on PE count and buffer size
7. Hardware constraint penalties ensuring realistic configurations
8. All calculations remain differentiable for PyTorch autograd
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from functools import reduce
from operator import mul

def sanitize_layer_name(layer_name: str) -> str:
    """
    Sanitize layer names to be compatible with PyTorch ModuleDict.
    Replace problematic characters with underscores.
    
    Args:
        layer_name: Original layer name from ONNX
        
    Returns:
        Sanitized layer name safe for PyTorch modules
    """
    # Replace dots, slashes, and other problematic characters with underscores
    sanitized = layer_name.replace('.', '_').replace('/', '_').replace('-', '_')
    # Remove leading/trailing underscores and collapse multiple underscores
    sanitized = '_'.join(filter(None, sanitized.split('_')))
    return sanitized

class Config:
    """
    Global configuration singleton loaded from YAML.
    """
    _instance = None
    
    def __init__(self):
        if Config._instance is not None:
            raise RuntimeError("Config is a singleton - use Config.get_instance()")
        
        with open('config.yaml', 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Extract commonly used values
        self.PROBLEM_DIMS = self._config['dimensions']['loop_dims']
        self.NUM_MEMORY_LEVELS = self._config['hardware']['num_memory_levels']
        self.BYTES_PER_ELEMENT = self._config['hardware']['bytes_per_element']
        
        # Cost model parameters
        self.FUSION_OVERHEAD_COST = self._config['costs']['fusion_overhead_cost_pj']
        self.AREA_COEFFICIENT = self._config['costs']['area_coefficient_mm2_per_kb']
        
        # Note: DRAM_ACCESS_COST, SRAM_BASE_COST, and SRAM_ENERGY_SCALING have been removed
        # as they are now calculated dynamically in the _get_dynamic_epa method
        
        # Hardware parameters
        self.DRAM_BANDWIDTH = self._config['hardware']['dram_bandwidth_gb_s']
        self.SRAM_BANDWIDTH = self._config['hardware']['sram_bandwidth_gb_s']
        self.MAC_THROUGHPUT = self._config['hardware']['mac_throughput_gops']
        
        # Loss weights
        self.PENALTY_WEIGHT = self._config['weights']['penalty_weight']
        self.PRODUCT_PENALTY_WEIGHT = self._config['weights'].get('product_penalty_weight', 1.0)
        
        # === NEW: Calibration factors for model tuning ===
        calibration = self._config.get('calibration', {})
        self.LATENCY_FUDGE_FACTOR = calibration.get('latency_fudge_factor', 1.0)
        self.DRAM_ENERGY_FACTOR = calibration.get('dram_energy_factor', 1.0)
        self.SRAM_ENERGY_FACTOR = calibration.get('sram_energy_factor', 1.0)
        self.COMPUTE_ENERGY_FACTOR = calibration.get('compute_energy_factor', 1.0)
        self.NOC_ENERGY_FACTOR = calibration.get('noc_energy_factor', 1.0)
        
        # === NEW: NoC (Network-on-Chip) energy parameters ===
        noc_config = self._config.get('noc', {})
        self.NOC_ENERGY_PER_BYTE_PJ = noc_config.get('noc_energy_per_byte_pj', 0.05)
        
        # === NEW: Fusion control overhead parameters ===
        fusion_config = self._config.get('fusion', {})
        self.FUSION_CONTROL_OVERHEAD_CYCLES = fusion_config.get('fusion_control_overhead_cycles', 50)
        
        # === NEW: Compute energy per MAC operation (made configurable) ===
        compute_config = self._config.get('compute', {})
        self.MAC_ENERGY_PJ = compute_config.get('mac_energy_pj', 0.1)
        
        # === NEW: Hardware design space exploration parameters ===
        hw_dse_config = self._config.get('hardware_dse', {})
        self.CLOCK_FREQUENCY_MHZ = hw_dse_config.get('clock_frequency_mhz', 1000.0)
        self.MACS_PER_PE_PER_CYCLE = hw_dse_config.get('macs_per_pe_per_cycle', 1.0)
        
        # === NEW: Area model coefficients ===
        area_model_config = self._config.get('area_model', {})
        self.AREA_PER_PE_MM2 = area_model_config.get('area_per_pe_mm2', 0.01)
        self.AREA_PER_KB_SRAM_MM2 = area_model_config.get('area_per_kb_sram_mm2', 0.1)
        self.AREA_BASE_MM2 = area_model_config.get('area_base_mm2', 1.0)  # Base area for control logic

        # --- FIX: Add configurable weights for the loss function ---
        # Allows us to tune the relative importance of different objectives.
        self.LATENCY_WEIGHT = self._config.get('weights', {}).get('latency_weight', 1.0)
        self.ENERGY_WEIGHT = self._config.get('weights', {}).get('energy_weight', 1.0)
        self.AREA_WEIGHT = self._config.get('weights', {}).get('area_weight', 1.0)
        
        # --- FIX: Add configurable latency penalty for fusion operations ---
        # Models the real-world control overhead of fusing layers.
        self.FUSION_LATENCY_PENALTY_CYCLES = self._config.get('fusion', {}).get('fusion_latency_penalty_cycles', 30)
    
    @staticmethod
    def get_instance():
        """Get the singleton instance of Config."""
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance
    
    def get_weight(self, weight_name: str, default_value: float) -> float:
        """
        Get a weight value from the configuration, with a default fallback.
        Args:
            weight_name: Name of the weight to retrieve.
            default_value: The default value to return if the weight is not found.
        
        Returns:
            The weight value as a float.
        """
        return self._config.get('weights', {}).get(weight_name, default_value)

def find_divisors(n: int) -> List[int]:
    """
    Find all positive divisors of n.
    
    Args:
        n: Integer to find divisors for
        
    Returns:
        Sorted list of all positive divisors
    """
    divisors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:  # Avoid duplicates for perfect squares
                divisors.append(n // i)
    return sorted(divisors)

def find_nearest_divisor(n: int, target: float) -> int:
    """
    Find the divisor of n that is closest to target.
    
    Args:
        n: Integer to find divisors for
        target: Target value to approximate
        
    Returns:
        Divisor of n closest to target
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    # Handle edge cases
    if target <= 0:
        return 1
    if target >= n:
        return n
        
    divisors = find_divisors(n)
    return min(divisors, key=lambda x: abs(x - target))

def calculate_penalty_loss(mapping_params: 'MappingParameters') -> torch.Tensor:
    """
    重构后的惩罚函数：适用于基于模板的参数结构。
    
    惩罚所有模板内部的 tiling 因子小于1的情况，因为这在物理上是不可能的。
    现在需要遍历所有 MappingDecisionModule 中的所有模板实例。
    
    Args:
        mapping_params: 映射策略管理器，包含所有决策模块
        
    Returns:
        总惩罚值，累计所有无效的 tiling 因子
    """
    total_penalty = torch.tensor(0.0, dtype=torch.float32)
    
    # 遍历所有决策模块
    all_decision_modules = mapping_params.get_all_decision_modules()
    
    for group_key, decision_module in all_decision_modules.items():
        # 遍历该决策模块中的所有模板实例
        for template_name, template_instance in decision_module.templates.items():
            # 检查该模板的所有 tiling 因子参数
            if hasattr(template_instance, 'get_M0'):
                m0 = template_instance.get_M0()
                if m0 < 1.0:
                    penalty = (1.0 - m0) ** 2
                    total_penalty = total_penalty + penalty
            
            if hasattr(template_instance, 'get_K0'):
                k0 = template_instance.get_K0()
                # 注意：对于某些模板，K0 可能等于 K（固定值），不需要惩罚
                # 只惩罚可学习的 K0 参数
                if hasattr(template_instance, 'log_K0') and k0 < 1.0:
                    penalty = (1.0 - k0) ** 2
                    total_penalty = total_penalty + penalty
            
            if hasattr(template_instance, 'get_N0'):
                n0 = template_instance.get_N0()
                # 注意：对于某些模板，N0 可能等于 N（固定值），不需要惩罚
                # 只惩罚可学习的 N0 参数
                if hasattr(template_instance, 'log_N0') and n0 < 1.0:
                    penalty = (1.0 - n0) ** 2
                    total_penalty = total_penalty + penalty
    
    return total_penalty


def calculate_template_constraint_penalty(
    mapping_params: 'MappingParameters',
    hardware_params: 'HardwareParameters',
    graph: 'ComputationGraph'
) -> torch.Tensor:
    """
    新的模板约束惩罚函数：实现 Buffer 容量约束。
    
    关键约束：
    1. Buffer 容量约束：对于任何计算组在任何模板下，计算出的总 Buffer 需求
       不能超过硬件提供的总 Buffer 容量
    2. Tiling 因子有效性：已在 calculate_penalty_loss 中处理
    
    Args:
        mapping_params: 映射策略管理器
        hardware_params: 硬件参数
        graph: 计算图
        
    Returns:
        总约束惩罚值
    """
    total_penalty = torch.tensor(0.0, dtype=torch.float32)
    available_buffer_bytes = hardware_params.get_buffer_size_bytes()
    
    # 获取所有决策模块
    all_decision_modules = mapping_params.get_all_decision_modules()
    group_mapping = mapping_params.group_mapping
    
    # === 遍历所有计算组，检查每个模板的 Buffer 容量约束 ===
    for group_key, decision_module in all_decision_modules.items():
        group_layers = group_mapping[group_key]
        
        # 对该组的每个可能模板检查约束
        for template_name, template_instance in decision_module.templates.items():
            
            # 计算该模板在该组下的 Buffer 需求
            max_buffer_requirements = {'input': torch.tensor(0.0), 'weight': torch.tensor(0.0), 'output': torch.tensor(0.0)}
            
            for layer_name in group_layers:
                # 使用模板计算该层的缓冲区需求
                layer_buffer_reqs = template_instance.get_buffer_requirements()
                
                # 取各张量的最大缓冲区需求（遵循融合链的 max_e(BufReq_e) 规则）
                for tensor_type in ['input', 'weight', 'output']:
                    if tensor_type in layer_buffer_reqs:
                        max_buffer_requirements[tensor_type] = torch.maximum(
                            max_buffer_requirements[tensor_type],
                            layer_buffer_reqs[tensor_type]
                        )
            
            # 计算总 Buffer 需求（以 bytes 为单位）
            config = Config.get_instance()
            total_buffer_requirement_bytes = torch.tensor(0.0, dtype=torch.float32)
            
            for tensor_type, buffer_req_words in max_buffer_requirements.items():
                buffer_req_bytes = buffer_req_words * config.BYTES_PER_ELEMENT
                total_buffer_requirement_bytes = total_buffer_requirement_bytes + buffer_req_bytes
            
            # === 检查 Buffer 容量约束 ===
            if total_buffer_requirement_bytes > available_buffer_bytes:
                # 惩罚：超出量的平方
                excess = total_buffer_requirement_bytes - available_buffer_bytes
                penalty = (excess / available_buffer_bytes) ** 2  # 归一化惩罚
                total_penalty = total_penalty + penalty
    
    # === 硬件参数的合理性约束 ===
    num_pes = hardware_params.get_num_pes()
    buffer_size_kb = hardware_params.get_buffer_size_kb()
    
    # 惩罚不合理的硬件配置
    if num_pes < 1.0:
        total_penalty = total_penalty + (1.0 - num_pes) ** 2
    
    if num_pes > 1024.0:  # 超过 1024 个 PE 不现实
        total_penalty = total_penalty + (num_pes - 1024.0) ** 2
    
    if buffer_size_kb < 1.0:  # 小于 1 KB 不现实
        total_penalty = total_penalty + (1.0 - buffer_size_kb) ** 2
    
    if buffer_size_kb > 10240.0:  # 超过 10 MB 不现实
        total_penalty = total_penalty + (buffer_size_kb - 10240.0) ** 2
    
    return total_penalty

def calculate_hardware_constraint_penalty(
    mapping_params: 'MappingParameters',
    hardware_params: 'HardwareParameters',
    graph: 'ComputationGraph'
) -> torch.Tensor:
    """
    Calculate penalty for hardware configurations that violate physical constraints.
    Ensures that the required buffer size for any mapping doesn't exceed available hardware buffer.
    
    Args:
        mapping_params: MappingParameters object containing layer mappings
        hardware_params: HardwareParameters object containing hardware config
        graph: ComputationGraph object defining the network structure
        
    Returns:
        Total penalty for constraint violations
    """
    total_penalty = torch.tensor(0.0)
    available_buffer_bytes = hardware_params.get_buffer_size_bytes()
    
    # Check buffer constraints for individual layers
    for layer_name, layer_info in graph.layers.items():
        layer_mapping = mapping_params.get_mapping_by_original_name(layer_name)
        buffer_reqs = layer_mapping.get_buffer_requirements()
        required_buffer_bytes = sum(buffer_reqs.values())
        
        # Penalty if required buffer exceeds available buffer
        if required_buffer_bytes > available_buffer_bytes:
            penalty = (required_buffer_bytes - available_buffer_bytes) ** 2
            total_penalty = total_penalty + penalty
    
    # Check buffer constraints for fusion groups
    for group in graph.fusion_groups:
        group_buffer_req = calculate_fused_group_buffer_req(group, mapping_params, graph)
        
        # Penalty if fused group buffer requirement exceeds available buffer
        if group_buffer_req > available_buffer_bytes:
            penalty = (group_buffer_req - available_buffer_bytes) ** 2
            total_penalty = total_penalty + penalty
    
    # Add penalties for unrealistic hardware configurations
    num_pes = hardware_params.get_num_pes()
    buffer_size_kb = hardware_params.get_buffer_size_kb()
    
    # Penalty for too few PEs (less than 1)
    if num_pes < 1.0:
        total_penalty = total_penalty + (1.0 - num_pes) ** 2
    
    # Penalty for too many PEs (more than 1024, which is unrealistic for most accelerators)
    if num_pes > 1024.0:
        total_penalty = total_penalty + (num_pes - 1024.0) ** 2
    
    # Penalty for too small buffer (less than 1 KB)
    if buffer_size_kb < 1.0:
        total_penalty = total_penalty + (1.0 - buffer_size_kb) ** 2
    
    # Penalty for too large buffer (more than 10 MB, which is unrealistic for on-chip SRAM)
    if buffer_size_kb > 10240.0:  # 10 MB
        total_penalty = total_penalty + (buffer_size_kb - 10240.0) ** 2
    
    return total_penalty

def round_tiling_factors(factor_tensor: torch.Tensor, problem_size: int) -> torch.Tensor:
    """
    Round continuous tiling factors to nearest valid divisors.
    This is a non-differentiable operation used only during evaluation.
    
    Args:
        factor_tensor: Continuous tiling factors from optimizer
        problem_size: Problem dimension size to find divisors for
        
    Returns:
        Rounded factors that are valid divisors
    """
    # Convert to numpy for integer operations
    continuous_factor = factor_tensor.detach().cpu().numpy()
    rounded_factor = find_nearest_divisor(problem_size, continuous_factor)
    return torch.tensor(rounded_factor, dtype=torch.float32)

def calculate_macs(dims: Dict[str, int], op_type: str = 'Conv') -> torch.Tensor:
    """
    Calculate total number of multiply-accumulate operations.
    Handles both CNN (convolution) and Transformer (matrix multiplication) operations.
    
    Args:
        dims: Dictionary of problem dimensions
        op_type: Operation type to determine MAC calculation method
        
    Returns:
        Total number of MAC operations
    """
    # === CNN OPERATORS ===
    if op_type == 'Conv':
        # Convolution: N * K * P * Q * C * R * S
        if 'R' in dims and 'S' in dims:
            return torch.tensor(
                dims['N'] * dims['K'] * dims['P'] * dims['Q'] * dims['C'] * dims['R'] * dims['S'],
                dtype=torch.float32
            )
        else:
            # Fallback for malformed conv layers
            return torch.tensor(
                dims['N'] * dims['K'] * dims['P'] * dims['Q'] * dims.get('C', dims['K']),
                dtype=torch.float32
            )
    
    # === TRANSFORMER OPERATORS ===
    elif op_type in ['MatMul', 'Gemm']:
        # Matrix multiplication: N * SeqLen * C_in * K_out
        # For Transformers: (N, P, C) × (C, K) -> (N, P, K)
        # Total MACs = N * P * C * K
        seq_len = dims.get('P', 1)  # Sequence length (mapped to P)
        c_in = dims.get('C', dims['K'])  # Input features
        k_out = dims['K']  # Output features
        
        return torch.tensor(
            dims['N'] * seq_len * c_in * k_out,
            dtype=torch.float32
        )
    
    elif op_type == 'LayerNormalization':
        # Layer normalization: N * SeqLen * Features (element-wise operations)
        # Each element requires: mean calculation, variance calculation, normalization
        # Approximate as 5 operations per element (mean, var, subtract, divide, scale/shift)
        seq_len = dims.get('P', 1)
        features = dims['K']
        
        return torch.tensor(
            dims['N'] * seq_len * features * 5,  # 5 ops per element
            dtype=torch.float32
        )
    
    elif op_type == 'Attention':
        # Multi-head attention (if present as fused operator)
        # Simplified: 3 matrix mults (Q, K, V) + attention computation + output projection
        # Q×K^T: N * P * C * P, Softmax: N * P * P, (Q×K^T)×V: N * P * P * C, Proj: N * P * C * K
        seq_len = dims.get('P', 1)
        features = dims['K']
        
        qk_macs = dims['N'] * seq_len * features * seq_len  # Q×K^T
        attention_macs = dims['N'] * seq_len * seq_len * features  # Attention×V
        proj_macs = dims['N'] * seq_len * features * features  # Output projection
        
        return torch.tensor(qk_macs + attention_macs + proj_macs, dtype=torch.float32)
    
    # === ELEMENT-WISE OPERATORS ===
    elif op_type in ['Relu', 'Gelu', 'Sigmoid', 'Tanh', 'Add', 'Mul', 'Softmax', 'Erf']:
        # Element-wise operations: minimal computational cost
        # For Transformers: N * SeqLen * Features
        # For CNNs: N * K * P * Q
        if 'P' in dims and dims.get('P', 1) > 1 and 'Q' in dims and dims.get('Q', 1) == 1:
            # Transformer format: (N, SeqLen, Features)
            return torch.tensor(
                dims['N'] * dims['P'] * dims['K'],
                dtype=torch.float32
            )
        else:
            # CNN format: (N, K, P, Q)
            return torch.tensor(
                dims['N'] * dims['K'] * dims.get('P', 1) * dims.get('Q', 1),
                dtype=torch.float32
            )
    
    # === POOLING AND OTHER OPERATORS ===
    elif op_type in ['MaxPool', 'AveragePool', 'GlobalAveragePool']:
        # Pooling operations: relatively low computational cost
        return torch.tensor(
            dims['N'] * dims['K'] * dims.get('P', 1) * dims.get('Q', 1),
            dtype=torch.float32
        )
    
    elif op_type in ['BatchNormalization']:
        # Batch normalization: similar to element-wise operations
        return torch.tensor(
            dims['N'] * dims['K'] * dims.get('P', 1) * dims.get('Q', 1) * 2,  # 2 ops per element
            dtype=torch.float32
        )
    
    # === DEFAULT CASE ===
    else:
        # Default to element-wise operation cost
        if 'P' in dims and dims.get('P', 1) > 1 and 'Q' in dims and dims.get('Q', 1) == 1:
            # Likely Transformer format
            return torch.tensor(
                dims['N'] * dims['P'] * dims['K'],
                dtype=torch.float32
            )
        else:
            # Likely CNN format or unknown
            return torch.tensor(
                dims['N'] * dims['K'] * dims.get('P', 1) * dims.get('Q', 1),
                dtype=torch.float32
            )

# ============================
# 可微映射模板 (Differentiable Mapping Templates, DMT)
# ============================

class DifferentiableMappingTemplate(nn.Module):
    """
    可微映射模板的抽象基类。
    
    映射模板定义了算子（或融合算子组）在硬件上执行时的数据排布和计算顺序的
    结构化、模式化描述。它将无限的映射可能性约束在几个有明确物理意义的、
    可分析的模式中。
    
    Args:
        dims: 层的维度信息，包含 M, K, N 等维度
        template_name: 模板名称，用于标识
    """
    
    def __init__(self, dims: Dict[str, int], template_name: str):
        super().__init__()
        self.dims = dims
        self.template_name = template_name
        self.config = Config.get_instance()
        
        # 每个模板都有自己的可学习参数
        self._init_template_parameters()
    
    def _init_template_parameters(self):
        """初始化模板特定的可学习参数。子类需要重写此方法。"""
        raise NotImplementedError("Subclasses must implement _init_template_parameters")
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        模板的前向传播，计算性能指标。
        
        Returns:
            Dict containing latency, energy, buffer_requirements, etc.
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    def get_buffer_requirements(self) -> Dict[str, torch.Tensor]:
        """计算此模板的缓冲区需求。"""
        raise NotImplementedError("Subclasses must implement get_buffer_requirements")
    
    def get_access_counts(self) -> Dict[str, torch.Tensor]:
        """计算此模板的访存次数。"""
        raise NotImplementedError("Subclasses must implement get_access_counts")


class DMT_GEMM_Full(DifferentiableMappingTemplate):
    """
    GEMM Full 映射模板 (FFMT-Full)
    
    只对 M 维度进行切分，一次性读入完整的输入行，并产生完整的输出行。
    对 Buffer 的 K 和 N 维度占用最大。
    
    在这个模板中：K0=K, N0=N (完整维度), 只有 M0 是可学习的切分参数
    """
    
    def __init__(self, dims: Dict[str, int]):
        super().__init__(dims, "GEMM_Full")
    
    def _init_template_parameters(self):
        """
        初始化 Full 模板的参数：M0 (M维度的切分因子)
        
        Full 模式特点：K0=K, N0=N，只对 M 维度进行切分
        因此 M1 = M/M0, K1=1, N1=1
        """
        M = self.dims.get('M', self.dims.get('N', 1))  # 兼容不同的命名约定
        K = self.dims.get('K', self.dims.get('C', 1))
        N = self.dims.get('N', self.dims.get('P', self.dims.get('Q', 1)))
        
        # 使用对数尺度参数化，确保值始终为正
        # 初始化为维度的平方根，这是一个合理的起始点
        epsilon = 1e-6
        init_m0 = max(1.0, min(M, np.sqrt(M)))
        
        self.log_M0 = nn.Parameter(torch.log(torch.tensor(init_m0 + epsilon, dtype=torch.float32)))
        
        # 存储固定的维度值（不可学习）
        self.register_buffer('M_total', torch.tensor(M, dtype=torch.float32))
        self.register_buffer('K_total', torch.tensor(K, dtype=torch.float32))
        self.register_buffer('N_total', torch.tensor(N, dtype=torch.float32))
    
    def get_M0(self) -> torch.Tensor:
        """获取 M 维度的切分因子"""
        return torch.exp(self.log_M0)
    
    def get_K0(self) -> torch.Tensor:
        """Full 模式：K0 = K (完整维度)"""
        return self.K_total
    
    def get_N0(self) -> torch.Tensor:
        """Full 模式：N0 = N (完整维度)"""
        return self.N_total
    
    def get_M1(self) -> torch.Tensor:
        """计算 M1 = M / M0"""
        return self.M_total / self.get_M0()
    
    def get_K1(self) -> torch.Tensor:
        """Full 模式：K1 = 1"""
        return torch.tensor(1.0, dtype=torch.float32)
    
    def get_N1(self) -> torch.Tensor:
        """Full 模式：N1 = 1"""
        return torch.tensor(1.0, dtype=torch.float32)
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """计算 Full 模板的性能指标"""
        buffer_reqs = self.get_buffer_requirements()
        access_counts = self.get_access_counts()
        
        return {
            'buffer_requirements': buffer_reqs,
            'access_counts': access_counts,
            'template_name': self.template_name
        }
    
    def get_buffer_requirements(self) -> Dict[str, torch.Tensor]:
        """
        Full 模板的缓冲区需求计算
        
        基于论文公式：
        - BufReq_I = M0 * K0 = M0 * K
        - BufReq_W = K0 * N0 = K * N  
        - BufReq_O = M0 * N0 = M0 * N
        """
        M0 = self.get_M0()
        K0 = self.get_K0()  # = K
        N0 = self.get_N0()  # = N
        
        return {
            'input': M0 * K0,      # M0 * K
            'weight': K0 * N0,     # K * N (最大)
            'output': M0 * N0      # M0 * N
        }
    
    def get_access_counts(self) -> Dict[str, torch.Tensor]:
        """
        Full 模板的 DRAM 访问次数计算
        
        基于论文公式：
        - Access_I = M * K * N1 = M * K * 1 = M * K
        - Access_W = K * N * M1 = K * N * (M/M0)
        - Access_O = M * N
        """
        M1 = self.get_M1()  # M / M0
        
        return {
            'input_dram': self.M_total * self.K_total,                    # M * K
            'input_sram': torch.tensor(0.0, dtype=torch.float32),         # Full模板不使用SRAM缓存输入
            'weight_dram': self.K_total * self.N_total * M1,              # K * N * M1  
            'weight_sram': torch.tensor(0.0, dtype=torch.float32),        # Full模板不使用SRAM缓存权重
            'output_dram': self.M_total * self.N_total,                   # M * N
            'output_sram': torch.tensor(0.0, dtype=torch.float32),        # Full模板不使用SRAM缓存输出
            # 兼容性保持
            'input': self.M_total * self.K_total,
            'weight': self.K_total * self.N_total * M1,
            'output': self.M_total * self.N_total
        }


class DMT_GEMM_TiledK(DifferentiableMappingTemplate):
    """
    GEMM TiledK 映射模板 (FFMT-TiledK)
    
    对 M 和 K 维度进行切分。只消耗输入行的子集（sub-partition），
    并产生部分和（partial sums）。
    
    在这个模板中：N0=N (完整维度), M0 和 K0 是可学习的切分参数
    """
    
    def __init__(self, dims: Dict[str, int]):
        super().__init__(dims, "GEMM_TiledK")
    
    def _init_template_parameters(self):
        """
        初始化 TiledK 模板的参数：M0, K0 (M和K维度的切分因子)
        
        TiledK 模式特点：N0=N，对 M 和 K 维度进行切分
        因此 M1 = M/M0, K1 = K/K0, N1=1
        """
        M = self.dims.get('M', self.dims.get('N', 1))
        K = self.dims.get('K', self.dims.get('C', 1))
        N = self.dims.get('N', self.dims.get('P', self.dims.get('Q', 1)))
        
        # 使用对数尺度参数化，确保值始终为正
        epsilon = 1e-6
        init_m0 = max(1.0, min(M, np.sqrt(M)))
        init_k0 = max(1.0, min(K, np.sqrt(K)))
        
        self.log_M0 = nn.Parameter(torch.log(torch.tensor(init_m0 + epsilon, dtype=torch.float32)))
        self.log_K0 = nn.Parameter(torch.log(torch.tensor(init_k0 + epsilon, dtype=torch.float32)))
        
        # 存储固定的维度值（不可学习）
        self.register_buffer('M_total', torch.tensor(M, dtype=torch.float32))
        self.register_buffer('K_total', torch.tensor(K, dtype=torch.float32))
        self.register_buffer('N_total', torch.tensor(N, dtype=torch.float32))
    
    def get_M0(self) -> torch.Tensor:
        """获取 M 维度的切分因子"""
        return torch.exp(self.log_M0)
    
    def get_K0(self) -> torch.Tensor:
        """获取 K 维度的切分因子"""
        return torch.exp(self.log_K0)
    
    def get_N0(self) -> torch.Tensor:
        """TiledK 模式：N0 = N (完整维度)"""
        return self.N_total
    
    def get_M1(self) -> torch.Tensor:
        """计算 M1 = M / M0"""
        return self.M_total / self.get_M0()
    
    def get_K1(self) -> torch.Tensor:
        """计算 K1 = K / K0"""
        return self.K_total / self.get_K0()
    
    def get_N1(self) -> torch.Tensor:
        """TiledK 模式：N1 = 1"""
        return torch.tensor(1.0, dtype=torch.float32)
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """计算 TiledK 模板的性能指标"""
        buffer_reqs = self.get_buffer_requirements()
        access_counts = self.get_access_counts()
        
        return {
            'buffer_requirements': buffer_reqs,
            'access_counts': access_counts,
            'template_name': self.template_name
        }
    
    def get_buffer_requirements(self) -> Dict[str, torch.Tensor]:
        """
        TiledK 模板的缓冲区需求计算
        
        基于论文公式：
        - BufReq_I = M0 * K0
        - BufReq_W = K0 * N0 = K0 * N
        - BufReq_O = M0 * N0 = M0 * N
        """
        M0 = self.get_M0()
        K0 = self.get_K0()
        N0 = self.get_N0()  # = N
        
        return {
            'input': M0 * K0,       # M0 * K0
            'weight': K0 * N0,      # K0 * N
            'output': M0 * N0       # M0 * N
        }
    
    def get_access_counts(self) -> Dict[str, torch.Tensor]:
        """
        TiledK 模板的 DRAM 访问次数计算
        
        基于论文公式：
        - Access_I = M * K * N1 = M * K * 1 = M * K
        - Access_W = K * N * M1 = K * N * (M/M0)
        - Access_O = M * N
        """
        M1 = self.get_M1()  # M / M0
        
        return {
            'input_dram': self.M_total * self.K_total,                    # M * K
            'input_sram': torch.tensor(0.0, dtype=torch.float32),         # TiledK模板不使用SRAM缓存输入
            'weight_dram': self.K_total * self.N_total * M1,              # K * N * M1
            'weight_sram': torch.tensor(0.0, dtype=torch.float32),        # TiledK模板不使用SRAM缓存权重
            'output_dram': self.M_total * self.N_total,                   # M * N
            'output_sram': torch.tensor(0.0, dtype=torch.float32),        # TiledK模板不使用SRAM缓存输出
            # 兼容性保持
            'input': self.M_total * self.K_total,
            'weight': self.K_total * self.N_total * M1,
            'output': self.M_total * self.N_total
        }


class DMT_GEMM_TiledN(DifferentiableMappingTemplate):
    """
    GEMM TiledN 映射模板 (FFMT-TiledN)
    
    对 M 和 N 维度进行切分。消耗完整的输入行，但只产生输出行的子集。
    
    在这个模板中：K0=K (完整维度), M0 和 N0 是可学习的切分参数
    """
    
    def __init__(self, dims: Dict[str, int]):
        super().__init__(dims, "GEMM_TiledN")
    
    def _init_template_parameters(self):
        """
        初始化 TiledN 模板的参数：M0, N0 (M和N维度的切分因子)
        
        TiledN 模式特点：K0=K，对 M 和 N 维度进行切分
        因此 M1 = M/M0, K1=1, N1 = N/N0
        """
        M = self.dims.get('M', self.dims.get('N', 1))
        K = self.dims.get('K', self.dims.get('C', 1))
        N = self.dims.get('N', self.dims.get('P', self.dims.get('Q', 1)))
        
        # 使用对数尺度参数化，确保值始终为正
        epsilon = 1e-6
        init_m0 = max(1.0, min(M, np.sqrt(M)))
        init_n0 = max(1.0, min(N, np.sqrt(N)))
        
        self.log_M0 = nn.Parameter(torch.log(torch.tensor(init_m0 + epsilon, dtype=torch.float32)))
        self.log_N0 = nn.Parameter(torch.log(torch.tensor(init_n0 + epsilon, dtype=torch.float32)))
        
        # 存储固定的维度值（不可学习）
        self.register_buffer('M_total', torch.tensor(M, dtype=torch.float32))
        self.register_buffer('K_total', torch.tensor(K, dtype=torch.float32))
        self.register_buffer('N_total', torch.tensor(N, dtype=torch.float32))
    
    def get_M0(self) -> torch.Tensor:
        """获取 M 维度的切分因子"""
        return torch.exp(self.log_M0)
    
    def get_K0(self) -> torch.Tensor:
        """TiledN 模式：K0 = K (完整维度)"""
        return self.K_total
    
    def get_N0(self) -> torch.Tensor:
        """获取 N 维度的切分因子"""
        return torch.exp(self.log_N0)
    
    def get_M1(self) -> torch.Tensor:
        """计算 M1 = M / M0"""
        return self.M_total / self.get_M0()
    
    def get_K1(self) -> torch.Tensor:
        """TiledN 模式：K1 = 1"""
        return torch.tensor(1.0, dtype=torch.float32)
    
    def get_N1(self) -> torch.Tensor:
        """计算 N1 = N / N0"""
        return self.N_total / self.get_N0()
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """计算 TiledN 模板的性能指标"""
        buffer_reqs = self.get_buffer_requirements()
        access_counts = self.get_access_counts()
        
        return {
            'buffer_requirements': buffer_reqs,
            'access_counts': access_counts,
            'template_name': self.template_name
        }
    
    def get_buffer_requirements(self) -> Dict[str, torch.Tensor]:
        """
        TiledN 模板的缓冲区需求计算
        
        基于论文公式：
        - BufReq_I = M0 * K0 = M0 * K
        - BufReq_W = K0 * N0 = K * N0
        - BufReq_O = M0 * N0
        """
        M0 = self.get_M0()
        K0 = self.get_K0()  # = K
        N0 = self.get_N0()
        
        return {
            'input': M0 * K0,       # M0 * K
            'weight': K0 * N0,      # K * N0
            'output': M0 * N0       # M0 * N0
        }
    
    def get_access_counts(self) -> Dict[str, torch.Tensor]:
        """
        TiledN 模板的 DRAM 访问次数计算
        
        基于论文公式：
        - Access_I = M * K * N1 = M * K * (N/N0)
        - Access_W = K * N * M1 = K * N * (M/M0)
        - Access_O = M * N
        """
        M1 = self.get_M1()  # M / M0
        N1 = self.get_N1()  # N / N0
        
        return {
            'input_dram': self.M_total * self.K_total * N1,               # M * K * N1
            'input_sram': torch.tensor(0.0, dtype=torch.float32),         # TiledN模板不使用SRAM缓存输入
            'weight_dram': self.K_total * self.N_total * M1,              # K * N * M1
            'weight_sram': torch.tensor(0.0, dtype=torch.float32),        # TiledN模板不使用SRAM缓存权重
            'output_dram': self.M_total * self.N_total,                   # M * N
            'output_sram': torch.tensor(0.0, dtype=torch.float32),        # TiledN模板不使用SRAM缓存输出
            # 兼容性保持
            'input': self.M_total * self.K_total * N1,
            'weight': self.K_total * self.N_total * M1,
            'output': self.M_total * self.N_total
        }


class DMT_GEMM_TiledKN(DifferentiableMappingTemplate):
    """
    GEMM TiledKN 映射模板 (FFMT-TiledKN)
    
    对 M, K, N 三个维度都进行切分。消耗输入行的子集，也产生部分和的输出子集。
    这是最灵活、对 Buffer 要求最低的模式。
    
    在这个模板中：M0, K0, N0 都是可学习的切分参数
    """
    
    def __init__(self, dims: Dict[str, int]):
        super().__init__(dims, "GEMM_TiledKN")
    
    def _init_template_parameters(self):
        """
        初始化 TiledKN 模板的参数：M0, K0, N0 (M, K, N三个维度的切分因子)
        
        TiledKN 模式特点：对所有维度进行切分
        因此 M1 = M/M0, K1 = K/K0, N1 = N/N0
        """
        M = self.dims.get('M', self.dims.get('N', 1))
        K = self.dims.get('K', self.dims.get('C', 1))
        N = self.dims.get('N', self.dims.get('P', self.dims.get('Q', 1)))
        
        # 使用对数尺度参数化，确保值始终为正
        epsilon = 1e-6
        init_m0 = max(1.0, min(M, np.sqrt(M)))
        init_k0 = max(1.0, min(K, np.sqrt(K)))
        init_n0 = max(1.0, min(N, np.sqrt(N)))
        
        self.log_M0 = nn.Parameter(torch.log(torch.tensor(init_m0 + epsilon, dtype=torch.float32)))
        self.log_K0 = nn.Parameter(torch.log(torch.tensor(init_k0 + epsilon, dtype=torch.float32)))
        self.log_N0 = nn.Parameter(torch.log(torch.tensor(init_n0 + epsilon, dtype=torch.float32)))
        
        # 存储固定的维度值（不可学习）
        self.register_buffer('M_total', torch.tensor(M, dtype=torch.float32))
        self.register_buffer('K_total', torch.tensor(K, dtype=torch.float32))
        self.register_buffer('N_total', torch.tensor(N, dtype=torch.float32))
    
    def get_M0(self) -> torch.Tensor:
        """获取 M 维度的切分因子"""
        return torch.exp(self.log_M0)
    
    def get_K0(self) -> torch.Tensor:
        """获取 K 维度的切分因子"""
        return torch.exp(self.log_K0)
    
    def get_N0(self) -> torch.Tensor:
        """获取 N 维度的切分因子"""
        return torch.exp(self.log_N0)
    
    def get_M1(self) -> torch.Tensor:
        """计算 M1 = M / M0"""
        return self.M_total / self.get_M0()
    
    def get_K1(self) -> torch.Tensor:
        """计算 K1 = K / K0"""
        return self.K_total / self.get_K0()
    
    def get_N1(self) -> torch.Tensor:
        """计算 N1 = N / N0"""
        return self.N_total / self.get_N0()
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """计算 TiledKN 模板的性能指标"""
        buffer_reqs = self.get_buffer_requirements()
        access_counts = self.get_access_counts()
        
        return {
            'buffer_requirements': buffer_reqs,
            'access_counts': access_counts,
            'template_name': self.template_name
        }
    
    def get_buffer_requirements(self) -> Dict[str, torch.Tensor]:
        """
        TiledKN 模板的缓冲区需求计算 (最低缓冲区需求)
        
        基于论文公式：
        - BufReq_I = M0 * K0
        - BufReq_W = K0 * N0
        - BufReq_O = M0 * N0
        """
        M0 = self.get_M0()
        K0 = self.get_K0()
        N0 = self.get_N0()
        
        return {
            'input': M0 * K0,       # M0 * K0
            'weight': K0 * N0,      # K0 * N0
            'output': M0 * N0       # M0 * N0
        }
    
    def get_access_counts(self) -> Dict[str, torch.Tensor]:
        """
        TiledKN 模板的 DRAM 访问次数计算
        
        基于论文公式：
        - Access_I = M * K * N1 = M * K * (N/N0)
        - Access_W = K * N * M1 = K * N * (M/M0)
        - Access_O = M * N
        """
        M1 = self.get_M1()  # M / M0
        N1 = self.get_N1()  # N / N0
        
        return {
            'input_dram': self.M_total * self.K_total * N1,               # M * K * N1
            'input_sram': torch.tensor(0.0, dtype=torch.float32),         # TiledKN模板不使用SRAM缓存输入
            'weight_dram': self.K_total * self.N_total * M1,              # K * N * M1
            'weight_sram': torch.tensor(0.0, dtype=torch.float32),        # TiledKN模板不使用SRAM缓存权重
            'output_dram': self.M_total * self.N_total,                   # M * N
            'output_sram': torch.tensor(0.0, dtype=torch.float32),        # TiledKN模板不使用SRAM缓存输出
            # 兼容性保持
            'input': self.M_total * self.K_total * N1,
            'weight': self.K_total * self.N_total * M1,
            'output': self.M_total * self.N_total
        }


class MappingDecisionModule(nn.Module):
    """
    映射决策模块 (Mapping Decision Module)
    
    为一个特定的计算组（层或融合组）管理映射模板的选择和参数。
    包含所有可能的映射模板实例以及选择这些模板的可学习 logits。
    
    Args:
        dims: 该计算组的维度信息
        group_name: 计算组的名称（用于调试和日志）
    """
    
    def __init__(self, dims: Dict[str, int], group_name: str):
        super().__init__()
        
        self.dims = dims
        self.group_name = group_name
        
        # === 模板选择参数 ===
        # 使用可学习的 logits，通过 softmax 得到选择概率
        # logits 的维度等于可选模板数量（当前是4个 GEMM 模板）
        self.template_selection_logits = nn.Parameter(
            torch.randn(4, dtype=torch.float32)  # [Full, TiledK, TiledN, TiledKN]
        )
        
        # === 模板实例 ===
        # 为每个可能的映射模板创建实例，每个模板都有独立的可学习参数
        self.templates = nn.ModuleDict({
            'full': DMT_GEMM_Full(dims),
            'tiled_k': DMT_GEMM_TiledK(dims),
            'tiled_n': DMT_GEMM_TiledN(dims),
            'tiled_kn': DMT_GEMM_TiledKN(dims)
        })
        
        # 模板名称列表，用于索引
        self.template_names = ['full', 'tiled_k', 'tiled_n', 'tiled_kn']
    
    def get_template_probabilities(self) -> torch.Tensor:
        """
        获取当前的模板选择概率分布。
        
        Returns:
            四种模板的选择概率 [P(Full), P(TiledK), P(TiledN), P(TiledKN)]
        """
        return torch.softmax(self.template_selection_logits, dim=0)
    
    def get_selected_template(self) -> DifferentiableMappingTemplate:
        """
        根据当前学习到的概率分布，选择最佳映射模板。
        
        Returns:
            选中的映射模板实例
        """
        probabilities = self.get_template_probabilities()
        selected_idx = torch.argmax(probabilities).item()
        selected_template_name = self.template_names[selected_idx]
        return self.templates[selected_template_name]
    
    def get_template_by_name(self, template_name: str) -> DifferentiableMappingTemplate:
        """
        根据名称获取特定的模板实例。
        
        Args:
            template_name: 模板名称 ('full', 'tiled_k', 'tiled_n', 'tiled_kn')
            
        Returns:
            指定的映射模板实例
        """
        if template_name not in self.templates:
            raise ValueError(f"Unknown template name: {template_name}")
        return self.templates[template_name]
    
    def forward(self, use_probabilistic_selection: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播：计算该决策模块的性能指标。
        
        Args:
            use_probabilistic_selection: 是否使用概率性选择（用于训练）
                                       还是确定性选择（用于推理）
        
        Returns:
            性能指标字典，包含缓冲区需求、访存次数等
        """
        if use_probabilistic_selection:
            # 训练时：使用加权平均，考虑所有模板的贡献
            probabilities = self.get_template_probabilities()
            
            # 收集所有模板的结果
            all_results = {}
            for i, template_name in enumerate(self.template_names):
                template_result = self.templates[template_name].forward()
                for key, value in template_result.items():
                    if key not in all_results:
                        all_results[key] = []
                    all_results[key].append(value * probabilities[i])
            
            # 加权求和
            weighted_results = {}
            for key, value_list in all_results.items():
                if key == 'template_name':
                    # 模板名称特殊处理，返回概率最高的模板名称
                    selected_template = self.get_selected_template()
                    weighted_results[key] = selected_template.template_name
                else:
                    weighted_results[key] = sum(value_list)
            
            return weighted_results
        else:
            # 推理时：只使用选中的模板
            selected_template = self.get_selected_template()
            return selected_template.forward()
    
    def __str__(self):
        probabilities = self.get_template_probabilities()
        selected_template = self.get_selected_template()
        
        lines = [f"MappingDecisionModule for '{self.group_name}':"]
        lines.append("  Template Probabilities:")
        lines.append(f"    Full: {probabilities[0].item():.4f}")
        lines.append(f"    TiledK: {probabilities[1].item():.4f}")
        lines.append(f"    TiledN: {probabilities[2].item():.4f}")
        lines.append(f"    TiledKN: {probabilities[3].item():.4f}")
        lines.append(f"  Selected Template: {selected_template.template_name}")
        
        return "\n".join(lines)


class LayerMapping(nn.Module):
    """
    Mapping parameters for a single layer.
    Currently implements single-level tiling with L2 buffer.
    """
    def __init__(self, dims: Dict[str, int]):
        super().__init__()
        
        # Initialize tiling factors for each dimension
        self.temporal_factors_L2 = nn.ParameterDict({
            dim: nn.Parameter(torch.rand(1))
            for dim in Config.get_instance().PROBLEM_DIMS
            if dim in dims
        })
        
        # Store problem dimensions
        self.dims = dims
    
    def get_rounded_factors(self, dims: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Get rounded tiling factors that are valid divisors.
        
        Args:
            dims: Dictionary of problem dimensions
            
        Returns:
            Dictionary of rounded tiling factors
        """
        return {
            dim: round_tiling_factors(self.temporal_factors_L2[dim], dims[dim])
            for dim in self.temporal_factors_L2.keys()
            if dim in dims
        }
    
    def calculate_buffer_requirements(self, dims: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Calculate buffer size requirements for each dimension.
        
        Args:
            dims: Dictionary of problem dimensions
            
        Returns:
            Dictionary of buffer sizes per dimension
        """
        config = Config.get_instance()
        rounded_factors = self.get_rounded_factors(dims)
        
        # Calculate buffer requirements for each dimension
        buffer_reqs = {}
        for dim in rounded_factors:
            # Simple model: buffer size is proportional to tiling factor
            buffer_reqs[dim] = rounded_factors[dim] * config.BYTES_PER_ELEMENT
        
        return buffer_reqs
    
    def calculate_access_counts(self, dims: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """
        Calculate number of memory accesses for each dimension.
        
        Args:
            dims: Dictionary of problem dimensions
            
        Returns:
            Dictionary of access counts per dimension
        """
        rounded_factors = self.get_rounded_factors(dims)
        
        # Calculate access counts for each dimension
        access_counts = {}
        for dim in rounded_factors:
            # Simple model: access count is problem size divided by tiling factor
            access_counts[dim] = torch.tensor(dims[dim], dtype=torch.float32) / rounded_factors[dim]
        
        return access_counts

class ComputationGraph:
    """
    Represents the computation graph of the neural network.
    Tracks layer dimensions and fusion opportunities.
    """
    def __init__(self, layers: Dict[str, Dict] = None, edges: List[Tuple[str, str]] = None, fusion_groups: List[List[str]] = None):
        """
        Initialize computation graph.
        
        Args:
            layers: Optional dictionary of layer configurations
            edges: Optional list of edges between layers
            fusion_groups: Optional list of layer groups that can be fused
        """
        self.layers = layers or {}
        self.edges = edges or []
        self.fusion_groups = fusion_groups or []
    
    def add_layer(self, name: str, dims: Dict[str, int], op_type: str) -> None:
        """
        Add a layer to the graph.
        
        Args:
            name: Layer name
            dims: Dictionary of dimension sizes
            op_type: The operation type of the layer (e.g., 'Conv', 'ReLU')
        """
        self.layers[name] = {'dims': dims, 'type': op_type}
    
    def add_edge(self, src: str, dst: str) -> None:
        """
        Add an edge between layers.
        
        Args:
            src: Source layer name
            dst: Destination layer name
        """
        self.edges.append((src, dst))
    
    def add_fusion_group(self, group: List[str]) -> None:
        """
        Add a group of layers that can be fused.
        
        Args:
            group: List of layer names that can be fused together
        """
        self.fusion_groups.append(group)
    
    def get_layer_names(self) -> List[str]:
        """Get list of all layer names."""
        return list(self.layers.keys())
    
    def get_fusion_group_key(self, group: List[str]) -> str:
        """
        Generate a unique key for a fusion group.
        
        Args:
            group: List of layer names in the fusion group
            
        Returns:
            String key created by joining sorted layer names
        """
        return '__'.join(sorted(group))
    
    def get_node_producers(self, node_name: str) -> List[str]:
        """
        Get the names of all parent (producer) nodes for a given node.
        
        Args:
            node_name: Name of the target node
            
        Returns:
            List of producer node names
        """
        producers = []
        for src, dest in self.edges:
            if dest == node_name:
                producers.append(src)
        return producers
    
    def get_node_consumers(self, node_name: str) -> List[str]:
        """
        Get the names of all child (consumer) nodes for a given node.
        
        Args:
            node_name: Name of the target node
            
        Returns:
            List of consumer node names
        """
        consumers = []
        for src, dest in self.edges:
            if src == node_name:
                consumers.append(dest)
        return consumers
    
    def validate(self) -> None:
        """
        Validate graph structure.
        Ensures all referenced layers exist and fusion groups are valid.
        """
        layer_names = set(self.layers.keys())
        
        # Validate edges
        for src, dst in self.edges:
            if src not in layer_names:
                raise ValueError(f"Edge source layer '{src}' not found")
            if dst not in layer_names:
                raise ValueError(f"Edge destination layer '{dst}' not found")
        
        # Validate fusion groups
        for group in self.fusion_groups:
            for layer in group:
                if layer not in layer_names:
                    raise ValueError(f"Fusion group layer '{layer}' not found")
            
            # Check that fusion group layers are connected
            for i in range(len(group) - 1):
                if (group[i], group[i + 1]) not in self.edges:
                    raise ValueError(
                        f"Fusion group layers '{group[i]}' and '{group[i + 1]}' "
                        "must be connected by an edge"
                    )

class MappingParameters(nn.Module):
    """
    映射策略管理器 (Mapping Strategy Manager)
    
    重构后的映射策略管理器，使用 MappingDecisionModule 来管理映射策略。
    为计算图中的每一个融合组（fusion group）和每一个未被融合的独立层（standalone layer）
    都创建一个对应的 MappingDecisionModule。
    
    这种设计将优化的对象从"独立层的平铺参数"转变为"为一个计算组选择最佳映射模板，
    并微调该模板参数"。
    """
    
    def __init__(self, graph: ComputationGraph):
        super().__init__()
        
        self.graph = graph
        
        # === 决策模块管理 ===
        # 为每个计算组（融合组或独立层）创建一个映射决策模块
        self.decision_modules = nn.ModuleDict()
        
        # 跟踪哪些层属于融合组，哪些是独立层
        self.fused_layers = set()
        self.standalone_layers = set()
        self.group_mapping = {}  # 映射：组名 -> 组内层名列表
        
        # === 步骤1：处理融合组 ===
        for fusion_group in graph.fusion_groups:
            if fusion_group:  # 确保融合组非空
                group_key = self._get_sanitized_group_key(fusion_group)
                
                # 标记这些层为已融合
                for layer_name in fusion_group:
                    self.fused_layers.add(layer_name)
                
                # 计算融合组的代表性维度
                # TODO: 实际应用中需要更复杂的融合维度计算
                representative_dims = self._compute_fusion_group_dims(fusion_group)
                
                # 为融合组创建决策模块
                self.decision_modules[group_key] = MappingDecisionModule(
                    dims=representative_dims,
                    group_name=f"FusionGroup_{group_key}"
                )
                
                # 记录组映射
                self.group_mapping[group_key] = fusion_group.copy()
        
        # === 步骤2：处理独立层（未被融合的层）===
        for layer_name in graph.get_layer_names():
            if layer_name not in self.fused_layers:
                self.standalone_layers.add(layer_name)
                
                # 获取层的维度信息
                layer_dims = graph.layers[layer_name]['dims']
                
                # 为独立层创建决策模块
                sanitized_layer_name = sanitize_layer_name(layer_name)
                self.decision_modules[sanitized_layer_name] = MappingDecisionModule(
                    dims=layer_dims,
                    group_name=f"Layer_{layer_name}"
                )
                
                # 记录组映射（独立层也视为只有一个成员的"组"）
                self.group_mapping[sanitized_layer_name] = [layer_name]
    
    def _get_sanitized_group_key(self, fusion_group: List[str]) -> str:
        """为融合组生成清理后的键名"""
        group_key = self.graph.get_fusion_group_key(fusion_group)
        return sanitize_layer_name(group_key)
    
    def _compute_fusion_group_dims(self, fusion_group: List[str]) -> Dict[str, int]:
        """
        计算融合组的代表性维度。
        
        当前实现：使用第一个层的维度作为近似
        TODO: 实际应用中需要更复杂的融合维度计算，考虑层间的数据流和变换
        
        Args:
            fusion_group: 融合组中的层名称列表
            
        Returns:
            融合组的代表性维度字典
        """
        if not fusion_group:
            raise ValueError("Fusion group cannot be empty")
        
        # 简单策略：使用第一个层的维度
        first_layer_dims = self.graph.layers[fusion_group[0]]['dims']
        
        # TODO: 更复杂的融合维度计算可以在这里实现
        # 例如，考虑各层的维度变换、数据重用模式等
        
        return first_layer_dims.copy()
    
    def get_decision_module_for_layer(self, layer_name: str) -> MappingDecisionModule:
        """
        获取指定层对应的决策模块。
        
        Args:
            layer_name: 层名称
            
        Returns:
            该层对应的映射决策模块（可能是独立层的模块，也可能是融合组的模块）
        """
        # 首先检查是否为独立层
        sanitized_layer_name = sanitize_layer_name(layer_name)
        if sanitized_layer_name in self.decision_modules:
            return self.decision_modules[sanitized_layer_name]
        
        # 如果不是独立层，查找包含该层的融合组
        for group_key, group_layers in self.group_mapping.items():
            if layer_name in group_layers and len(group_layers) > 1:  # 融合组
                return self.decision_modules[group_key]
        
        raise ValueError(f"Layer '{layer_name}' not found in any decision module")
    
    def get_decision_module_for_fusion_group(self, fusion_group: List[str]) -> MappingDecisionModule:
        """
        获取指定融合组对应的决策模块。
        
        Args:
            fusion_group: 融合组中的层名称列表
            
        Returns:
            该融合组对应的映射决策模块
        """
        group_key = self._get_sanitized_group_key(fusion_group)
        if group_key not in self.decision_modules:
            raise ValueError(f"Fusion group {fusion_group} not found in decision modules")
        
        return self.decision_modules[group_key]
    
    def get_all_decision_modules(self) -> Dict[str, MappingDecisionModule]:
        """
        获取所有的决策模块。
        
        Returns:
            字典：{组名 -> MappingDecisionModule}
        """
        return dict(self.decision_modules)
    
    # === 兼容性方法：为了与现有代码兼容而保留 ===
    def get_mapping_by_original_name(self, original_name: str) -> DifferentiableMappingTemplate:
        """
        兼容性方法：根据层名称获取当前选中的映射模板。
        注意：这个方法返回的是 DifferentiableMappingTemplate 而不是 LayerMapping。
        """
        decision_module = self.get_decision_module_for_layer(original_name)
        return decision_module.get_selected_template()
    
    def get_layer_template_probabilities(self, layer_name: str) -> torch.Tensor:
        """
        兼容性方法：获取指定层的模板选择概率分布。
        
        Args:
            layer_name: 层名称
            
        Returns:
            四种模板的选择概率 [P(Full), P(TiledK), P(TiledN), P(TiledKN)]
        """
        decision_module = self.get_decision_module_for_layer(layer_name)
        return decision_module.get_template_probabilities()
    
    def get_layer_selected_template(self, layer_name: str) -> DifferentiableMappingTemplate:
        """
        兼容性方法：根据当前学习到的概率分布，为指定层选择最佳映射模板。
        
        Args:
            layer_name: 层名称
            
        Returns:
            选中的映射模板实例
        """
        decision_module = self.get_decision_module_for_layer(layer_name)
        return decision_module.get_selected_template()
    
    def get_fusion_group_template_probabilities(self, fusion_group: List[str]) -> torch.Tensor:
        """
        兼容性方法：获取指定融合组的模板选择概率分布。
        
        Args:
            fusion_group: 融合组中的层名称列表
            
        Returns:
            四种模板的选择概率 [P(Full), P(TiledK), P(TiledN), P(TiledKN)]
        """
        decision_module = self.get_decision_module_for_fusion_group(fusion_group)
        return decision_module.get_template_probabilities()
    
    def get_fusion_group_selected_template(self, fusion_group: List[str]) -> DifferentiableMappingTemplate:
        """
        兼容性方法：根据当前学习到的概率分布，为指定融合组选择最佳映射模板。
        
        Args:
            fusion_group: 融合组中的层名称列表
            
        Returns:
            选中的映射模板实例
        """
        decision_module = self.get_decision_module_for_fusion_group(fusion_group)
        return decision_module.get_selected_template()
    
    def __str__(self):
        lines = ["Mapping Strategy Manager (DMT-based):"]
        lines.append(f"  Total Decision Modules: {len(self.decision_modules)}")
        lines.append(f"  Standalone Layers: {len(self.standalone_layers)}")
        lines.append(f"  Fusion Groups: {len(self.graph.fusion_groups)}")
        lines.append("")
        
        # 显示所有决策模块的状态
        for group_key, decision_module in self.decision_modules.items():
            group_layers = self.group_mapping.get(group_key, [])
            if len(group_layers) == 1:
                lines.append(f"  Standalone Layer: {group_layers[0]}")
            else:
                lines.append(f"  Fusion Group: {group_layers}")
            
            # 显示概率分布和选中的模板
            probabilities = decision_module.get_template_probabilities()
            selected_template = decision_module.get_selected_template()
            lines.append(f"    Probabilities: Full={probabilities[0].item():.3f}, "
                        f"TiledK={probabilities[1].item():.3f}, "
                        f"TiledN={probabilities[2].item():.3f}, "
                        f"TiledKN={probabilities[3].item():.3f}")
            lines.append(f"    Selected: {selected_template.template_name}")
            lines.append("")
        
        return "\n".join(lines)

class HardwareParameters(nn.Module):
    """
    Learnable hardware configuration parameters for joint hardware-software co-design.
    This enables the optimization of architectural choices alongside mapping and fusion decisions.
    
    ENHANCED FOR 40NM PROCESS: Now supports hierarchical memory system with separate
    accumulator and scratchpad sizes for more accurate energy modeling.
    """
    def __init__(self, initial_num_pes: int = 64, initial_accumulator_kb: float = 64.0, initial_scratchpad_kb: float = 192.0):
        super().__init__()
        
        # --- FIX: Add epsilon protection for log operations ---
        epsilon = 1e-9  # Small positive value to prevent log(0)
        
        # === Learnable parameters in log-scale for better optimization ===
        # Log-scale helps with gradient flow and ensures positive values
        # Ensure inputs are positive and add epsilon for numerical stability
        safe_num_pes = max(initial_num_pes, epsilon) + epsilon
        safe_accumulator_size = max(initial_accumulator_kb, epsilon) + epsilon
        safe_scratchpad_size = max(initial_scratchpad_kb, epsilon) + epsilon
        
        self.log_num_pes = nn.Parameter(torch.log(torch.tensor(safe_num_pes, dtype=torch.float32)))
        self.log_accumulator_size_kb = nn.Parameter(torch.log(torch.tensor(safe_accumulator_size, dtype=torch.float32)))
        self.log_scratchpad_size_kb = nn.Parameter(torch.log(torch.tensor(safe_scratchpad_size, dtype=torch.float32)))
        
        # Store initial values for reference
        self.initial_num_pes = initial_num_pes
        self.initial_accumulator_kb = initial_accumulator_kb
        self.initial_scratchpad_kb = initial_scratchpad_kb
    
    def get_num_pes(self) -> torch.Tensor:
        """Get the number of processing elements (differentiable)."""
        return torch.exp(self.log_num_pes)
    
    def get_accumulator_size_kb(self) -> torch.Tensor:
        """Get the accumulator size in KB (differentiable)."""
        return torch.exp(self.log_accumulator_size_kb)
    
    def get_scratchpad_size_kb(self) -> torch.Tensor:
        """Get the scratchpad size in KB (differentiable)."""
        return torch.exp(self.log_scratchpad_size_kb)
    
    def get_buffer_size_kb(self) -> torch.Tensor:
        """Get the total buffer size in KB (accumulator + scratchpad)."""
        return self.get_accumulator_size_kb() + self.get_scratchpad_size_kb()
    
    def get_buffer_size_bytes(self) -> torch.Tensor:
        """Get the total buffer size in bytes (differentiable)."""
        return self.get_buffer_size_kb() * 1024.0
    
    def get_mac_throughput_gops(self) -> torch.Tensor:
        """
        Calculate MAC throughput based on number of PEs and clock frequency.
        
        Returns:
            MAC throughput in GOPS (Giga Operations Per Second)
        """
        config = Config.get_instance()
        num_pes = self.get_num_pes()
        
        # MAC throughput = num_PEs * MACs_per_PE_per_cycle * clock_frequency_MHz / 1000
        # Division by 1000 converts from MHz to GHz for GOPS
        mac_throughput = (
            num_pes * 
            torch.tensor(config.MACS_PER_PE_PER_CYCLE, dtype=torch.float32) * 
            torch.tensor(config.CLOCK_FREQUENCY_MHZ / 1000.0, dtype=torch.float32)
        )
        return mac_throughput
    
    def get_area_cost(self) -> torch.Tensor:
        """
        Calculate total chip area based on PEs and hierarchical buffer sizes.
        
        Returns:
            Total area in mm²
        """
        config = Config.get_instance()
        num_pes = self.get_num_pes()
        total_buffer_size_kb = self.get_buffer_size_kb()
        
        # Total area = base_area + PE_area + buffer_area
        area_cost = (
            torch.tensor(config.AREA_BASE_MM2, dtype=torch.float32) +
            num_pes * torch.tensor(config.AREA_PER_PE_MM2, dtype=torch.float32) +
            total_buffer_size_kb * torch.tensor(config.AREA_PER_KB_SRAM_MM2, dtype=torch.float32)
        )
        return area_cost
    
    def __str__(self):
        lines = ["Hardware Parameters:"]
        lines.append(f"  Number of PEs: {self.get_num_pes().item():.1f}")
        lines.append(f"  Accumulator Size: {self.get_accumulator_size_kb().item():.1f} KB")
        lines.append(f"  Scratchpad Size: {self.get_scratchpad_size_kb().item():.1f} KB")
        lines.append(f"  Total Buffer Size: {self.get_buffer_size_kb().item():.1f} KB")
        lines.append(f"  MAC Throughput: {self.get_mac_throughput_gops().item():.1f} GOPS")
        lines.append(f"  Area Cost: {self.get_area_cost().item():.2f} mm²")
        return "\n".join(lines)

def calculate_hardware_config(
    mapping_params: 'MappingParameters',
    graph: 'ComputationGraph'
) -> Dict[str, torch.Tensor]:
    """
    DEPRECATED: This function is deprecated in favor of HardwareParameters class.
    
    Hardware configuration is now handled by the learnable HardwareParameters module
    which enables joint optimization of hardware and software parameters.
    
    This function is kept for backward compatibility but should not be used in new code.
    Use HardwareParameters.get_area_cost() and HardwareParameters.get_buffer_size_bytes() instead.
    
    Args:
        mapping_params: MappingParameters object containing layer mappings
        graph: ComputationGraph object defining the network structure
    
    Returns:
        Dictionary containing hardware configuration parameters (for compatibility only)
    """
    # Return minimal placeholder for backward compatibility
    return {
        'buffer_size': torch.tensor(0.0),
        'area_cost': torch.tensor(0.0)
    }

def calculate_fused_group_buffer_req(
    group: List[str],
    mapping_params: 'MappingParameters',
    graph: 'ComputationGraph'
) -> torch.Tensor:
    """
    Calculates the total buffer requirement for a (potentially) fused group.
    Uses a simple sum of individual layer requirements.
    
    Args:
        group: A list of layer names in the fusion group.
        mapping_params: The mapping parameters for all layers.
        graph: The computation graph.
        
    Returns:
        The total buffer size required for the group.
    """
    total_buffer_req = torch.tensor(0.0)
    for layer_name in group:
        layer_mapping = mapping_params.get_mapping_by_original_name(layer_name)
        buffer_reqs = layer_mapping.get_buffer_requirements()
        total_buffer_req += sum(buffer_reqs.values())
    return total_buffer_req

class FusionParameters(nn.Module):
    """
    Learnable parameters for fusion decisions.
    Each potential fusion group has a learnable probability.
    """
    def __init__(self, graph: ComputationGraph):
        super().__init__()
        
        # Create fusion probability parameters for each potential fusion group
        self.fusion_probs = nn.ParameterDict()
        self.group_name_mapping = {}  # Map sanitized group names to original groups
        
        for group in graph.fusion_groups:
            # Create a sanitized group key
            group_key = graph.get_fusion_group_key(group)
            sanitized_key = sanitize_layer_name(group_key)
            
            self.fusion_probs[sanitized_key] = nn.Parameter(torch.rand(1))
            self.group_name_mapping[sanitized_key] = group
    
    def get_fusion_probability(self, group: List[str]) -> torch.Tensor:
        """
        Get the fusion probability for a given group.
        
        Args:
            group: List of layer names in the fusion group
            
        Returns:
            Fusion probability (sigmoid-activated)
        """
        # Create group key and sanitize it
        group_key = '__'.join(sorted(group))
        sanitized_key = sanitize_layer_name(group_key)
        
        if sanitized_key in self.fusion_probs:
            return torch.sigmoid(self.fusion_probs[sanitized_key])
        else:
            # If group not found, return 0 probability
            return torch.tensor(0.0)
    
    def __str__(self):
        lines = ["Fusion Parameters:"]
        for sanitized_key, prob in self.fusion_probs.items():
            original_group = self.group_name_mapping[sanitized_key]
            group_str = ' -> '.join(original_group)
            lines.append(f"  {group_str}: {torch.sigmoid(prob).item():.4f}")
        return "\n".join(lines)

class ConditionalPerformanceModel(nn.Module):
    """
    Models both performance and hardware costs of neural networks based on 
    mapping and fusion decisions. Implements analytical cost models from DOSA paper.
    
    ENHANCED FOR 40NM PROCESS: Now implements physics-based dynamic energy model
    with hierarchical memory system (Registers, Accumulator, Scratchpad, DRAM).
    
    CNN Operators Supported:
    - Conv: Full convolution cost modeling with spatial kernels
    - BatchNormalization, Pooling: Standard CNN operations
    - Relu, Add: Element-wise operations with spatial dimensions
    
    Transformer Operators Supported:
    - MatMul/Gemm: Matrix multiplication with sequence-aware MAC calculation
    - LayerNormalization: Element-wise normalization with increased computational cost
    - Gelu, Softmax: Transformer-specific activations treated as element-wise
    - Attention: Multi-head attention (if fused) with quadratic sequence complexity
    
    Cost Model Adaptations for Transformers:
    - Sequence length (SeqLen) is mapped to the P dimension
    - Hidden features are mapped to the K dimension  
    - Transformer tensors are typically 3D: (N, SeqLen, Features) vs CNN 4D: (N, C, H, W)
    - MAC calculations are adjusted for matrix multiplication vs convolution patterns
    - Memory access patterns account for sequence-based data movement
    
    Important Assumption:
    The fusion_groups defined in the graph are assumed to be non-overlapping.
    This means no layer can appear in more than one fusion group.
    """
    def __init__(self):
        super().__init__()
    
    def _get_dynamic_epa(self, hardware_params: HardwareParameters) -> dict:
        """
        根据新的40nm工艺公式计算各组件的动态EPA。
        返回一个包含各级别EPA的字典，单位是 pJ。
        
        Args:
            hardware_params: HardwareParameters对象，包含当前的硬件配置
            
        Returns:
            包含各级别EPA的字典，单位是 pJ
        """
        # 从 hardware_params 获取实时、可微的硬件配置
        num_pes = hardware_params.get_num_pes()
        c1_kb = hardware_params.get_accumulator_size_kb()  # Level 1: Accumulator
        c2_kb = hardware_params.get_scratchpad_size_kb()   # Level 2: Scratchpad

        # 单位转换因子：从 µJ 转换为 pJ
        UJ_TO_PJ = 1e6

        # 计算各级别EPA (单位: pJ)
        # PE: 一次乘加运算 (MAC) 的能耗
        epa_pe = 0.561 * UJ_TO_PJ
        
        # Registers (Level 0): 访问一次寄存器的能耗
        epa_reg = 0.487 * UJ_TO_PJ
        
        # Accumulator (Level 1): 动态公式，基于容量和PE数量
        epa_acc = (1.94 + 0.1005 * (c1_kb / torch.sqrt(num_pes))) * UJ_TO_PJ
        
        # Scratchpad (Level 2): 动态公式，基于容量
        epa_sp = (0.49 + 0.025 * c2_kb) * UJ_TO_PJ
        
        # DRAM (Level 3): 固定值
        epa_dram = 100.0 * UJ_TO_PJ

        return {
            'pe': epa_pe,
            'reg': epa_reg,
            'acc': epa_acc,
            'sp': epa_sp,
            'dram': epa_dram
        }
    
    def _calculate_compute_latency(self, total_macs: torch.Tensor, hardware_params: 'HardwareParameters') -> torch.Tensor:
        """
        Calculate compute latency based on MAC throughput from hardware parameters.
        
        Args:
            total_macs: Total number of multiply-accumulate operations
            hardware_params: HardwareParameters object containing learnable hardware config
            
        Returns:
            Compute latency in cycles, with calibration factor applied
        """
        config = Config.get_instance()
        mac_throughput = hardware_params.get_mac_throughput_gops()
        compute_latency = total_macs / mac_throughput
        # Apply calibration factor for model tuning
        return compute_latency * torch.tensor(config.LATENCY_FUDGE_FACTOR, dtype=torch.float32)
    
    def _calculate_memory_latency(self, access_counts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate memory latency based on DRAM bandwidth limitations.
        
        Args:
            access_counts: Dictionary of memory access counts per dimension
            
        Returns:
            Memory latency in cycles, with calibration factor applied
        """
        config = Config.get_instance()
        total_accesses = sum(access_counts.values())
        memory_latency = total_accesses / torch.tensor(config.DRAM_BANDWIDTH, dtype=torch.float32)
        # Apply calibration factor for model tuning
        return memory_latency * torch.tensor(config.LATENCY_FUDGE_FACTOR, dtype=torch.float32)
    
    def _calculate_compute_energy(self, total_macs: torch.Tensor) -> torch.Tensor:
        """
        DEPRECATED: This method is deprecated in favor of the new dynamic energy model.
        Use _get_dynamic_epa() method instead.
        
        Args:
            total_macs: Total number of multiply-accumulate operations
            
        Returns:
            Compute energy in picojoules (legacy calculation)
        """
        config = Config.get_instance()
        compute_energy = total_macs * torch.tensor(config.MAC_ENERGY_PJ, dtype=torch.float32)
        # Apply calibration factor for model tuning
        return compute_energy * torch.tensor(config.COMPUTE_ENERGY_FACTOR, dtype=torch.float32)
    
    def _calculate_sram_energy(self, buffer_requirements: Dict[str, torch.Tensor], hardware_params: 'HardwareParameters') -> torch.Tensor:
        """
        DEPRECATED: This method is deprecated in favor of the new dynamic energy model.
        Use _get_dynamic_epa() method instead.
        
        Args:
            buffer_requirements: Dictionary of buffer size requirements per dimension
            hardware_params: HardwareParameters object containing learnable buffer size
            
        Returns:
            SRAM energy in picojoules (legacy calculation)
        """
        # Use the new dynamic EPA model instead of old config parameters
        dynamic_epa = self._get_dynamic_epa(hardware_params)
        
        # Calculate total buffer accesses
        total_buffer_accesses = sum(buffer_requirements.values())
        
        # Use the new dynamic scratchpad EPA
        sram_energy = total_buffer_accesses * dynamic_epa['sp']
        
        return sram_energy
    
    def _calculate_dram_energy(self, access_counts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        DEPRECATED: This method is deprecated in favor of the new dynamic energy model.
        Use _get_dynamic_epa() method instead.
        
        Args:
            access_counts: Dictionary of memory access counts per dimension
            
        Returns:
            DRAM energy in picojoules (legacy calculation)
        """
        # For legacy compatibility, we need hardware_params to get dynamic EPA
        # Since this method doesn't have hardware_params, we'll use a default value
        # This is a temporary fix - ideally this method should be removed
        total_dram_accesses = sum(access_counts.values())
        
        # Use the fixed DRAM EPA from 40nm model (100 µJ = 100,000,000 pJ)
        dram_epa = 100.0 * 1e6  # 100 µJ in pJ
        dram_energy = total_dram_accesses * torch.tensor(dram_epa, dtype=torch.float32)
        
        return dram_energy
    
    def _calculate_noc_energy(self, layer_name: str, graph: ComputationGraph, is_fused: bool = False) -> torch.Tensor:
        """
        Calculate Network-on-Chip (NoC) energy for data movement.
        Models on-chip data movement between processing elements and memory hierarchy.
        
        Args:
            layer_name: Name of the layer
            graph: ComputationGraph object
            is_fused: Whether this layer is part of a fused operation
            
        Returns:
            NoC energy in picojoules, with calibration factor applied
        """
        config = Config.get_instance()
        layer_dims = graph.layers[layer_name]['dims']
        
        # Estimate data movement on NoC
        # For conv layers: weights + activations movement
        if 'R' in layer_dims and 'S' in layer_dims:
            # Weight data movement: C * K * R * S
            weight_data = torch.tensor(
                layer_dims['C'] * layer_dims['K'] * layer_dims['R'] * layer_dims['S'] * config.BYTES_PER_ELEMENT,
                dtype=torch.float32
            )
            # Activation data movement: N * C * P * Q (input) + N * K * P * Q (output)
            activation_data = torch.tensor(
                layer_dims['N'] * (layer_dims['C'] + layer_dims['K']) * layer_dims['P'] * layer_dims['Q'] * config.BYTES_PER_ELEMENT,
                dtype=torch.float32
            )
        else:
            # For element-wise operations: primarily activation movement
            weight_data = torch.tensor(0.0, dtype=torch.float32)
            activation_data = torch.tensor(
                layer_dims['N'] * layer_dims.get('K', layer_dims.get('C', 1)) * 
                layer_dims.get('P', 1) * layer_dims.get('Q', 1) * config.BYTES_PER_ELEMENT,
                dtype=torch.float32
            )
        
        total_data_movement = weight_data + activation_data
        
        # For fused operations, there's additional control overhead but less intermediate data movement
        if is_fused:
            # Increase NoC energy due to more complex routing and control
            noc_overhead_factor = torch.tensor(1.2, dtype=torch.float32)
            total_data_movement = total_data_movement * noc_overhead_factor
        
        noc_energy = total_data_movement * torch.tensor(config.NOC_ENERGY_PER_BYTE_PJ, dtype=torch.float32)
        
        # Apply calibration factor for model tuning
        return noc_energy * torch.tensor(config.NOC_ENERGY_FACTOR, dtype=torch.float32)
    


    def calculate_group_costs(
        self,
        group_layers: List[str],
        template_instance: DifferentiableMappingTemplate,
        hardware_params: HardwareParameters,
        graph: ComputationGraph,
        is_fusion_calculation: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        核心成本计算函数：基于映射模板精确计算融合链的成本。
        
        ENHANCED FOR MAPPING-FUSION SYNERGY: 当用于融合成本计算时，
        融合开销会根据映射模板的Tiling因子动态调整，实现真正的协同优化。
        
        严格遵循《Mind the Gap》论文 V-A 至 V-C 节的融合链成本计算逻辑：
        - 总 Buffer 需求 = max_e(BufReq_e)
        - 总 DRAM 访问 = Access_W (所有权重) + Access_I,0 (第一个算子输入) + Access_O,E-1 (最后一个算子输出)
        - 中间张量的 DRAM 读写被完全消除
        
        Args:
            group_layers: 算子组中的层名称列表（可能只有一个层）
            template_instance: 被指定的映射模板实例
            hardware_params: 硬件参数
            graph: 计算图
            is_fusion_calculation: 如果为True，表示这是为融合组计算成本，会应用动态融合开销
            
        Returns:
            (total_latency, total_energy): 该组的总延迟和总能耗
        """
        config = Config.get_instance()
        
        # === 1. 计算总计算量（MACs）===
        total_macs = torch.tensor(0.0, dtype=torch.float32)
        for layer_name in group_layers:
            layer_info = graph.layers[layer_name]
            layer_macs = calculate_macs(layer_info['dims'], layer_info['type'])
            total_macs = total_macs + layer_macs
        
        # === 2. 计算融合链的缓冲区需求 ===
        # 模板实例已经为整个组计算了缓冲区需求，直接使用
        max_buffer_requirements = template_instance.get_buffer_requirements()
        
        # === 3. 计算融合链的 DRAM 访问量 ===
        # 模板实例已经为整个组计算了 DRAM 访问量，直接使用
        # 这里包含了论文中的逻辑：Access_W (所有权重) + Access_I,0 (第一个算子输入) + Access_O,E-1 (最后一个算子输出)
        total_dram_accesses = template_instance.get_access_counts()
        
        # 如果是多层融合组，需要根据层数调整访问量
        if len(group_layers) > 1:
            # 权重访问需要按层数比例调整（每层都有自己的权重）
            if 'weight' in total_dram_accesses:
                total_dram_accesses['weight'] = total_dram_accesses['weight'] * len(group_layers)
            
            # 输入输出访问保持不变（按融合链逻辑：只有首层输入和末层输出需要DRAM访问）
        
        # === 4. 计算延迟 ===
        # 4.1 计算延迟 (REVISED to simple additive model)
        # 真实延迟更接近计算延迟和内存延迟的和，而不是最大值。
        # 这个更真实的模型会给优化器正确的信号。
        compute_latency = self._calculate_compute_latency(total_macs, hardware_params)
        # 注意: _calculate_memory_latency 内部已经处理了带宽限制，所以它代表了DRAM传输时间。
        memory_latency = self._calculate_memory_latency(total_dram_accesses)
        
        # 总延迟是计算时间和内存时间的累加。这是更符合物理现实的模型。
        total_latency = compute_latency + memory_latency
        
        # 4.2 动态融合控制开销 (REFACTORED FOR STABILITY)
        if len(group_layers) > 1 and is_fusion_calculation:
            # 使用简化的、固定的融合开销模型以保证数值稳定性
            base_overhead_cycles = config.FUSION_LATENCY_PENALTY_CYCLES
            # 为融合组中的额外层增加少量固定开销，模型更稳定
            per_layer_overhead_cycles = 10.0  # 可配置的稳定超参数
            
            total_fusion_overhead = torch.tensor(
                base_overhead_cycles + per_layer_overhead_cycles * (len(group_layers) - 1),
                dtype=torch.float32
            )
            total_latency = total_latency + total_fusion_overhead
            
        elif len(group_layers) > 1:
            # 非融合计算时的静态开销（兼容性保持）
            fusion_control_overhead = torch.tensor(config.FUSION_LATENCY_PENALTY_CYCLES, dtype=torch.float32)
            total_latency = total_latency + fusion_control_overhead
        
        # === 5. 计算能耗：使用新的40nm工艺动态能耗模型 ===
        # 5.1 获取动态EPA值
        dynamic_epa = self._get_dynamic_epa(hardware_params)
        
        # 5.2 计算分级访存次数（需要从模板获取详细的分级访问信息）
        # 注意：这里需要模板提供分级访问次数，暂时使用简化的估算
        # TODO: 重构模板的get_access_counts方法以返回分级访问次数
        
        # 估算各级别的访问次数（基于当前模板的访问模式）
        # 这是一个简化的实现，实际应该从模板获取精确的分级访问次数
        total_accesses = sum(total_dram_accesses.values())
        
        # 简化的分级访问估算：
        # - Registers: 假设每个MAC操作需要2次寄存器访问（读+写）
        register_accesses = total_macs * 2.0
        
        # - Accumulator: 假设每个MAC操作需要1次累加器访问
        accumulator_accesses = total_macs
        
        # - Scratchpad: 基于缓冲区需求估算
        total_buffer_size = sum(max_buffer_requirements.values())
        scratchpad_accesses = total_buffer_size * 2.0  # 读写各一次
        
        # - DRAM: 使用模板提供的DRAM访问次数
        dram_accesses = total_accesses
        
        # 5.3 计算分级能耗
        compute_energy = total_macs * dynamic_epa['pe']  # PE能耗
        register_energy = register_accesses * dynamic_epa['reg']  # 寄存器能耗
        accumulator_energy = accumulator_accesses * dynamic_epa['acc']  # 累加器能耗
        scratchpad_energy = scratchpad_accesses * dynamic_epa['sp']  # 暂存器能耗
        dram_energy = dram_accesses * dynamic_epa['dram']  # DRAM能耗
        
        # 5.4 计算 NoC 能耗（考虑融合的影响）
        noc_energy = torch.tensor(0.0, dtype=torch.float32)
        is_fused = len(group_layers) > 1
        for layer_name in group_layers:
            noc_energy = noc_energy + self._calculate_noc_energy(layer_name, graph, is_fused)
        
        # 5.5 动态融合开销能耗 (REFACTORED FOR STABILITY)
        fusion_overhead_energy = torch.tensor(0.0, dtype=torch.float32)
        if is_fused and is_fusion_calculation:
            # 使用简化的、固定的融合能耗开销
            base_fusion_energy = config.FUSION_OVERHEAD_COST
            per_layer_energy_overhead = 50.0  # 另一个可配置的稳定超参数
            fusion_overhead_energy = torch.tensor(
                base_fusion_energy + per_layer_energy_overhead * (len(group_layers) - 1),
                dtype=torch.float32
            )
            
        elif is_fused:
            # 静态融合能耗开销（兼容性保持）
            fusion_overhead_energy = torch.tensor(config.FUSION_OVERHEAD_COST, dtype=torch.float32)
        
        # 总能耗：所有分级能耗的总和
        total_energy = (compute_energy + register_energy + accumulator_energy + 
                       scratchpad_energy + dram_energy + noc_energy + fusion_overhead_energy)
        
        # === 6. 数值稳定性检查 ===
        min_value = 1e-6
        total_latency = torch.clamp(total_latency, min=min_value)
        total_energy = torch.clamp(total_energy, min=min_value)
        
        return total_latency, total_energy


    def _calculate_analytical_costs(
        self,
        layer_name: str,
        mapping_params: MappingParameters,
        graph: ComputationGraph,
        hardware_params: HardwareParameters,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compatibility method for baseline experiments using the corrected ADDITIVE latency model.
        Returns: (latency, comp_energy, sram_energy, dram_energy, noc_energy)
        """
        template_instance = mapping_params.get_layer_selected_template(layer_name)
        
        # Use the main cost function to get total latency and energy
        total_latency, total_energy = self.calculate_group_costs(
            [layer_name], template_instance, hardware_params, graph, is_fusion_calculation=False
        )

        # For compatibility with baselines, we provide an estimated breakdown.
        # This breakdown doesn't affect FA-DOSA's own optimization, which uses total_energy.
        comp_energy = total_energy * 0.3  # Rough estimation
        dram_energy = total_energy * 0.5  
        sram_energy = total_energy * 0.15 
        noc_energy = total_energy * 0.05
        
        return total_latency, comp_energy, sram_energy, dram_energy, noc_energy


    def forward(
        self,
        mapping_params: MappingParameters,
        fusion_params: FusionParameters,
        hardware_params: HardwareParameters,
        graph: ComputationGraph
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        REFACTORED FORWARD METHOD: Unified, Expectation-Based Cost Calculation.

        This method calculates total latency, energy, and area by consistently applying
        an expectation model over mapping templates for both fused and non-fused scenarios.
        This resolves the conflicting gradient problem of the previous implementation.
        """
        config = Config.get_instance()
        area_cost = hardware_params.get_area_cost()

        total_latency = torch.tensor(0.0, dtype=torch.float32)
        total_energy = torch.tensor(0.0, dtype=torch.float32)

        # Identify all computational groups (standalone layers and fusion groups)
        processed_layers = set()
        computational_groups = []
        for group in graph.fusion_groups:
            computational_groups.append(group)
            for layer in group:
                processed_layers.add(layer)
        for layer_name in graph.get_layer_names():
            if layer_name not in processed_layers:
                computational_groups.append([layer_name])

        # Iterate over each computational group
        for group_layers in computational_groups:
            is_fusion_group = len(group_layers) > 1

            # A. Calculate the cost of the group if it runs UNFUSED (sum of individual layer costs)
            non_fused_latency = torch.tensor(0.0, dtype=torch.float32)
            non_fused_energy = torch.tensor(0.0, dtype=torch.float32)
            for layer_name in group_layers:
                # Each layer's cost is the expectation over its own mapping templates
                layer_decision_module = mapping_params.get_decision_module_for_layer(layer_name)
                template_probs = layer_decision_module.get_template_probabilities()
                
                exp_lat = torch.tensor(0.0, dtype=torch.float32)
                exp_eng = torch.tensor(0.0, dtype=torch.float32)
                for i, template_name in enumerate(layer_decision_module.template_names):
                    template = layer_decision_module.get_template_by_name(template_name)
                    # Calculate cost for this specific template as a standalone unit
                    lat, eng = self.calculate_group_costs(
                        [layer_name], template, hardware_params, graph, is_fusion_calculation=False
                    )
                    # Ensure scalar multiplication
                    prob_scalar = template_probs[i].item() if template_probs[i].dim() > 0 else template_probs[i]
                    exp_lat = exp_lat + prob_scalar * lat
                    exp_eng = exp_eng + prob_scalar * eng
                
                non_fused_latency = non_fused_latency + exp_lat
                non_fused_energy = non_fused_energy + exp_eng

            # B. If it's a fusion group, calculate the cost if it runs FUSED
            if is_fusion_group:
                # The fused cost is the expectation over the FUSION GROUP's mapping templates
                group_decision_module = mapping_params.get_decision_module_for_fusion_group(group_layers)
                template_probs = group_decision_module.get_template_probabilities()

                fused_latency = torch.tensor(0.0, dtype=torch.float32)
                fused_energy = torch.tensor(0.0, dtype=torch.float32)

                for i, template_name in enumerate(group_decision_module.template_names):
                    template = group_decision_module.get_template_by_name(template_name)
                    # Calculate cost for this template over the WHOLE FUSED GROUP
                    lat, eng = self.calculate_group_costs(
                        group_layers, template, hardware_params, graph, is_fusion_calculation=True
                    )
                    # Ensure scalar multiplication
                    prob_scalar = template_probs[i].item() if template_probs[i].dim() > 0 else template_probs[i]
                    fused_latency = fused_latency + prob_scalar * lat
                    fused_energy = fused_energy + prob_scalar * eng
                
                # C. Combine fused and non-fused costs using the learnable fusion probability
                p_fuse = fusion_params.get_fusion_probability(group_layers)
                p_fuse_scalar = p_fuse.item() if p_fuse.dim() > 0 else p_fuse
                final_group_latency = p_fuse_scalar * fused_latency + (1 - p_fuse_scalar) * non_fused_latency
                final_group_energy = p_fuse_scalar * fused_energy + (1 - p_fuse_scalar) * non_fused_energy
            else:
                # It's a standalone layer, its cost is simply the non_fused_cost
                final_group_latency = non_fused_latency
                final_group_energy = non_fused_energy
            
            # D. Accumulate the final cost for this group
            total_latency = total_latency + final_group_latency
            total_energy = total_energy + final_group_energy

        # Final numerical stability checks
        min_value = 1e-9
        total_latency = torch.clamp(total_latency, min=min_value)
        total_energy = torch.clamp(total_energy, min=min_value)
        area_cost = torch.clamp(area_cost, min=min_value)

        assert not torch.isinf(total_latency), f"Latency is inf"
        assert not torch.isnan(total_latency), f"Latency is NaN"
        assert not torch.isinf(total_energy), f"Energy is inf"
        assert not torch.isnan(total_energy), f"Energy is NaN"

        return total_latency, total_energy, area_cost

    def calculate_group_costs(
        self,
        group_layers: List[str],
        template_instance: DifferentiableMappingTemplate,
        hardware_params: HardwareParameters,
        graph: ComputationGraph,
        is_fusion_calculation: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core cost calculation function. Calculates latency and energy for a computational group
        (either a single layer or a fused group) based on a GIVEN mapping template.

        Args:
            group_layers: List of layer names in the group.
            template_instance: The specific mapping template to use for cost calculation.
            hardware_params: The current hardware parameters.
            graph: The computation graph.
            is_fusion_calculation: A boolean indicating if we should apply fusion physics.

        Returns:
            A tuple of (latency, energy).
        """
        config = Config.get_instance()

        # 1. Calculate total MACs for the group
        total_macs = sum(calculate_macs(graph.layers[name]['dims'], graph.layers[name]['type']) for name in group_layers)

        # 2. Get memory access counts from the template
        access_counts = template_instance.get_access_counts()

        # 3. Apply fusion physics if this is a fused calculation
        if is_fusion_calculation and len(group_layers) > 1:
            # Key fusion benefit: eliminate intermediate DRAM access.
            # The provided template access counts are for a SINGLE op. We must adjust them.
            # This is a simplified model. A more advanced one would analyze the dataflow inside the group.
            
            # Total weights access = sum of all layers' weights access
            total_weight_access = access_counts['weight'] * len(group_layers)
            
            # Total I/O access = Input of first layer + Output of last layer
            # The template's 'input' and 'output' counts represent one op, which is a good approximation here.
            total_io_access = access_counts['input'] + access_counts['output']
            
            total_dram_access_bytes = (total_weight_access + total_io_access) * config.BYTES_PER_ELEMENT
        else:
            # For non-fused calculation, just sum up accesses for all layers in the group
            total_dram_access_bytes = sum(access_counts.values()) * len(group_layers) * config.BYTES_PER_ELEMENT

        # 4. Calculate Latency (REVISED to additive model)
        compute_latency = self._calculate_compute_latency(total_macs, hardware_params)
        memory_latency = total_dram_access_bytes / (config.DRAM_BANDWIDTH * 1e9) # Convert GB/s to B/s
        latency = compute_latency + memory_latency
        
        # Add fusion overhead for latency if fused
        if is_fusion_calculation and len(group_layers) > 1:
            latency += config.FUSION_LATENCY_PENALTY_CYCLES

        # 5. Calculate Energy
        dynamic_epa = self._get_dynamic_epa(hardware_params)
        compute_energy = total_macs * dynamic_epa['pe']
        dram_energy = total_dram_access_bytes * dynamic_epa['dram'] # Simplified DRAM energy
        
        # Simplified SRAM energy based on buffer requirements from the template
        buffer_reqs_words = sum(template_instance.get_buffer_requirements().values())
        sram_energy = buffer_reqs_words * config.BYTES_PER_ELEMENT * dynamic_epa['sp']
        
        total_energy = compute_energy + dram_energy + sram_energy
        
        # Add fusion overhead for energy if fused
        if is_fusion_calculation and len(group_layers) > 1:
            total_energy += config.FUSION_OVERHEAD_COST
            
        return latency, total_energy

def calculate_product_constraint_penalty(
    mapping_params: 'MappingParameters',
    graph: 'ComputationGraph'
) -> torch.Tensor:
    """
    Calculate penalty for violation of the product constraint.
    The product of tiling factors for each dimension should equal the problem size.
    Uses log-domain difference for numerical stability.
    
    Args:
        mapping_params: MappingParameters object containing layer mappings
        graph: ComputationGraph object defining the network structure
        
    Returns:
        Total penalty summed across all dimensions and layers
    """
    total_penalty = torch.tensor(0.0)
    
    # --- FIX: Add epsilon for numerical stability in log operations ---
    epsilon = 1e-9  # Small positive value to prevent log(0)
    
    for layer_name, layer_info in graph.layers.items():
        layer_mapping = mapping_params.get_mapping_by_original_name(layer_name)
        dims = layer_info['dims']
        
        # For each problem dimension
        for dim in Config.get_instance().PROBLEM_DIMS:
            if dim in dims:
                # Get continuous factor (we only have L2 level for now)
                f_continuous = layer_mapping.temporal_factors_L2[dim]
                
                # --- FIX: Add epsilon protection to prevent log(0) or log(negative) ---
                # Ensure f_continuous is positive and add epsilon for stability
                f_continuous_safe = torch.clamp(f_continuous, min=epsilon) + epsilon
                problem_size_safe = max(dims[dim], epsilon) + epsilon
                
                # Calculate log-domain difference with numerical stability
                log_factor = torch.log(f_continuous_safe)
                log_problem_size = torch.log(torch.tensor(problem_size_safe, dtype=torch.float32))
                
                # Square the difference for the penalty
                penalty = (log_factor - log_problem_size) ** 2
                total_penalty = total_penalty + penalty
    
    return total_penalty

def create_example_optimization_setup(graph: ComputationGraph) -> Tuple[MappingParameters, FusionParameters, HardwareParameters]:
    """
    Create an example setup for joint hardware-software co-design optimization.
    
    Args:
        graph: ComputationGraph object defining the network structure
        
    Returns:
        Tuple of (mapping_params, fusion_params, hardware_params) ready for optimization
    """
    # Create learnable parameters for mapping, fusion, and hardware
    mapping_params = MappingParameters(graph)
    fusion_params = FusionParameters(graph)
    
    # Initialize hardware parameters with reasonable starting values
    # These will be jointly optimized with mapping and fusion decisions
    hardware_params = HardwareParameters(
        initial_num_pes=64,           # Start with 64 processing elements
        initial_accumulator_kb=64.0,  # Start with 64 KB accumulator
        initial_scratchpad_kb=192.0   # Start with 192 KB scratchpad
    )
    
    return mapping_params, fusion_params, hardware_params

def create_joint_optimizer(mapping_params: MappingParameters, 
                          fusion_params: FusionParameters,
                          hardware_params: HardwareParameters,
                          lr: float = 0.01) -> torch.optim.Adam:
    """
    Create an Adam optimizer for joint hardware-software co-design.
    
    Args:
        mapping_params: MappingParameters object
        fusion_params: FusionParameters object  
        hardware_params: HardwareParameters object
        lr: Learning rate for optimization
        
    Returns:
        Adam optimizer configured for all parameter types
    """
    # Collect all learnable parameters
    all_params = []
    all_params.extend(mapping_params.parameters())
    all_params.extend(fusion_params.parameters())
    all_params.extend(hardware_params.parameters())
    
    return torch.optim.Adam(all_params, lr=lr)

def calculate_total_loss_with_hardware_constraints(
    performance_model: 'ConditionalPerformanceModel',
    mapping_params: 'MappingParameters',
    fusion_params: 'FusionParameters',
    hardware_params: 'HardwareParameters',
    graph: 'ComputationGraph'
) -> torch.Tensor:
    """
    重构后的总损失函数：使用EDP作为核心优化目标，并将惩罚项统一处理。
    """
    config = Config.get_instance()
    
    # 步骤 1: 计算原始的性能指标 (latency, energy, area)
    latency, energy, area = performance_model(mapping_params, fusion_params, hardware_params, graph)
    
    # 步骤 2: 计算所有约束惩罚的总和
    mapping_penalty = calculate_penalty_loss(mapping_params)  # Tiling因子 > 1 约束
    template_constraint_penalty = calculate_template_constraint_penalty(
        mapping_params, hardware_params, graph
    )  # Buffer 和硬件参数约束
    total_penalty = mapping_penalty + template_constraint_penalty

    # 步骤 3: 构建新的、更稳定的总损失函数
    # 主要优化目标是 EDP (Energy-Delay Product)
    edp = latency * energy
    
    epsilon = 1e-9  # 保证数值稳定，防止 log(0)

    # 在对数域中组合所有成本项
    log_edp = torch.log(edp + epsilon)
    log_area = torch.log(area + epsilon)
    
    # 新的总损失: 以 log(EDP) 为核心，加上加权的 log(Area) 和总惩罚项
    # 注意：我们直接将线性的 total_penalty 加到对数损失上，其影响由 penalty_weight 控制
    total_loss = (
        config.get_weight('edp_weight', 1.0) * log_edp +       # 主要目标
        config.get_weight('area_weight', 0.5) * log_area +     # 次要目标/约束
        config.PENALTY_WEIGHT * total_penalty                # 惩罚项
    )

    return total_loss

if __name__ == "__main__":
    # This script is now intended to be used as a module.
    # The main experiment runner is in run_experiments.py
    # 
    # Example usage for joint hardware-software co-design:
    #
    # # Create computation graph
    # graph = ComputationGraph()
    # # ... add layers and fusion groups ...
    #
    # # Set up optimization
    # mapping_params, fusion_params, hardware_params = create_example_optimization_setup(graph)
    # optimizer = create_joint_optimizer(mapping_params, fusion_params, hardware_params)
    # performance_model = ConditionalPerformanceModel()
    #
    # # Training loop
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     loss = calculate_total_loss_with_hardware_constraints(
    #         performance_model, mapping_params, fusion_params, hardware_params, graph
    #     )
    #     loss.backward()
    #     optimizer.step()
    #
    #     if epoch % 10 == 0:
    #         print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    #         print(f"Hardware Config: {hardware_params}")
    
    pass 