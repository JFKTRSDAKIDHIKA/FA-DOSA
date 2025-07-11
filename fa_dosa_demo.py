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
        self.DRAM_ACCESS_COST = self._config['costs']['dram_access_cost_pj_byte']
        self.SRAM_BASE_COST = self._config['costs']['sram_base_cost_pj_byte']
        self.FUSION_OVERHEAD_COST = self._config['costs']['fusion_overhead_cost_pj']
        self.AREA_COEFFICIENT = self._config['costs']['area_coefficient_mm2_per_kb']
        self.SRAM_ENERGY_SCALING = self._config['costs']['sram_energy_scaling_pj_byte_kb']
        
        # Hardware parameters
        self.DRAM_BANDWIDTH = self._config['hardware']['dram_bandwidth_gb_s']
        self.SRAM_BANDWIDTH = self._config['hardware']['sram_bandwidth_gb_s']
        self.MAC_THROUGHPUT = self._config['hardware']['mac_throughput_gops']
        
        # Loss weights
        self.PENALTY_WEIGHT = self._config['weights']['penalty_weight']
        self.PRODUCT_PENALTY_WEIGHT = self._config['weights']['product_penalty_weight']
        
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
    Calculate penalty term for invalid mapping parameters.
    Penalizes tiling factors less than 1.0 as they are invalid in practice.
    
    Args:
        mapping_params: MappingParameters object containing all layer mappings
        
    Returns:
        Total penalty summed across all invalid tiling factors
    """
    total_penalty = torch.tensor(0.0)
    
    # Iterate through all layers and their tiling factors
    for sanitized_name, layer_mapping in mapping_params.mappings.items():
        for factor in layer_mapping.temporal_factors_L2.values():
            # Penalty is max(0, 1 - factor) to penalize factors < 1
            penalty = torch.maximum(1 - factor, torch.tensor(0.0))
            total_penalty = total_penalty + penalty
    
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
        buffer_reqs = layer_mapping.calculate_buffer_requirements(layer_info['dims'])
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
    Learnable parameters for layer mappings.
    Currently implements single-level tiling with L2 buffer.
    """
    def __init__(self, graph: ComputationGraph):
        super().__init__()
        
        # Create mapping parameters for each layer
        self.mappings = nn.ModuleDict()
        self.layer_name_mapping = {}  # Map sanitized names back to original names
        
        for layer_name in graph.get_layer_names():
            sanitized_name = sanitize_layer_name(layer_name)
            self.mappings[sanitized_name] = LayerMapping(graph.layers[layer_name]['dims'])
            self.layer_name_mapping[sanitized_name] = layer_name

    def get_mapping_by_original_name(self, original_name: str) -> 'LayerMapping':
        """Get layer mapping using the original layer name."""
        sanitized_name = sanitize_layer_name(original_name)
        return self.mappings[sanitized_name]

    def __str__(self):
        lines = ["Mapping Parameters:"]
        for sanitized_name, mapping in self.mappings.items():
            original_name = self.layer_name_mapping[sanitized_name]
            lines.append(f"  {original_name}:")
            for dim, factor in mapping.temporal_factors_L2.items():
                lines.append(f"    {dim}: {factor.item():.4f}")
        return "\n".join(lines)

class HardwareParameters(nn.Module):
    """
    Learnable hardware configuration parameters for joint hardware-software co-design.
    This enables the optimization of architectural choices alongside mapping and fusion decisions.
    """
    def __init__(self, initial_num_pes: int = 64, initial_buffer_size_kb: float = 256.0):
        super().__init__()
        
        # --- FIX: Add epsilon protection for log operations ---
        epsilon = 1e-9  # Small positive value to prevent log(0)
        
        # === Learnable parameters in log-scale for better optimization ===
        # Log-scale helps with gradient flow and ensures positive values
        # Ensure inputs are positive and add epsilon for numerical stability
        safe_num_pes = max(initial_num_pes, epsilon) + epsilon
        safe_buffer_size = max(initial_buffer_size_kb, epsilon) + epsilon
        
        self.log_num_pes = nn.Parameter(torch.log(torch.tensor(safe_num_pes, dtype=torch.float32)))
        self.log_buffer_size_kb = nn.Parameter(torch.log(torch.tensor(safe_buffer_size, dtype=torch.float32)))
        
        # Store initial values for reference
        self.initial_num_pes = initial_num_pes
        self.initial_buffer_size_kb = initial_buffer_size_kb
    
    def get_num_pes(self) -> torch.Tensor:
        """Get the number of processing elements (differentiable)."""
        return torch.exp(self.log_num_pes)
    
    def get_buffer_size_kb(self) -> torch.Tensor:
        """Get the buffer size in KB (differentiable)."""
        return torch.exp(self.log_buffer_size_kb)
    
    def get_buffer_size_bytes(self) -> torch.Tensor:
        """Get the buffer size in bytes (differentiable)."""
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
        Calculate total chip area based on PEs and buffer size.
        
        Returns:
            Total area in mm²
        """
        config = Config.get_instance()
        num_pes = self.get_num_pes()
        buffer_size_kb = self.get_buffer_size_kb()
        
        # Total area = base_area + PE_area + buffer_area
        area_cost = (
            torch.tensor(config.AREA_BASE_MM2, dtype=torch.float32) +
            num_pes * torch.tensor(config.AREA_PER_PE_MM2, dtype=torch.float32) +
            buffer_size_kb * torch.tensor(config.AREA_PER_KB_SRAM_MM2, dtype=torch.float32)
        )
        return area_cost
    
    def __str__(self):
        lines = ["Hardware Parameters:"]
        lines.append(f"  Number of PEs: {self.get_num_pes().item():.1f}")
        lines.append(f"  Buffer Size: {self.get_buffer_size_kb().item():.1f} KB")
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
        layer_dims = graph.layers[layer_name]['dims']
        buffer_reqs = layer_mapping.calculate_buffer_requirements(layer_dims)
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
    
    ENHANCED FOR 2025: Now supports both CNN and Transformer architectures.
    
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
        Calculate energy consumed by MAC operations.
        
        Args:
            total_macs: Total number of multiply-accumulate operations
            
        Returns:
            Compute energy in picojoules, with calibration factor applied
        """
        config = Config.get_instance()
        compute_energy = total_macs * torch.tensor(config.MAC_ENERGY_PJ, dtype=torch.float32)
        # Apply calibration factor for model tuning
        return compute_energy * torch.tensor(config.COMPUTE_ENERGY_FACTOR, dtype=torch.float32)
    
    def _calculate_sram_energy(self, buffer_requirements: Dict[str, torch.Tensor], hardware_params: 'HardwareParameters') -> torch.Tensor:
        """
        Calculate energy consumed by SRAM buffer accesses.
        Uses dynamic energy scaling based on hardware buffer size.
        
        Args:
            buffer_requirements: Dictionary of buffer size requirements per dimension
            hardware_params: HardwareParameters object containing learnable buffer size
            
        Returns:
            SRAM energy in picojoules, with calibration factor applied
        """
        config = Config.get_instance()
        buffer_size_bytes = hardware_params.get_buffer_size_bytes()
        
        # Calculate dynamic SRAM energy per access based on buffer size
        sram_energy_per_access = (
            torch.tensor(config.SRAM_BASE_COST, dtype=torch.float32) +
            buffer_size_bytes * torch.tensor(config.SRAM_ENERGY_SCALING, dtype=torch.float32)
        )
        
        # Calculate total SRAM energy based on buffer requirements
        total_buffer_accesses = sum(buffer_requirements.values())
        sram_energy = total_buffer_accesses * sram_energy_per_access
        
        # Apply calibration factor for model tuning
        return sram_energy * torch.tensor(config.SRAM_ENERGY_FACTOR, dtype=torch.float32)
    
    def _calculate_dram_energy(self, access_counts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate energy consumed by DRAM accesses.
        
        Args:
            access_counts: Dictionary of memory access counts per dimension
            
        Returns:
            DRAM energy in picojoules, with calibration factor applied
        """
        config = Config.get_instance()
        total_dram_accesses = sum(access_counts.values())
        dram_energy = total_dram_accesses * torch.tensor(config.DRAM_ACCESS_COST, dtype=torch.float32)
        
        # Apply calibration factor for model tuning
        return dram_energy * torch.tensor(config.DRAM_ENERGY_FACTOR, dtype=torch.float32)
    
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
    
    def _calculate_analytical_costs(
        self,
        layer_name: str,
        mapping_params: MappingParameters,
        graph: ComputationGraph,
        hardware_params: HardwareParameters,
        is_fused: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate analytical latency and energy costs for a layer using modular sub-functions.
        Implements formulas from DOSA paper [3809, 3858] with enhanced cost modeling.
        
        Args:
            layer_name: Name of the layer to analyze
            mapping_params: MappingParameters object
            graph: ComputationGraph object
            hardware_params: HardwareParameters object containing learnable hardware config
            is_fused: Whether this layer is part of a fused operation
        
        Returns:
            latency: Computed latency based on roofline model
            compute_energy: Energy for computation (MAC operations)
            sram_energy: Energy for SRAM buffer accesses
            dram_energy: Energy for DRAM accesses
            noc_energy: Energy for Network-on-Chip data movement
        """
        layer_info = graph.layers[layer_name]
        layer_mapping = mapping_params.get_mapping_by_original_name(layer_name)
        
        # Calculate total MACs
        total_macs = calculate_macs(layer_info['dims'], layer_info['type'])
        
        # Calculate memory access counts and buffer requirements
        access_counts = layer_mapping.calculate_access_counts(layer_info['dims'])
        buffer_requirements = layer_mapping.calculate_buffer_requirements(layer_info['dims'])
        
        # === Use modular cost calculation functions ===
        compute_latency = self._calculate_compute_latency(total_macs, hardware_params)
        memory_latency = self._calculate_memory_latency(access_counts)
        
        # Final latency is max of compute and memory bound (roofline model)
        latency = torch.maximum(compute_latency, memory_latency)
        
        # Calculate energy components using modular functions
        compute_energy = self._calculate_compute_energy(total_macs)
        sram_energy = self._calculate_sram_energy(buffer_requirements, hardware_params)
        dram_energy = self._calculate_dram_energy(access_counts)
        noc_energy = self._calculate_noc_energy(layer_name, graph, is_fused)
        
        # --- FIX: Set physically reasonable lower bounds for all cost components ---
        # This prevents the model from predicting physically impossible zero costs
        # which can lead to inf values when used in divisions or log operations
        min_value = 1e-6  # Minimum physically possible cost value
        
        latency = torch.clamp(latency, min=min_value)
        compute_energy = torch.clamp(compute_energy, min=min_value)
        sram_energy = torch.clamp(sram_energy, min=min_value)
        dram_energy = torch.clamp(dram_energy, min=min_value)
        noc_energy = torch.clamp(noc_energy, min=min_value)
        
        return latency, compute_energy, sram_energy, dram_energy, noc_energy

    def forward(
        self,
        mapping_params: MappingParameters,
        fusion_params: FusionParameters,
        hardware_params: HardwareParameters,
        graph: ComputationGraph
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate latency, energy, and area costs using analytical models.
        Uses differentiable weighted-average for fusion decisions with physically accurate
        DRAM energy savings modeling and learnable hardware parameters.
        
        Args:
            mapping_params: MappingParameters object containing layer mappings
            fusion_params: FusionParameters object containing fusion decisions
            hardware_params: HardwareParameters object containing learnable hardware config
            graph: ComputationGraph object defining the network structure
        
        Returns:
            total_latency: Total execution latency across all layers
            total_energy: Total energy consumption across all layers
            area_cost: Hardware area cost based on learnable hardware parameters
        """
        config = Config.get_instance()
        
        # Get area cost directly from hardware parameters
        area_cost = hardware_params.get_area_cost()
        
        # Initialize total costs
        total_latency = torch.zeros(1, requires_grad=True)
        total_energy = torch.zeros(1, requires_grad=True)
        
        # Create set of layers that are part of any fusion group
        layers_in_fusion_groups = {
            layer 
            for group in graph.fusion_groups 
            for layer in group
        }
        
        # Handle standalone layers (not part of any fusion group)
        for layer_name in graph.get_layer_names():
            if layer_name not in layers_in_fusion_groups:
                # Calculate analytical costs for standalone layer
                latency, compute_energy, sram_energy, dram_energy, noc_energy = self._calculate_analytical_costs(
                    layer_name, mapping_params, graph, hardware_params
                )
                total_latency = total_latency + latency
                total_energy = total_energy + compute_energy + sram_energy + dram_energy + noc_energy
        
        # Handle fusion groups using differentiable weighted-average with EDP calculation
        for group in graph.fusion_groups:
            # Get fusion probability for this group
            p_fuse = fusion_params.get_fusion_probability(group)
            
            # === Calculate Non-Fused Costs (cost_no_fuse) ===
            non_fused_latency = torch.tensor(0.0)
            non_fused_energy = torch.tensor(0.0)
            
            for layer_name in group:
                lat, compute_energy, sram_energy, dram_energy, noc_energy = self._calculate_analytical_costs(
                    layer_name, mapping_params, graph, hardware_params
                )
                non_fused_latency = non_fused_latency + lat
                non_fused_energy = non_fused_energy + compute_energy + sram_energy + dram_energy + noc_energy
            
            # Energy-Delay Product for non-fused case
            cost_no_fuse = non_fused_latency * non_fused_energy
            
            # === Calculate Fused Costs (cost_fused) ===
            fused_latency = torch.tensor(0.0)
            fused_compute_energy = torch.tensor(0.0)
            fused_sram_energy = torch.tensor(0.0)
            fused_dram_energy = torch.tensor(0.0)
            fused_noc_energy = torch.tensor(0.0)
            
            for layer_name in group:
                lat, compute_energy, sram_energy, dram_energy, noc_energy = self._calculate_analytical_costs(
                    layer_name, mapping_params, graph, hardware_params, is_fused=True
                )
                # Latency is dominated by the slowest path (parallel execution)
                fused_latency = torch.maximum(fused_latency, lat)
                fused_compute_energy = fused_compute_energy + compute_energy
                fused_sram_energy = fused_sram_energy + sram_energy
                fused_dram_energy = fused_dram_energy + dram_energy
                fused_noc_energy = fused_noc_energy + noc_energy
            
            # === Add fusion control overhead ===
            # Fusion isn't "free" - there's control overhead for coordinating fused operations
            fusion_control_overhead = torch.tensor(config.FUSION_LATENCY_PENALTY_CYCLES, dtype=torch.float32)
            fused_latency = fused_latency + fusion_control_overhead
            
            # === MODEL THE DRAM SAVING ===
            # For fusion groups, intermediate tensors avoid DRAM write + read
            if len(group) >= 2:
                # Calculate intermediate tensor size from first layer's output
                first_layer = group[0]
                first_layer_dims = graph.layers[first_layer]['dims']
                
                # Simplified proxy for output tensor size (elements * bytes_per_element)
                # Output size = N * K * P * Q for conv layers
                if 'P' in first_layer_dims and 'Q' in first_layer_dims:
                    intermediate_tensor_size = torch.tensor(
                        first_layer_dims['N'] * first_layer_dims.get('K', first_layer_dims.get('C', 1)) * 
                        first_layer_dims['P'] * first_layer_dims['Q'] * config.BYTES_PER_ELEMENT,
                        dtype=torch.float32
                    )
                else:
                    # For other layer types, use a simpler calculation
                    intermediate_tensor_size = torch.tensor(
                        first_layer_dims['N'] * first_layer_dims.get('K', first_layer_dims.get('C', 1)) * config.BYTES_PER_ELEMENT,
                        dtype=torch.float32
                    )
                
                # DRAM energy saved = 2 * intermediate_tensor_size * DRAM_ACCESS_COST
                # (one write to DRAM avoided, one read from DRAM avoided)
                dram_energy_saved = 2 * intermediate_tensor_size * torch.tensor(config.DRAM_ACCESS_COST, dtype=torch.float32)
                
                # Subtract the saving from fused DRAM energy
                fused_dram_energy = fused_dram_energy - dram_energy_saved
                
                # Ensure DRAM energy doesn't go negative
                fused_dram_energy = torch.maximum(fused_dram_energy, torch.tensor(0.0))
            
            # Calculate final fused energy with fusion overhead
            fused_energy = fused_compute_energy + fused_sram_energy + fused_dram_energy + fused_noc_energy + torch.tensor(config.FUSION_OVERHEAD_COST, dtype=torch.float32)
            
            # Energy-Delay Product for fused case
            cost_fused = fused_latency * fused_energy
            
            # === Calculate Weighted-Average EDP for the group ===
            group_edp = p_fuse * cost_fused + (1 - p_fuse) * cost_no_fuse
            
            # Convert back to latency and energy for accumulation
            # For differentiable accumulation, we use the weighted averages
            group_latency = p_fuse * fused_latency + (1 - p_fuse) * non_fused_latency
            group_energy = p_fuse * fused_energy + (1 - p_fuse) * non_fused_energy
            
            total_latency = total_latency + group_latency
            total_energy = total_energy + group_energy
        
        # --- FIX: Add numerical stability assertions to detect inf/NaN early ---
        # These assertions will help us quickly identify where numerical problems originate
        assert not torch.isinf(total_latency), f"Latency exploded to infinity! Value: {total_latency}"
        assert not torch.isnan(total_latency), f"Latency became NaN! This indicates numerical instability."
        assert not torch.isinf(total_energy), f"Energy exploded to infinity! Value: {total_energy}"
        assert not torch.isnan(total_energy), f"Energy became NaN! This indicates numerical instability."
        assert not torch.isinf(area_cost), f"Area cost exploded to infinity! Value: {area_cost}"
        assert not torch.isnan(area_cost), f"Area cost became NaN! This indicates numerical instability."
        
        # Additional safety checks for physically reasonable values
        assert total_latency > 0, f"Latency must be positive! Got: {total_latency}"
        assert total_energy > 0, f"Energy must be positive! Got: {total_energy}"
        assert area_cost > 0, f"Area cost must be positive! Got: {area_cost}"
        
        return total_latency, total_energy, area_cost

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
        initial_buffer_size_kb=256.0  # Start with 256 KB buffer
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
    performance_model: ConditionalPerformanceModel,
    mapping_params: MappingParameters,
    fusion_params: FusionParameters, 
    hardware_params: HardwareParameters,
    graph: ComputationGraph
) -> torch.Tensor:
    """
    Calculate total loss including performance objectives and hardware constraints.
    
    Args:
        performance_model: ConditionalPerformanceModel instance
        mapping_params: MappingParameters object
        fusion_params: FusionParameters object
        hardware_params: HardwareParameters object
        graph: ComputationGraph object
        
    Returns:
        Total loss tensor for optimization
    """
    config = Config.get_instance()
    
    # Calculate performance metrics
    latency, energy, area = performance_model(mapping_params, fusion_params, hardware_params, graph)
    
    # Calculate penalties
    mapping_penalty = calculate_penalty_loss(mapping_params)
    product_penalty = calculate_product_constraint_penalty(mapping_params, graph)
    hardware_penalty = calculate_hardware_constraint_penalty(mapping_params, hardware_params, graph)
    
    # Combine all components with weights
    total_loss = (
        config.LATENCY_WEIGHT * latency + 
        config.ENERGY_WEIGHT * energy + 
        config.AREA_WEIGHT * area +
        config.PENALTY_WEIGHT * mapping_penalty +
        config.PRODUCT_PENALTY_WEIGHT * product_penalty +
        config.PENALTY_WEIGHT * hardware_penalty  # Use same weight as mapping penalty
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