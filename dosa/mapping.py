import torch
import torch.nn as nn
from typing import Dict, List

class ProjectToNearestDivisor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, continuous_factor, problem_dim):
        # Clamp the continuous factor to a minimum of 1.0
        continuous_factor = torch.clamp(continuous_factor, min=1.0)
        # Project to the nearest integer divisor
        divisors = torch.arange(1, problem_dim.item() + 1, device=continuous_factor.device)
        valid_divisors = divisors[problem_dim % divisors == 0]
        
        # Find the closest valid divisor
        abs_diff = torch.abs(valid_divisors - continuous_factor)
        nearest_divisor = valid_divisors[torch.argmin(abs_diff)]
        
        return nearest_divisor

    @staticmethod
    def backward(ctx, grad_output):
        # Pass the gradient through, ensuring correct shape
        # grad_output might be a scalar, but we need to match input shape
        if grad_output.numel() == 1 and grad_output.dim() == 0:
            grad_output = grad_output.unsqueeze(0)
        return grad_output, None

class FineGrainedMapping(nn.Module):
    """
    NEW: 细粒度的映射参数化模块，替代了旧的LearnableConvReluTemplate。
    """
    def __init__(self, problem_dims: Dict[str, int], hierarchy: List[Dict]):
        super().__init__()
        self.dims = problem_dims
        self.hierarchy = hierarchy
        
        # 创建一个嵌套的参数字典来存储所有tiling因子
        # 结构: self.factors[level_name][dim_name]['temporal' or 'spatial']
        self.factors = nn.ModuleDict()

        # 只为片上存储（on-chip buffers）创建可学习的参数
        on_chip_levels = [level['name'] for level in hierarchy if level['type'] == 'buffer']
        
        for level_name in on_chip_levels:
            self.factors[level_name] = nn.ModuleDict()
            for dim_name in self.dims.keys():
                # 使用ParameterDict来正确注册参数
                # 初始化为1（在log空间中为0）
                self.factors[level_name][dim_name] = nn.ParameterDict({
                    'temporal': nn.Parameter(torch.zeros(1)), # log(1) = 0
                    'spatial': nn.Parameter(torch.zeros(1))
                })

    def get_factor(self, level_name, dim_name, factor_type):
        """获取指定level, dim, type的tiling因子。"""
        return torch.clamp(torch.exp(self.factors[level_name][dim_name][factor_type]), min=1.0)

    def get_all_factors(self):
        """
        NEW: Returns physically valid, integer tiling factors using differentiable projection.
        This method replaces get_all_factors() for performance evaluation.
        """
        projected_factors = {}
        on_chip_levels = [level['name'] for level in self.hierarchy if level['type'] == 'buffer']

        for dim_name, total_size in self.dims.items():
            projected_factors[dim_name] = {}
            product_of_on_chip_factors = 1.0
            
            # Project on-chip factors to valid integer divisors
            for level_name in on_chip_levels:
                continuous_temporal = self.get_factor(level_name, dim_name, 'temporal')
                continuous_spatial = self.get_factor(level_name, dim_name, 'spatial')
                
                # Apply differentiable projection
                problem_dim_tensor = torch.tensor(float(total_size))
                projected_temporal = ProjectToNearestDivisor.apply(continuous_temporal, problem_dim_tensor)
                projected_spatial = ProjectToNearestDivisor.apply(continuous_spatial, problem_dim_tensor)
                
                projected_factors[dim_name][level_name] = {
                    'temporal': projected_temporal,
                    'spatial': projected_spatial
                }
                product_of_on_chip_factors *= projected_temporal * projected_spatial

            # Handle DRAM level factors
            dram_level = next((level for level in self.hierarchy if level['type'] == 'dram'), None)
            if dram_level:
                dram_level_name = dram_level['name']
                # The temporal factor at DRAM is what's left over from the on-chip factors
                dram_temporal_factor = total_size / product_of_on_chip_factors
                projected_dram_temporal = ProjectToNearestDivisor.apply(dram_temporal_factor, torch.tensor(float(total_size)))

                projected_factors[dim_name][dram_level_name] = {
                    'temporal': projected_dram_temporal,
                    'spatial': torch.tensor(1.0) # No spatial tiling in DRAM
                }

        return projected_factors