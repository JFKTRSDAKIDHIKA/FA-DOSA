# [CURSOR_START]
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any

# --- Prerequisite: Assuming necessary classes are available ---
# Stubs for Config, HardwareParameters, FusionParameters, etc. are included for completeness.

class Config:
    _instance = None
    def __init__(self):
        self.BYTES_PER_ELEMENT = 4
        self.DRAM_BANDWIDTH_GB_S = 128
        self.CLOCK_FREQUENCY_MHZ = 1000
        
        # --- 基于40nm工艺的Energy Per Access (EPA) 数据表 ---
        # 单位: μJ (微焦耳) -> 转换为 pJ (皮焦耳) 用于计算
        
        # PE: 一次乘加运算 (MAC) 的能耗
        self.PE_MAC_EPA_PJ = 0.561 * 1e6  # 0.561 μJ -> 561000 pJ
        
        # Registers: 访问一次寄存器的能耗
        self.REGISTER_EPA_PJ = 0.487 * 1e6  # 0.487 μJ -> 487000 pJ
        
        # Accumulator: 容量相关的能耗公式
        # EPA = 1.94 + 0.1005 × (C₁ / √CPE) μJ
        # 其中 C₁ 是accumulator容量，CPE是每个PE的容量
        self.ACCUMULATOR_BASE_EPA_PJ = 1.94 * 1e6  # 1.94 μJ -> 1940000 pJ
        self.ACCUMULATOR_CAPACITY_COEFF_PJ = 0.1005 * 1e6  # 0.1005 μJ -> 100500 pJ
        
        # Scratchpad: 容量相关的能耗公式
        # EPA = 0.49 + 0.025 × C₂ μJ
        # 其中 C₂ 是scratchpad容量 (KB)
        self.SCRATCHPAD_BASE_EPA_PJ = 0.49 * 1e6  # 0.49 μJ -> 490000 pJ
        self.SCRATCHPAD_CAPACITY_COEFF_PJ_PER_KB = 0.025 * 1e6  # 0.025 μJ/KB -> 25000 pJ/KB
        
        # DRAM: 访问一次DRAM的能耗
        self.DRAM_EPA_PJ = 100 * 1e6  # 100 μJ -> 100000000 pJ
        
        # 硬件配置参数
        self.ACCUMULATOR_CAPACITY_PER_PE_KB = 0.5  # 每个PE的accumulator容量 (KB)
        self.REGISTER_CAPACITY_PER_PE_KB = 0.1     # 每个PE的register容量 (KB)
        
        self.AREA_PER_PE_MM2 = 0.015
        self.AREA_PER_KB_SRAM_MM2 = 0.005
        self.AREA_BASE_MM2 = 1.0
        self.PENALTY_WEIGHT = 1e5
    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance

class HardwareParameters(nn.Module):
    def __init__(self, initial_num_pes=64, initial_buffer_kb=128):
        super().__init__()
        self.log_num_pes = nn.Parameter(torch.log(torch.tensor(float(initial_num_pes))))
        self.log_buffer_size_kb = nn.Parameter(torch.log(torch.tensor(float(initial_buffer_kb))))
    def get_num_pes(self):
        return torch.exp(self.log_num_pes)
    def get_buffer_size_kb(self):
        return torch.exp(self.log_buffer_size_kb)
    def get_buffer_size_bytes(self):
        return self.get_buffer_size_kb() * 1024
    def get_area_cost(self):
        config = Config.get_instance()
        return (config.AREA_BASE_MM2 +
                self.get_num_pes() * config.AREA_PER_PE_MM2 +
                self.get_buffer_size_kb() * config.AREA_PER_KB_SRAM_MM2)

class FusionParameters(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.fusion_probs = nn.ParameterDict()
        for group in graph.fusion_groups:
            group_key = '__'.join(sorted(group))
            self.fusion_probs[group_key] = nn.Parameter(torch.randn(1))
    def get_fusion_probability(self, group):
        group_key = '__'.join(sorted(group))
        return torch.sigmoid(self.fusion_probs[group_key]).squeeze()

class ComputationGraph:
    def __init__(self):
        self.layers = {}
        self.edges = []
        self.fusion_groups = []
    def add_layer(self, name, dims, type):
        self.layers[name] = {'dims': dims, 'type': type}
    def add_edge(self, src, dest):
        self.edges.append((src, dest))
    def add_fusion_group(self, group):
        self.fusion_groups.append(group)

def parse_onnx_to_graph(path):
    graph = ComputationGraph()
    for i in range(8):
        dims_conv = {'N': 1, 'C': 64*(2**(i//2)), 'K': 64*(2**(i//2)), 'P': 56//(2**(i//2)), 'Q': 56//(2**(i//2)), 'R': 3, 'S': 3}
        dims_relu = {k:v for k,v in dims_conv.items() if k not in ['C', 'R', 'S']}
        conv_name = f'conv_{i}'
        relu_name = f'relu_{i}'
        graph.add_layer(conv_name, dims_conv, 'Conv')
        graph.add_layer(relu_name, dims_relu, 'ReLU')
        graph.add_edge(conv_name, relu_name)
        graph.add_fusion_group([conv_name, relu_name])
    return graph

def calculate_macs(dims, op_type):
    return torch.tensor(float(dims.get('N', 1) * dims.get('K', 1) * dims.get('C', 1) * dims.get('P', 1) * dims.get('Q', 1) * dims.get('R', 1) * dims.get('S', 1)))

class DifferentiableMappingTemplate(nn.Module):
    def __init__(self, dims: Dict[str, int]):
        super().__init__()
        self.dims = dims
        self.config = Config.get_instance()
        self._init_template_parameters()
    def _init_template_parameters(self): pass
    def get_M0(self) -> torch.Tensor: raise NotImplementedError
    def get_K0(self) -> torch.Tensor: raise NotImplementedError
    def get_N0(self) -> torch.Tensor: raise NotImplementedError
    def get_buffer_requirements(self) -> Dict[str, torch.Tensor]: raise NotImplementedError
    def get_access_counts(self, is_fused: bool) -> Dict[str, torch.Tensor]: raise NotImplementedError

class LearnableConvReluTemplate(DifferentiableMappingTemplate):
    def __init__(self, dims: Dict[str, int]):
        super().__init__(dims)
    def _init_template_parameters(self):
        M_total = self.dims.get('N', 1) * self.dims.get('P', 1) * self.dims.get('Q', 1)
        K_total = self.dims.get('C', 1)
        N_total = self.dims.get('K', 1)
        self.log_M0 = nn.Parameter(torch.log(torch.sqrt(torch.tensor(float(M_total)))))
        self.log_K0 = nn.Parameter(torch.log(torch.sqrt(torch.tensor(float(K_total)))))
        self.log_N0 = nn.Parameter(torch.log(torch.sqrt(torch.tensor(float(N_total)))))
    def get_M0(self) -> torch.Tensor: return torch.exp(self.log_M0)
    def get_K0(self) -> torch.Tensor: return torch.exp(self.log_K0)
    def get_N0(self) -> torch.Tensor: return torch.exp(self.log_N0)
    def get_buffer_requirements(self) -> Dict[str, torch.Tensor]:
        M0, K0, N0 = self.get_M0(), self.get_K0(), self.get_N0()
        return {'input': M0 * K0 * self.config.BYTES_PER_ELEMENT, 'weight': K0 * N0 * self.config.BYTES_PER_ELEMENT, 'output': M0 * N0 * self.config.BYTES_PER_ELEMENT}
    def get_access_counts(self, is_fused: bool) -> Dict[str, torch.Tensor]:
        M0, K0, N0 = self.get_M0(), self.get_K0(), self.get_N0()
        M_total, K_total, N_total = self.dims.get('N', 1)*self.dims.get('P', 1)*self.dims.get('Q', 1), self.dims.get('C', 1), self.dims.get('K', 1)
        M1, K1, N1 = M_total / M0, K_total / K0, N_total / N0
        input_access, weight_access, output_access = M_total*K_total*N1, K_total*N_total*M1, M_total*N_total
        return {'input': input_access, 'weight': weight_access, 'output': output_access * (1 if is_fused else 2)}

class MappingParameters(nn.Module):
    def __init__(self, graph: ComputationGraph):
        super().__init__()
        self.expert_templates = {}
        for group in graph.fusion_groups:
            if len(group) == 2 and graph.layers[group[0]]['type'] == 'Conv' and graph.layers[group[1]]['type'] == 'ReLU':
                template = LearnableConvReluTemplate(graph.layers[group[0]]['dims'])
                self.expert_templates['__'.join(sorted(group))] = template
                # Register the template as a submodule so its parameters are tracked
                self.add_module(f'template_{len(self.expert_templates)}', template)
    def get_expert_template(self, group: List[str]):
        return self.expert_templates.get('__'.join(sorted(group)))
    def parameters(self):
        params = []
        for template in self.expert_templates.values():
            params.extend(template.parameters())
        return iter(params)

class RealisticPerformanceModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
    def forward(self, graph: ComputationGraph, hardware_params: HardwareParameters, fusion_params: FusionParameters, mapping_params: MappingParameters) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_latency = torch.tensor(0.0)
        total_energy = torch.tensor(0.0)
        for group in graph.fusion_groups:
            template = mapping_params.get_expert_template(group)
            if template:
                latency, energy = self._calculate_expert_template_performance(group, template, hardware_params, fusion_params)
                total_latency += latency
                total_energy += energy
        return total_latency, total_energy, hardware_params.get_area_cost()
    def _calculate_expert_template_performance(self, group: List[str], template: LearnableConvReluTemplate, hardware_params: HardwareParameters, fusion_params: FusionParameters) -> Tuple[torch.Tensor, torch.Tensor]:
        p_fuse = fusion_params.get_fusion_probability(group)
        costs = {}
        for case in ['fused', 'non_fused']:
            is_fused = (case == 'fused')
            macs = calculate_macs(template.dims, 'Conv')
            gops = hardware_params.get_num_pes() * (self.config.CLOCK_FREQUENCY_MHZ / 1000.0)
            compute_latency = macs / (gops * 1e9)
            access_counts = template.get_access_counts(is_fused)
            total_bytes_to_dram = sum(access_counts.values()) * self.config.BYTES_PER_ELEMENT
            memory_latency = total_bytes_to_dram / (self.config.DRAM_BANDWIDTH_GB_S * 1e9)
            latency = torch.maximum(compute_latency, torch.tensor(memory_latency, dtype=torch.float32))
            
            # --- 基于40nm工艺数据表的精确能耗计算 ---
            
            # 1. PE能耗: MAC运算
            pe_energy = macs * self.config.PE_MAC_EPA_PJ
            
            # 2. Register能耗: 基于访问次数
            # 假设每个MAC需要访问2个寄存器 (输入和权重)
            register_accesses = macs * 2
            register_energy = register_accesses * self.config.REGISTER_EPA_PJ
            
            # 3. Accumulator能耗: 容量相关公式
            # EPA = 1.94 + 0.1005 × (C₁ / √CPE) μJ
            # 其中 C₁ 是总accumulator容量，CPE是每个PE的容量
            total_accumulator_capacity_kb = hardware_params.get_num_pes() * self.config.ACCUMULATOR_CAPACITY_PER_PE_KB
            cpe_kb = self.config.ACCUMULATOR_CAPACITY_PER_PE_KB
            accumulator_epa_pj = self.config.ACCUMULATOR_BASE_EPA_PJ + \
                                self.config.ACCUMULATOR_CAPACITY_COEFF_PJ * (total_accumulator_capacity_kb / torch.sqrt(torch.tensor(cpe_kb, dtype=torch.float32)))
            # 假设每个MAC需要一次accumulator访问
            accumulator_energy = macs * accumulator_epa_pj
            
            # 4. Scratchpad能耗: 容量相关公式
            # EPA = 0.49 + 0.025 × C₂ μJ
            # 其中 C₂ 是scratchpad容量 (KB)
            scratchpad_capacity_kb = hardware_params.get_buffer_size_kb()
            scratchpad_epa_pj = self.config.SCRATCHPAD_BASE_EPA_PJ + \
                               self.config.SCRATCHPAD_CAPACITY_COEFF_PJ_PER_KB * scratchpad_capacity_kb
            
            # 计算scratchpad访问次数 (基于buffer requirements)
            buffer_req_bytes = sum(template.get_buffer_requirements().values())
            buffer_req_kb = buffer_req_bytes / 1024.0
            # 假设每个tile需要一次scratchpad访问
            scratchpad_accesses = buffer_req_kb / scratchpad_capacity_kb * macs  # 简化的访问模型
            scratchpad_energy = scratchpad_accesses * scratchpad_epa_pj
            
            # 5. DRAM能耗: 基于访问次数
            dram_accesses = total_bytes_to_dram / self.config.BYTES_PER_ELEMENT  # 按元素计算访问次数
            dram_energy = dram_accesses * self.config.DRAM_EPA_PJ
            
            # 总能耗
            energy = pe_energy + register_energy + accumulator_energy + scratchpad_energy + dram_energy
            
            penalty_multiplier = torch.relu(buffer_req_bytes / hardware_params.get_buffer_size_bytes() - 1.0) * self.config.PENALTY_WEIGHT + 1.0
            costs[case] = {'latency': latency * penalty_multiplier, 'energy': energy * penalty_multiplier}
        latency = p_fuse * costs['fused']['latency'] + (1 - p_fuse) * costs['non_fused']['latency']
        energy = p_fuse * costs['fused']['energy'] + (1 - p_fuse) * costs['non_fused']['energy']
        return latency, energy

# --- NEW: Configuration Saving and Comparison Table ---

def save_configuration_to_file(method_name: str, results: Dict[str, Any]):
    """Saves the detailed configuration of a method to a text file."""
    filename = f"{method_name.replace(' ', '_').lower()}_config.txt"
    with open(filename, 'w') as f:
        f.write(f"--- Configuration Report for: {method_name} ---\n\n")
        
        # Performance Metrics
        f.write("=== Performance Metrics ===\n")
        f.write(f"  EDP (pJ*s): {results['EDP']:.4e}\n")
        f.write(f"  Latency (s): {results['Latency']:.4e}\n")
        f.write(f"  Energy (pJ): {results['Energy']:.4e}\n\n")

        # Hardware Configuration
        f.write("=== Hardware Configuration ===\n")
        hw_params = results['hardware_params']
        f.write(f"  Processing Elements (PEs): {hw_params.get_num_pes().item():.0f}\n")
        f.write(f"  On-chip Buffer (KB): {hw_params.get_buffer_size_kb().item():.2f}\n")
        f.write(f"  Total Area (mm²): {hw_params.get_area_cost().item():.2f}\n\n")
        
        # 40nm工艺能耗模型参数
        f.write("=== 40nm工艺能耗模型参数 ===\n")
        config = Config.get_instance()
        f.write(f"  PE MAC EPA: {config.PE_MAC_EPA_PJ/1e6:.3f} μJ\n")
        f.write(f"  Register EPA: {config.REGISTER_EPA_PJ/1e6:.3f} μJ\n")
        f.write(f"  Accumulator Base EPA: {config.ACCUMULATOR_BASE_EPA_PJ/1e6:.3f} μJ\n")
        f.write(f"  Accumulator Capacity Coeff: {config.ACCUMULATOR_CAPACITY_COEFF_PJ/1e6:.3f} μJ\n")
        f.write(f"  Scratchpad Base EPA: {config.SCRATCHPAD_BASE_EPA_PJ/1e6:.3f} μJ\n")
        f.write(f"  Scratchpad Capacity Coeff: {config.SCRATCHPAD_CAPACITY_COEFF_PJ_PER_KB/1e6:.3f} μJ/KB\n")
        f.write(f"  DRAM EPA: {config.DRAM_EPA_PJ/1e6:.0f} μJ\n")
        f.write(f"  Accumulator Capacity per PE: {config.ACCUMULATOR_CAPACITY_PER_PE_KB:.1f} KB\n")
        f.write(f"  Register Capacity per PE: {config.REGISTER_CAPACITY_PER_PE_KB:.1f} KB\n\n")

        # Fusion Decisions
        f.write("=== Fusion Decisions (p_fuse > 0.5) ===\n")
        fusion_params = results['fusion_params']
        for group in results['graph'].fusion_groups:
            prob = fusion_params.get_fusion_probability(group).item()
            if prob > 0.5:
                f.write(f"  - FUSED: {' -> '.join(group)} (Prob: {prob:.4f})\n")
        f.write("\n")

        # Mapping Parameters (Tiling Factors)
        f.write("=== Mapping Tiling Factors (from Learnable Expert Templates) ===\n")
        mapping_params = results['mapping_params']
        for group, template in mapping_params.expert_templates.items():
            f.write(f"  Group: {' -> '.join(group)}\n")
            f.write(f"    - M0 (Feature Map Tile): {template.get_M0().item():.2f}\n")
            f.write(f"    - K0 (Input Channel Tile): {template.get_K0().item():.2f}\n")
            f.write(f"    - N0 (Output Channel Tile): {template.get_N0().item():.2f}\n")
            
    print(f"Configuration for {method_name} saved to {filename}")

def generate_comparison_table(result1: Dict[str, Any], result2: Dict[str, Any]):
    """Prints a markdown table comparing the configurations of two methods."""
    hw1, hw2 = result1['hardware_params'], result2['hardware_params']
    
    table = "| Parameter | {r1_name} | {r2_name} |\n".format(r1_name=result1['Method'], r2_name=result2['Method'])
    table += "|:---|:---:|:---:|\n"
    table += f"| **EDP (pJ*s)** | {result1['EDP']:.3e} | {result2['EDP']:.3e} |\n"
    table += f"| **Latency (s)** | {result1['Latency']:.3e} | {result2['Latency']:.3e} |\n"
    table += f"| **Energy (pJ)** | {result1['Energy']:.3e} | {result2['Energy']:.3e} |\n"
    table += "|---|---|---|\n"
    table += f"| **Area (mm²)** | {hw1.get_area_cost().item():.2f} | {hw2.get_area_cost().item():.2f} |\n"
    table += f"| **PEs** | {hw1.get_num_pes().item():.0f} | {hw2.get_num_pes().item():.0f} |\n"
    table += f"| **Buffer (KB)** | {hw1.get_buffer_size_kb().item():.2f} | {hw2.get_buffer_size_kb().item():.2f} |\n"
    
    print("\n\n" + "="*60)
    print("                 CONFIGURATION COMPARISON")
    print("="*60)
    print(table)

# --- Experiment Functions (Modified to return all params) ---

def run_expert_fa_dosa_experiment(graph: ComputationGraph, config: Config, num_iterations=200) -> dict:
    print("\nRunning Fusion-aware DOSA (Learnable Expert Template) experiment...")
    hardware_params = HardwareParameters(initial_num_pes=64, initial_buffer_kb=128)
    fusion_params = FusionParameters(graph)
    mapping_params = MappingParameters(graph)
    perf_model = RealisticPerformanceModel(config)
    
    all_params = list(hardware_params.parameters()) + list(fusion_params.parameters()) + list(mapping_params.parameters())
    optimizer = optim.Adam(all_params, lr=1e-2)

    for i in range(num_iterations):
        optimizer.zero_grad()
        latency, energy, area = perf_model(graph, hardware_params, fusion_params, mapping_params)
        loss = torch.log(latency + 1e-12) + torch.log(energy + 1e-12) + 0.01 * area
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print(f"[FA-DOSA] Iter {i}: Loss={loss.item():.4f}, Latency={latency.item():.2e}s, Energy={energy.item():.2e}pJ, Area={area.item():.2f}mm²")

    with torch.no_grad():
        for param in fusion_params.parameters():
            param.data = (param.data > 0).float() * 10.0 - (param.data <= 0).float() * 10.0
        final_latency, final_energy, final_area = perf_model(graph, hardware_params, fusion_params, mapping_params)
    
    return {
        'Method': 'Fusion-aware DOSA',
        'EDP': final_latency.item() * final_energy.item(),
        'Latency': final_latency.item(),
        'Energy': final_energy.item(),
        'Area': final_area.item(),
        'hardware_params': hardware_params,
        'fusion_params': fusion_params,
        'mapping_params': mapping_params,
        'graph': graph
    }

def run_decoupled_sota_experiment(graph: ComputationGraph, config: Config) -> dict:
    print("\nRunning Decoupled SOTA baseline experiment...")
    buffer_sizes_kb = [64, 128, 256, 512]
    pe_counts = [32, 64, 128, 256]
    best_edp_nf = float('inf')
    best_hw_params, best_mapping_params = None, None
    perf_model = RealisticPerformanceModel(config)
    
    print("  Step 1: Hardware DSE (optimizing mapping for each fixed HW)...")
    for buf_kb in buffer_sizes_kb:
        for pes in pe_counts:
            hw_params = HardwareParameters(initial_num_pes=pes, initial_buffer_kb=buf_kb)
            mapping_params = MappingParameters(graph)
            fusion_params_nf = FusionParameters(graph)
            with torch.no_grad():
                for param in fusion_params_nf.parameters(): param.fill_(-10.0)
            
            # Short optimization for mapping on fixed hardware
            optimizer = optim.Adam(mapping_params.parameters(), lr=1e-2)
            for _ in range(50):
                optimizer.zero_grad()
                latency, energy, _ = perf_model(graph, hw_params, fusion_params_nf, mapping_params)
                loss = torch.log(latency + 1e-12) + torch.log(energy + 1e-12)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                latency, energy, _ = perf_model(graph, hw_params, fusion_params_nf, mapping_params)
            edp = latency.item() * energy.item()
            
            if edp < best_edp_nf:
                best_edp_nf = edp
                best_hw_params = hw_params
                best_mapping_params = mapping_params

    print(f"  Best non-fused HW found: {best_hw_params.get_num_pes().item():.0f} PEs, {best_hw_params.get_buffer_size_kb().item():.0f} KB Buffer")
    
    print("  Step 2: Applying fusion heuristics on best hardware...")
    final_fusion_params = FusionParameters(graph)
    with torch.no_grad():
        for group in graph.fusion_groups:
            p_fused = FusionParameters(graph); p_fused.get_fusion_probability(group).data.fill_(10.0)
            p_nf = FusionParameters(graph); p_nf.get_fusion_probability(group).data.fill_(-10.0)
            lat_f, en_f, _ = perf_model(graph, best_hw_params, p_fused, best_mapping_params)
            lat_nf, en_nf, _ = perf_model(graph, best_hw_params, p_nf, best_mapping_params)
            if (lat_f * en_f) < (lat_nf * en_nf):
                final_fusion_params.get_fusion_probability(group).data.fill_(10.0)
            else:
                final_fusion_params.get_fusion_probability(group).data.fill_(-10.0)

    final_latency, final_energy, final_area = perf_model(graph, best_hw_params, final_fusion_params, best_mapping_params)
    
    return {
        'Method': 'Decoupled SOTA',
        'EDP': final_latency.item() * final_energy.item(),
        'Latency': final_latency.item(),
        'Energy': final_energy.item(),
        'Area': final_area.item(),
        'hardware_params': best_hw_params,
        'fusion_params': final_fusion_params,
        'mapping_params': best_mapping_params,
        'graph': graph
    }

# --- Main Execution Block ---
def main():
    config = Config.get_instance()
    print("Parsing ResNet-18 ONNX model...")
    graph = parse_onnx_to_graph("resnet18.onnx")
    print(f"Loaded graph with {len(graph.fusion_groups)} potential fusion groups.")
    
    # Run experiments
    expert_results = run_expert_fa_dosa_experiment(graph, config, num_iterations=200)
    sota_results = run_decoupled_sota_experiment(graph, config)
    
    # Save detailed configurations to files
    save_configuration_to_file("Expert_FA-DOSA", expert_results)
    save_configuration_to_file("Decoupled_SOTA", sota_results)

    # Print comparative summary table
    generate_comparison_table(expert_results, sota_results)
    
    # Visualization
    labels = [expert_results['Method'], sota_results['Method']]
    edps = [expert_results['EDP'], sota_results['EDP']]
    areas = [expert_results['Area'], sota_results['Area']]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Methodology')
    ax1.set_ylabel('EDP (pJ*s, log scale)', color=color)
    ax1.bar([labels[0][:15]+"...", labels[1]], edps, color=color, alpha=0.6, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Area (mm²)', color=color)
    ax2.plot([labels[0][:15]+"...", labels[1]], areas, color=color, marker='o', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.title('FA-DOSA (Learnable Expert Template) vs. Decoupled SOTA')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()