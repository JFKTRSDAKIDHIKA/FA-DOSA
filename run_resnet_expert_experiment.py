# [CURSOR_START]
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# --- Prerequisite: Assuming necessary classes are available ---
# The following classes are assumed to be defined in accompanying files
# or pasted here: Config, HardwareParameters, FusionParameters,
# ComputationGraph, parse_onnx_to_graph, calculate_macs, ExpertConvReluTemplate.
# For a self-contained script, these definitions would be included here.

# Let's stub the prerequisite classes for completeness.
class Config:
    _instance = None
    def __init__(self):
        self.BYTES_PER_ELEMENT = 4
        self.DRAM_BANDWIDTH_GB_S = 128
        self.CLOCK_FREQUENCY_MHZ = 1000
        # Energy costs in pJ
        self.DRAM_ACCESS_COST_PJ_PER_BYTE = 2.5
        self.SRAM_ACCESS_COST_PJ_PER_BYTE = 0.2
        self.MAC_ENERGY_PJ = 0.1
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
    # This is a stub for the actual ONNX parser.
    # In a real run, this would parse the 'resnet18.onnx' file.
    graph = ComputationGraph()
    # Simplified ResNet-18 block structure
    for i in range(8): # 8 blocks in ResNet-18
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

class ExpertConvReluTemplate(torch.nn.Module):
    def __init__(self, dims: dict, hardware_params: 'HardwareParameters'):
        super().__init__()
        self.dims = dims
        self.hardware_params = hardware_params
        self.config = Config.get_instance()
    def get_buffer_requirements_bytes(self) -> torch.Tensor:
        dims = self.dims
        input_size = dims.get('N',1)*dims.get('C',1)*dims.get('P',1)*dims.get('Q',1)*self.config.BYTES_PER_ELEMENT
        weight_size = dims.get('K',1)*dims.get('C',1)*dims.get('R',1)*dims.get('S',1)*self.config.BYTES_PER_ELEMENT
        output_size = dims.get('N',1)*dims.get('K',1)*dims.get('P',1)*dims.get('Q',1)*self.config.BYTES_PER_ELEMENT
        return torch.tensor(float(input_size + weight_size + output_size))
    def get_access_counts(self, is_fused: bool) -> dict:
        dims = self.dims
        input_access = dims.get('N',1)*dims.get('C',1)*dims.get('P',1)*dims.get('Q',1)
        weight_access = dims.get('K',1)*dims.get('C',1)*dims.get('R',1)*dims.get('S',1)
        output_access = dims.get('N',1)*dims.get('K',1)*dims.get('P',1)*dims.get('Q',1)
        if is_fused:
            # Intermediate output (Conv->ReLU) is not written to DRAM
            return {'input': input_access, 'weight': weight_access, 'output': output_access}
        else:
            # Conv output is written, ReLU input is read. Access to this tensor is doubled.
            return {'input': input_access, 'weight': weight_access, 'output': output_access * 2}

# --- Step 1: Define the New, More Realistic Performance Model ---

class RealisticPerformanceModel(nn.Module):
    """
    A more physically realistic performance model that addresses the flaws of the simplified version.
    - Latency is modeled using a Roofline model.
    - Energy costs are coupled with hardware parameters.
    - Buffer constraint penalties are more robust.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(self, graph: ComputationGraph, hardware_params: HardwareParameters, fusion_params: FusionParameters) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_latency = torch.tensor(0.0)
        total_energy = torch.tensor(0.0)
        
        for group in graph.fusion_groups:
            # Assuming all fusion groups are Conv -> ReLU as per the experiment's focus
            conv_name, _ = group
            dims = graph.layers[conv_name]['dims']
            
            template = ExpertConvReluTemplate(dims, hardware_params)
            p_fuse = fusion_params.get_fusion_probability(group)
            
            # --- Calculate costs for fused and non-fused scenarios ---
            costs = {'fused': {}, 'non_fused': {}}
            for case in ['fused', 'non_fused']:
                is_fused = (case == 'fused')
                
                # 1. Latency Calculation (Roofline Model)
                macs = calculate_macs(dims, 'Conv')
                
                # Compute latency depends on #PEs
                gops = hardware_params.get_num_pes() * (self.config.CLOCK_FREQUENCY_MHZ / 1000.0)
                macs = macs if isinstance(macs, torch.Tensor) else torch.tensor(macs, dtype=torch.float32)
                gops = gops if isinstance(gops, torch.Tensor) else torch.tensor(gops, dtype=torch.float32)
                compute_latency = macs / (gops * 1e9) # in seconds
                
                # Memory latency depends on DRAM access
                access_counts = template.get_access_counts(is_fused)
                total_bytes_to_dram = sum(access_counts.values()) * self.config.BYTES_PER_ELEMENT
                dram_bandwidth_bytes_s = self.config.DRAM_BANDWIDTH_GB_S * 1e9
                memory_latency = total_bytes_to_dram / dram_bandwidth_bytes_s # in seconds
                # --- 修复 torch.maximum 参数类型 ---
                if not isinstance(compute_latency, torch.Tensor):
                    compute_latency = torch.tensor(compute_latency, dtype=torch.float32)
                if not isinstance(memory_latency, torch.Tensor):
                    memory_latency = torch.tensor(memory_latency, dtype=torch.float32)
                latency = torch.maximum(compute_latency, memory_latency)

                # 2. Energy Calculation (Hardware-Coupled)
                # Static energy is proportional to area
                static_energy_per_second = hardware_params.get_area_cost() * 10 # Simplified static power factor
                static_energy = static_energy_per_second * latency

                # Dynamic energy
                mac_energy = macs * self.config.MAC_ENERGY_PJ
                dram_energy = total_bytes_to_dram * self.config.DRAM_ACCESS_COST_PJ_PER_BYTE
                
                # SRAM energy is proportional to its size and accesses (simplified)
                buffer_req_bytes = template.get_buffer_requirements_bytes()
                sram_energy = buffer_req_bytes * self.config.SRAM_ACCESS_COST_PJ_PER_BYTE
                
                energy = static_energy + mac_energy + dram_energy + sram_energy
                
                # 3. Buffer Constraint Penalty (Multiplicative)
                buffer_req = template.get_buffer_requirements_bytes()
                available_buffer = hardware_params.get_buffer_size_bytes()
                
                # Apply a steep multiplicative penalty for violations
                penalty_multiplier = torch.relu(buffer_req / available_buffer - 1.0) * self.config.PENALTY_WEIGHT + 1.0

                costs[case]['latency'] = latency * penalty_multiplier
                costs[case]['energy'] = energy * penalty_multiplier

            # --- Weighted average of costs based on fusion probability ---
            total_latency += p_fuse * costs['fused']['latency'] + (1 - p_fuse) * costs['non_fused']['latency']
            total_energy += p_fuse * costs['fused']['energy'] + (1 - p_fuse) * costs['non_fused']['energy']
            
        return total_latency, total_energy, hardware_params.get_area_cost()

# --- Step 2: Implement the Revised Experiment Functions ---

def run_expert_fa_dosa_experiment(graph: ComputationGraph, config: Config, num_iterations=200) -> dict:
    """
    Runs the Fusion-Aware DOSA experiment with the new realistic performance model.
    """
    print("\nRunning Expert FA-DOSA experiment (Realistic Model)...")
    hardware_params = HardwareParameters(initial_num_pes=64, initial_buffer_kb=128)
    fusion_params = FusionParameters(graph)
    perf_model = RealisticPerformanceModel(config)
    
    optimizer = optim.Adam(list(hardware_params.parameters()) + list(fusion_params.parameters()), lr=1e-2)

    for i in range(num_iterations):
        optimizer.zero_grad()
        latency, energy, area = perf_model(graph, hardware_params, fusion_params)
        
        # Loss function remains in log domain for stability
        loss = torch.log(latency + 1e-12) + torch.log(energy + 1e-12) + 0.01 * area
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print(f"[Expert FA-DOSA] Iter {i}: Loss={loss.item():.4f}, Latency={latency.item():.2e}s, Energy={energy.item():.2e}pJ, Area={area.item():.2f}mm²")

    # Final evaluation with deterministic fusion decisions
    with torch.no_grad():
        for param in fusion_params.parameters():
            # Set to a large positive/negative logit for a hard 0/1 decision
            param.data = (param.data > 0).float() * 10.0 - (param.data <= 0).float() * 10.0
        final_latency, final_energy, final_area = perf_model(graph, hardware_params, fusion_params)
    
    edp = final_latency.item() * final_energy.item()
    return {
        'Method': 'Expert FA-DOSA',
        'EDP': edp,
        'Latency': final_latency.item(),
        'Energy': final_energy.item(),
        'Area': final_area.item()
    }

def run_decoupled_sota_experiment(graph: ComputationGraph, config: Config) -> dict:
    """
    Runs the Decoupled SOTA baseline experiment using the new realistic model for fair comparison.
    """
    print("\nRunning Decoupled SOTA experiment (Realistic Model)...")
    # 1. Hardware DSE Grid
    buffer_sizes_kb = [64, 128, 256, 512]
    pe_counts = [32, 64, 128, 256]
    best_edp_nf = float('inf')
    best_hw_params = None

    perf_model = RealisticPerformanceModel(config)
    
    print("  Step 1: Hardware DSE (without fusion)...")
    for buf_kb in buffer_sizes_kb:
        for pes in pe_counts:
            hw_params = HardwareParameters(initial_num_pes=pes, initial_buffer_kb=buf_kb)
            # Create fusion_params with fusion turned off
            fusion_params_nf = FusionParameters(graph)
            with torch.no_grad():
                for param in fusion_params_nf.parameters():
                    param.fill_(-10.0) # Large negative logit -> p_fuse ≈ 0
            
            latency, energy, _ = perf_model(graph, hw_params, fusion_params_nf)
            edp = latency.item() * energy.item()

            if edp < best_edp_nf:
                best_edp_nf = edp
                best_hw_params = hw_params
    
    print(f"  Best non-fused HW found: {best_hw_params.get_num_pes().item():.0f} PEs, {best_hw_params.get_buffer_size_kb().item():.0f} KB Buffer")
    
    # 2. On best hardware, apply fusion heuristics
    print("  Step 2: Applying fusion heuristics on best hardware...")
    final_fusion_params = FusionParameters(graph)
    with torch.no_grad():
        for group in graph.fusion_groups:
            # Create two fusion param sets: one for fused, one for non-fused
            p_fused = FusionParameters(graph); p_fused.get_fusion_probability(group).data.fill_(10.0)
            p_nf = FusionParameters(graph); p_nf.get_fusion_probability(group).data.fill_(-10.0)

            lat_f, en_f, _ = perf_model(graph, best_hw_params, p_fused)
            edp_f = lat_f * en_f
            
            lat_nf, en_nf, _ = perf_model(graph, best_hw_params, p_nf)
            edp_nf = lat_nf * en_nf

            # Heuristic: fuse if it improves EDP
            if edp_f < edp_nf:
                final_fusion_params.get_fusion_probability(group).data.fill_(10.0)
            else:
                final_fusion_params.get_fusion_probability(group).data.fill_(-10.0)
                
    final_latency, final_energy, final_area = perf_model(graph, best_hw_params, final_fusion_params)
    edp = final_latency.item() * final_energy.item()
    
    return {
        'Method': 'Decoupled SOTA',
        'EDP': edp,
        'Latency': final_latency.item(),
        'Energy': final_energy.item(),
        'Area': final_area.item()
    }

# --- Step 3: Main Execution Block ---
def main():
    """
    Main function to orchestrate the experiment, run both methods, and compare results.
    """
    config = Config.get_instance()
    
    print("Parsing ResNet-18 ONNX model...")
    # In a real scenario, provide the path to 'resnet18.onnx'
    # For this self-contained script, we use the stubbed parser.
    graph = parse_onnx_to_graph("resnet18.onnx")
    print(f"Loaded graph with {len(graph.fusion_groups)} potential fusion groups.")
    
    # Run experiments
    expert_results = run_expert_fa_dosa_experiment(graph, config, num_iterations=200)
    sota_results = run_decoupled_sota_experiment(graph, config)
    
    # --- Print and Visualize Results ---
    print("\n" + "="*50)
    print("          COMPARATIVE EXPERIMENT RESULTS")
    print("="*50)
    print(f"{'Method':<20} {'EDP (pJ*s)':>15} {'Latency (s)':>15} {'Energy (pJ)':>15} {'Area (mm²)':>12}")
    print("-"*80)
    
    for res in [expert_results, sota_results]:
        print(f"{res['Method']:<20} {res['EDP']:>15.3e} {res['Latency']:>15.3e} {res['Energy']:>15.3e} {res['Area']:>12.2f}")
    
    # Visualization
    labels = [expert_results['Method'], sota_results['Method']]
    edps = [expert_results['EDP'], sota_results['EDP']]
    areas = [expert_results['Area'], sota_results['Area']]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Methodology')
    ax1.set_ylabel('EDP (pJ*s, log scale)', color=color)
    ax1.bar(labels, edps, color=color, alpha=0.6, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Area (mm²)', color=color)
    ax2.plot(labels, areas, color=color, marker='o', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Expert FA-DOSA vs. Decoupled SOTA on ResNet-18')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()

# [CURSOR_END]