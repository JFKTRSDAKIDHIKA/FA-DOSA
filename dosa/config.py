import torch
import torch.nn as nn

class Config:
    """全局配置类，已更新为支持多级存储层次结构。"""
    _instance = None
    def __init__(self):
        self.BYTES_PER_ELEMENT = 4
        self.CLOCK_FREQUENCY_MHZ = 1000
        self.L1_REG_BANDWIDTH_GB_S = 1024
        self.L2_SPM_BANDWIDTH_GB_S = 512
        self.DRAM_BANDWIDTH_GB_S = 100
        
        # --- NEW: 定义显式的多级存储层次 ---
        self.MEMORY_HIERARCHY = [
            # level 0: 虚拟层，代表PE内部的计算
            {'name': 'PE', 'type': 'compute'},
            # level 1: 最内层存储，例如Register File
            {'name': 'L1_Registers', 'type': 'buffer', 'size_kb': nn.Parameter(torch.log(torch.tensor(32.0))), 'bandwidth_gb_s': self.L1_REG_BANDWIDTH_GB_S},
            # level 2: 中间层存储，例如Scratchpad
            {'name': 'L2_Scratchpad', 'type': 'buffer', 'size_kb': nn.Parameter(torch.log(torch.tensor(256.0))), 'bandwidth_gb_s': self.L2_SPM_BANDWIDTH_GB_S},
            # level 3: 主存
            {'name': 'L3_DRAM', 'type': 'dram', 'bandwidth_gb_s': self.DRAM_BANDWIDTH_GB_S}
        ]
        
        # 能量模型（单位：pJ）
        self.PE_MAC_EPA_PJ = 0.561 * 1e6
        # 单位能耗（pJ/access），假设一个access为32-bit (4 bytes)
        self.L1_REG_BASE_EPA_PJ = 0.487 * 1e6 
        self.L2_SPM_BASE_EPA_PJ = 0.49 * 1e6
        self.L2_SPM_CAPACITY_COEFF_PJ_PER_KB = 0.025 * 1e6
        self.L3_DRAM_EPA_PJ = 100 * 1e6
        
        # 面积模型参数
        self.AREA_PER_PE_MM2 = 0.015
        self.AREA_PER_KB_L1_MM2 = 0.008 # L1通常更贵
        self.AREA_PER_KB_L2_MM2 = 0.005 # L2相对便宜
        self.AREA_BASE_MM2 = 1.0
        self.PENALTY_WEIGHT = 1e6

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance