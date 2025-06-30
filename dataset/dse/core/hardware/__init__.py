"""Hardware processing components."""

class HardwareOptimizer:
    def __init__(self, arch_name, output_dir):
        self.arch_name = arch_name
        self.output_dir = output_dir

class HardwareBounds:
    def __init__(self, arch_name):
        self.arch_name = arch_name

__all__ = ['HardwareOptimizer', 'HardwareBounds'] 