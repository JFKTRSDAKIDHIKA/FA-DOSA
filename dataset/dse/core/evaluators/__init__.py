"""Evaluation components."""

class PerformanceEvaluator:
    def __init__(self, arch_name, output_dir):
        self.arch_name = arch_name
        self.output_dir = output_dir

class BaselineEvaluator:
    def __init__(self, arch_name, output_dir):
        self.arch_name = arch_name
        self.output_dir = output_dir
    
    def cosa_baseline(self, hw_config):
        return {'baseline': 'cosa', 'status': 'placeholder'}
    
    def random_baseline(self, hw_config, num_mappings):
        return {'baseline': 'random', 'status': 'placeholder'}
    
    def exhaustive_baseline(self, hw_config):
        return {'baseline': 'exhaustive', 'status': 'placeholder'}

__all__ = ['PerformanceEvaluator', 'BaselineEvaluator'] 