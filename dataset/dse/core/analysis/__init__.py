"""
Analysis module for DOSA.

This module provides various analysis tools for dataset exploration,
model evaluation, and visualization of mapping performance.
"""

from .mlp_predictor import MLPPredictor
from .performance_analyzer import PerformanceAnalyzer
from .visualization import MappingVisualizer, ArchSearchVisualizer
from .utils import (
    theoretical_min_cycles,
    get_matching_rows,
    random_mapping_trajectories
)

__all__ = [
    'MLPPredictor',
    'PerformanceAnalyzer', 
    'MappingVisualizer',
    'ArchSearchVisualizer',
    'theoretical_min_cycles',
    'get_matching_rows',
    'random_mapping_trajectories'
] 