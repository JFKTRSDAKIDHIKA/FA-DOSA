"""
Model modules for DOSA.

This module contains refactored machine learning models for energy and latency prediction.
"""

from .energy_model_refactored import EnergyModelRefactored
from .latency_model_refactored import LatencyModelRefactored

__all__ = [
    'EnergyModelRefactored',
    'LatencyModelRefactored'
] 