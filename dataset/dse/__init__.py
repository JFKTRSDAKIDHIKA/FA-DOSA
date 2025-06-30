# __init__.py
# Use lazy imports to avoid immediate dependency loading

# Core search functionality (no heavy dependencies)
from .core import SearchEngine

# Main search interface (backward compatible)
from . import mapping_driven_hw_search

# Lazy imports for classes with heavy dependencies
def get_dla_dataset_creator():
    """Lazy import for DlaDatasetCreator to avoid sklearn dependency on module import."""
    from .dla_dataset_creator_class import DlaDatasetCreator
    return DlaDatasetCreator

def get_dla_dataset():
    """Lazy import for DlaDataset to avoid sklearn dependency on module import."""
    from .dla_dataset_creator_class import DlaDataset
    return DlaDataset

# For backward compatibility, expose these through __getattr__
def __getattr__(name):
    if name == 'DlaDatasetCreator':
        return get_dla_dataset_creator()
    elif name == 'DlaDataset':
        return get_dla_dataset()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'SearchEngine',
    'mapping_driven_hw_search',
    'DlaDatasetCreator',  # Available through __getattr__
    'DlaDataset'         # Available through __getattr__
]
