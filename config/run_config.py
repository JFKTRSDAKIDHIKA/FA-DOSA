"""
Run configuration class for managing experiment configuration and paths.
"""

import pathlib
from typing import List, Optional, Callable
from dataclasses import dataclass, field

from dataset.common.utils.file_utils import FileHandler
from dataset.common.utils.math_utils import MathUtils, RandomUtils


@dataclass
class RunConfig:
    """Configuration class for DOSA experiment runs."""
    
    # Architecture configuration
    arch_name: str
    arch_file: Optional[str] = None
    num_arch: int = 1
    
    # Workload configuration
    workloads: List[str] = field(default_factory=list)
    base_workload_path: str = ""
    layer_idx: Optional[str] = None
    
    # Mapping configuration
    mapper: str = "random"
    num_mappings: int = 1000
    min_metric: Optional[str] = None
    
    # Output configuration
    output_dir: str = "output_random"
    
    # General configuration
    random_seed: int = 1
    exist: bool = False
    
    # Derived paths (computed after initialization)
    output_path: Optional[pathlib.Path] = field(default=None, init=False)
    workload_layers: List[pathlib.Path] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Initialize derived configurations after object creation."""
        self._setup_output_path()
        self._setup_workload_layers()
        self._setup_random_seed()
    
    def _setup_output_path(self) -> None:
        """Setup the output directory path."""
        output_name = self.output_dir
        if self.layer_idx:
            output_name += f'_layer{self.layer_idx}'
        
        self.output_path = pathlib.Path(output_name).resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_workload_layers(self) -> None:
        """Setup the list of workload layer paths."""
        base_path = pathlib.Path(self.base_workload_path).resolve()
        
        self.workload_layers = []
        for workload in self.workloads:
            workload_path = base_path / workload
            unique_layers_file = workload_path / 'unique_layers.yaml'
            
            if not unique_layers_file.exists():
                raise FileNotFoundError(f"Unique layers file not found: {unique_layers_file}")
            
            unique_layers = FileHandler.load_yaml(unique_layers_file)
            for layer_name in unique_layers:
                layer_path = workload_path / f"{layer_name}.yaml"
                if layer_path.exists():
                    self.workload_layers.append(layer_path.resolve())
                else:
                    raise FileNotFoundError(f"Layer file not found: {layer_path}")
    
    def _setup_random_seed(self) -> None:
        """Setup the random seed for reproducibility."""
        RandomUtils.set_global_seed(self.random_seed)
    
    def get_min_metric_function(self) -> Optional[Callable]:
        """Get the function to compute the minimum metric if specified."""
        if not self.min_metric:
            return None
        
        if self.min_metric in ["cycle", "energy"]:
            return lambda row: row[f"target.{self.min_metric}"]
        elif self.min_metric == "edp":
            return lambda row: row["target.cycle"] * row["target.energy"]
        else:
            raise ValueError(f"Unsupported metric: {self.min_metric}")
    
    @property
    def logs_dir(self) -> pathlib.Path:
        """Get the logs directory path."""
        logs_path = self.output_path / "logs"
        logs_path.mkdir(parents=True, exist_ok=True)
        return logs_path
    
    @property
    def dataset_csv_path(self) -> pathlib.Path:
        """Get the dataset CSV file path."""
        return self.output_path / "dataset.csv"
    
    @property
    def dataset_tarball_path(self) -> pathlib.Path:
        """Get the dataset tarball path."""
        return self.output_path / "dataset.csv.tar.gz"
    
    def validate(self) -> None:
        """Validate the configuration for consistency."""
        # Check if architecture is supported
        supported_archs = ['gemmini', 'simba']
        if self.arch_name not in supported_archs:
            raise ValueError(f"Architecture '{self.arch_name}' not supported. "
                           f"Supported: {supported_archs}")
        
        # Check if mapper is supported
        supported_mappers = ['random', 'cosa']
        if self.mapper not in supported_mappers:
            raise ValueError(f"Mapper '{self.mapper}' not supported. "
                           f"Supported: {supported_mappers}")
        
        # Check architecture file exists if specified
        if self.arch_file and not pathlib.Path(self.arch_file).is_file():
            raise FileNotFoundError(f"Architecture file not found: {self.arch_file}")
        
        # Override num_arch if arch_file is specified
        if self.arch_file:
            self.num_arch = 1
    
    @classmethod
    def from_args(cls, args) -> 'RunConfig':
        """Create RunConfig from parsed arguments."""
        config = cls(
            arch_name=args.arch_name,
            arch_file=args.arch_file,
            num_arch=args.num_arch,
            workloads=args.workload,
            base_workload_path=args.base_workload_path,
            layer_idx=args.layer_idx if args.layer_idx else None,
            mapper=args.mapper,
            num_mappings=args.num_mappings,
            min_metric=args.min_metric,
            output_dir=args.output_dir,
            random_seed=args.random_seed,
            exist=args.exist
        )
        
        config.validate()
        return config 