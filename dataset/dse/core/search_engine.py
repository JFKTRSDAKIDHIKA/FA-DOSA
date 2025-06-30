"""
Main search engine for design space exploration.
"""

import pathlib
from typing import List, Dict, Any, Optional, Tuple
import logging

import torch
import numpy as np

# Matplotlib imports with Agg backend for server environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from ...common.utils import logger, FileHandler, ConfigDict, PerformanceTracker
from ...workloads import Prob
from ...common import utils
from .. import pytorch_util

from .search_strategies import BayesianOptimizer, GradientDescentOptimizer, RandomSearchOptimizer
from .mapping import MappingGenerator, MappingEvaluator
from .hardware import HardwareOptimizer
from .evaluators import PerformanceEvaluator, BaselineEvaluator


class SearchEngine:
    """
    Main engine for design space exploration with pluggable search strategies.
    """
    
    def __init__(self, 
                 arch_name: str,
                 output_dir: pathlib.Path,
                 workload: str,
                 metric: str = "cycle",
                 gpu_id: Optional[int] = 0,
                 log_times: bool = False):
        """
        Initialize search engine.
        
        Args:
            arch_name: Architecture name (e.g., 'gemmini')
            output_dir: Output directory for results
            workload: Workload name
            metric: Optimization metric
            gpu_id: GPU device ID (None for CPU mode)
            log_times: Whether to log timing information
        """
        logger.info(f"Initializing SearchEngine: arch={arch_name}, workload={workload}, metric={metric}")
        
        # Basic configuration
        self.arch_name = arch_name
        self.output_dir = pathlib.Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workload_name = workload
        self.metric = metric
        self.log_times = log_times
        self.gpu_id = gpu_id
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Initialize GPU
        if self.gpu_id is None:
            # Force CPU mode
            pytorch_util.init_gpu(use_gpu=False)
        else:
            # Use GPU with specified ID
            pytorch_util.init_gpu(use_gpu=True, gpu_id=self.gpu_id)
        
        # Load workload layers
        self._load_workload_layers()
        
        # Initialize components
        self.mapping_generator = MappingGenerator(arch_name)
        self.mapping_evaluator = MappingEvaluator(arch_name, output_dir)
        self.hardware_optimizer = HardwareOptimizer(arch_name, output_dir)
        self.performance_evaluator = PerformanceEvaluator(arch_name, output_dir)
        self.baseline_evaluator = BaselineEvaluator(arch_name, output_dir)
        
        # Initialize search strategies
        self._initialize_search_strategies()
        
        logger.info(f"SearchEngine initialized with {len(self.layers)} layers")
    
    def _load_workload_layers(self) -> None:
        """Load workload layers from configuration files."""
        from ... import DATASET_ROOT_PATH
        
        base_workload_path = DATASET_ROOT_PATH / "workloads"
        self.workload_path = base_workload_path / self.workload_name
        
        if not self.workload_path.exists():
            raise FileNotFoundError(f"Workload path not found: {self.workload_path}")
        
        # Load unique layers
        unique_layers_path = self.workload_path / 'unique_layers.yaml'
        if not unique_layers_path.exists():
            raise FileNotFoundError(f"Unique layers file not found: {unique_layers_path}")
        
        unique_layers = FileHandler.load_yaml(unique_layers_path)
        
        self.layers: List[Prob] = []
        for unique_layer in unique_layers:
            layer_path = self.workload_path / f'{unique_layer}.yaml'
            if not layer_path.exists():
                logger.warning(f"Layer file not found: {layer_path}")
                continue
            
            try:
                layer_prob = Prob(layer_path.resolve())
                self.layers.append(layer_prob)
                logger.debug(f"Loaded layer: {unique_layer}")
            except Exception as e:
                logger.error(f"Failed to load layer {unique_layer}: {e}")
        
        if not self.layers:
            raise ValueError("No valid layers found in workload")
        
        # Load layer counts
        self.layer_count = self._get_layer_count()
        
        # Precompute problem dimensions for GPU
        self._precompute_prob_dims()
    
    def _get_layer_count(self) -> List[int]:
        """Get layer occurrence counts."""
        try:
            layer_count_path = self.workload_path / 'layer_count.yaml'
            if layer_count_path.exists():
                layer_count_dict = FileHandler.load_yaml(layer_count_path)
                counts = [layer_count_dict[prob.config_str()]["count"] for prob in self.layers]
                logger.debug(f"Loaded layer counts: {counts}")
                return counts
        except Exception as e:
            logger.warning(f"Couldn't load layer count: {e}, using default counts")
        
        # Default to equal counts
        return [1 for _ in self.layers]
    
    def _precompute_prob_dims(self) -> None:
        """Precompute problem dimensions for efficient GPU computation."""
        num_dims = 7
        prob_dims = [[] for _ in range(num_dims)]
        
        for prob in self.layers:
            for dim_idx, dim in prob.prob_idx_name_dict.items():
                prob_dims[dim_idx].append(prob.prob[dim])
        
        # Convert to reciprocal tensor for GPU computation
        self.prob_dims_recip = None
        for dim_idx in range(len(prob_dims)):
            one_dim = pytorch_util.from_numpy(1/np.array(prob_dims[dim_idx])).unsqueeze(0)
            if self.prob_dims_recip is None:
                self.prob_dims_recip = one_dim
            else:
                self.prob_dims_recip = torch.cat((self.prob_dims_recip, one_dim), dim=0)
        
        logger.debug(f"Precomputed problem dimensions tensor: {self.prob_dims_recip.shape}")
    
    def _initialize_search_strategies(self) -> None:
        """Initialize available search strategies."""
        self.search_strategies = {
            'bayesian': BayesianOptimizer(
                arch_name=self.arch_name,
                output_dir=self.output_dir,
                layers=self.layers,
                mapping_evaluator=self.mapping_evaluator
            ),
            'gradient_descent': GradientDescentOptimizer(
                arch_name=self.arch_name,
                output_dir=self.output_dir,
                layers=self.layers,
                mapping_evaluator=self.mapping_evaluator,
                performance_tracker=self.performance_tracker
            ),
            'random': RandomSearchOptimizer(
                arch_name=self.arch_name,
                output_dir=self.output_dir,
                layers=self.layers,
                mapping_evaluator=self.mapping_evaluator
            )
        }
        
        logger.debug(f"Initialized search strategies: {list(self.search_strategies.keys())}")
    
    def search(self, 
               strategy: str,
               n_calls: int = 100,
               n_initial_points: int = 10,
               **strategy_kwargs) -> Dict[str, Any]:
        """
        Run design space exploration using specified strategy.
        
        Args:
            strategy: Search strategy ('bayesian', 'gradient_descent', 'random')
            n_calls: Number of optimization calls
            n_initial_points: Number of initial random points
            **strategy_kwargs: Additional strategy-specific arguments
            
        Returns:
            Dictionary with search results
        """
        if strategy not in self.search_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.search_strategies.keys())}")
        
        logger.info(f"Starting {strategy} search with {n_calls} calls, {n_initial_points} initial points")
        
        # Track search timing
        import time
        start_time = time.time()
        
        try:
            # Get strategy instance
            strategy_instance = self.search_strategies[strategy]
            
            # Run search
            results = strategy_instance.search(
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                **strategy_kwargs
            )
            
            # Add metadata
            search_time = time.time() - start_time
            results.update({
                'search_strategy': strategy,
                'search_time': search_time,
                'n_calls': n_calls,
                'n_initial_points': n_initial_points,
                'arch_name': self.arch_name,
                'workload': self.workload_name,
                'metric': self.metric
            })
            
            # Track performance
            self.performance_tracker.record_timing('total_search_time', search_time)
            self.performance_tracker.record_metric('n_calls', n_calls)
            
            logger.info(f"Search completed in {search_time:.2f}s")
            
            # Save results
            self._save_search_results(results, strategy)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def search_network(self,
                      dataset_path: str,
                      predictor: str = "analytical",
                      plot_only: bool = False,
                      ordering: str = "shuffle") -> Dict[str, Any]:
        """
        Search for optimal mappings across the entire network.
        
        Args:
            dataset_path: Path to dataset
            predictor: Predictor type ('analytical', 'learned')
            plot_only: Whether to only generate plots
            ordering: Layer ordering strategy
            
        Returns:
            Network search results
        """
        logger.info(f"Starting network search: dataset={dataset_path}, predictor={predictor}")
        
        if plot_only:
            logger.info("Plot-only mode: generating visualizations from existing data")
            return self._generate_plots_only(dataset_path)
        
        # Run actual search experiment
        results = {}
        
        try:
            # Run different search strategies based on predictor type
            if predictor in ['analytical', 'both']:
                logger.info("Running analytical search...")
                analytical_results = self.search(
                    strategy="random",  # Start with random search for robustness
                    n_calls=20,  # Reasonable number for quick results
                    n_initial_points=5
                )
                results['analytical'] = analytical_results
                
            if predictor in ['dnn', 'both']:
                logger.info("Running DNN-based search...")
                dnn_results = self.search(
                    strategy="bayesian",  # Use bayesian for DNN predictor
                    n_calls=15,
                    n_initial_points=3
                )
                results['dnn'] = dnn_results
            
            # Generate summary
            summary = {
                'status': 'completed',
                'dataset_path': dataset_path,
                'predictor': predictor,
                'ordering': ordering,
                'total_layers': len(self.layers),
                'layer_names': [layer.config_str() for layer in self.layers],
                'workload': self.workload_name,
                'architecture': self.arch_name,
                'search_results': results,
                'output_directory': str(self.output_dir)
            }
            
            # Save comprehensive results
            self._save_network_results(summary)
            
            logger.info(f"Network search completed successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Network search failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'dataset_path': dataset_path,
                'predictor': predictor
            }
    
    def evaluate_baselines(self, 
                          hw_config: Dict[str, Any],
                          baseline_types: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate baseline strategies.
        
        Args:
            hw_config: Hardware configuration
            baseline_types: List of baseline types to evaluate
            
        Returns:
            Baseline evaluation results
        """
        if baseline_types is None:
            baseline_types = ['cosa', 'random', 'exhaustive']
        
        logger.info(f"Evaluating baselines: {baseline_types}")
        
        results = {}
        for baseline_type in baseline_types:
            try:
                if baseline_type == 'cosa':
                    result = self.baseline_evaluator.cosa_baseline(hw_config)
                elif baseline_type == 'random':
                    result = self.baseline_evaluator.random_baseline(hw_config, num_mappings=100)
                elif baseline_type == 'exhaustive':
                    result = self.baseline_evaluator.exhaustive_baseline(hw_config)
                else:
                    logger.warning(f"Unknown baseline type: {baseline_type}")
                    continue
                
                results[baseline_type] = result
                logger.debug(f"Evaluated {baseline_type} baseline")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {baseline_type} baseline: {e}")
                results[baseline_type] = {'error': str(e)}
        
        return results
    
    def get_search_summary(self) -> Dict[str, Any]:
        """
        Get summary of search engine status and capabilities.
        
        Returns:
            Summary dictionary
        """
        return {
            'arch_name': self.arch_name,
            'workload': self.workload_name,
            'metric': self.metric,
            'num_layers': len(self.layers),
            'layer_names': [layer.config_str() for layer in self.layers],
            'layer_counts': self.layer_count,
            'available_strategies': list(self.search_strategies.keys()),
            'output_dir': str(self.output_dir),
            'gpu_id': self.gpu_id,
            'performance_summary': self.performance_tracker.get_summary()
        }
    
    def _save_search_results(self, results: Dict[str, Any], strategy: str) -> None:
        """Save search results to output directory."""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"{strategy}_{timestamp}.json"
            
            FileHandler.save_json(results_file, results)
            logger.info(f"Saved search results to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save search results: {e}")
    
    def _save_network_results(self, summary: Dict[str, Any]) -> None:
        """Save network-level search results."""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save main results
            results_file = self.output_dir / f"network_search_{timestamp}.json"
            FileHandler.save_json(results_file, summary)
            
            # Save CSV summary for easy analysis
            csv_file = self.output_dir / f"network_summary_{timestamp}.csv"
            self._save_results_csv(summary, csv_file)
            
            # Save experiment log
            log_file = self.output_dir / f"experiment_log_{timestamp}.txt"
            self._save_experiment_log(summary, log_file)
            
            logger.info(f"Network results saved:")
            logger.info(f"  JSON: {results_file}")
            logger.info(f"  CSV:  {csv_file}")
            logger.info(f"  Log:  {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save network results: {e}")
    
    def _save_results_csv(self, summary: Dict[str, Any], csv_file: pathlib.Path) -> None:
        """Save results summary as CSV."""
        try:
            import csv
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['Metric', 'Value'])
                
                # Write basic info
                writer.writerow(['Status', summary.get('status', 'unknown')])
                writer.writerow(['Workload', summary.get('workload', 'unknown')])
                writer.writerow(['Architecture', summary.get('architecture', 'unknown')])
                writer.writerow(['Predictor', summary.get('predictor', 'unknown')])
                writer.writerow(['Total Layers', summary.get('total_layers', 0)])
                
                # Write search results summary
                search_results = summary.get('search_results', {})
                for strategy, results in search_results.items():
                    if isinstance(results, dict):
                        writer.writerow([f'{strategy.title()} Strategy', 'Executed'])
                        writer.writerow([f'{strategy.title()} Calls', results.get('n_calls', 'N/A')])
                        writer.writerow([f'{strategy.title()} Time (s)', results.get('search_time', 'N/A')])
                
        except Exception as e:
            logger.error(f"Failed to save CSV summary: {e}")
    
    def _save_experiment_log(self, summary: Dict[str, Any], log_file: pathlib.Path) -> None:
        """Save detailed experiment log."""
        try:
            with open(log_file, 'w') as f:
                f.write("DOSA Network Search Experiment Log\n")
                f.write("=" * 50 + "\n\n")
                
                import datetime
                f.write(f"Experiment Time: {datetime.datetime.now()}\n")
                f.write(f"Status: {summary.get('status', 'unknown')}\n")
                f.write(f"Workload: {summary.get('workload', 'unknown')}\n")
                f.write(f"Architecture: {summary.get('architecture', 'unknown')}\n")
                f.write(f"Predictor: {summary.get('predictor', 'unknown')}\n")
                f.write(f"Dataset: {summary.get('dataset_path', 'unknown')}\n\n")
                
                f.write("Layer Information:\n")
                f.write("-" * 20 + "\n")
                layer_names = summary.get('layer_names', [])
                for i, layer in enumerate(layer_names):
                    f.write(f"  Layer {i+1}: {layer}\n")
                f.write(f"  Total Layers: {len(layer_names)}\n\n")
                
                f.write("Search Results:\n")
                f.write("-" * 15 + "\n")
                search_results = summary.get('search_results', {})
                for strategy, results in search_results.items():
                    f.write(f"  {strategy.title()} Strategy:\n")
                    if isinstance(results, dict):
                        f.write(f"    Calls: {results.get('n_calls', 'N/A')}\n")
                        f.write(f"    Time: {results.get('search_time', 'N/A')}s\n")
                        f.write(f"    Initial Points: {results.get('n_initial_points', 'N/A')}\n")
                    f.write("\n")
                
                f.write(f"Output Directory: {summary.get('output_directory', 'unknown')}\n")
                
        except Exception as e:
            logger.error(f"Failed to save experiment log: {e}")
    
    def _generate_plots_only(self, dataset_path: str) -> Dict[str, Any]:
        """Generate plots from existing data without running search."""
        logger.info("Generating visualizations from existing results...")
        
        try:
            # Look for existing result files
            result_files = list(self.output_dir.glob("*.json"))
            
            if not result_files:
                logger.warning("No existing result files found for visualization")
                return {
                    'status': 'no_data',
                    'message': 'No existing results found for plotting',
                    'output_directory': str(self.output_dir)
                }
            
            logger.info(f"Found {len(result_files)} result files for visualization")
            
            # Generate actual plots
            generated_plots = []
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create workload overview plot
            workload_plot = self._create_workload_overview_plot(timestamp)
            if workload_plot:
                generated_plots.append(workload_plot)
            
            # Create search results plot  
            search_plot = self._create_search_results_plot(result_files, timestamp)
            if search_plot:
                generated_plots.append(search_plot)
            
            # Create layer analysis plot
            layer_plot = self._create_layer_analysis_plot(timestamp)
            if layer_plot:
                generated_plots.append(layer_plot)
            
            # Create comprehensive dashboard
            dashboard_plot = self._create_comprehensive_dashboard(result_files, timestamp)
            if dashboard_plot:
                generated_plots.append(dashboard_plot)
            
            # Create plot summary
            plot_summary = {
                'status': 'plots_generated',
                'dataset_path': dataset_path,
                'num_result_files': len(result_files),
                'generated_plots': generated_plots,
                'total_plots': len(generated_plots),
                'output_directory': str(self.output_dir),
                'timestamp': timestamp
            }
            
            # Save plot summary
            summary_file = self.output_dir / f"visualization_summary_{timestamp}.json"
            FileHandler.save_json(summary_file, plot_summary)
            
            logger.info(f"🎨 Generated {len(generated_plots)} visualization plots:")
            for plot in generated_plots:
                logger.info(f"  📊 {plot}")
            logger.info(f"📝 Visualization summary: {summary_file}")
            
            return plot_summary
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
            return {
                'status': 'plot_failed',
                'error': str(e),
                'output_directory': str(self.output_dir)
            }
    
    def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        try:
            # Clean up temporary files
            from ...common.utils import TimeloopInterface
            TimeloopInterface.cleanup_temp_files(self.output_dir)
            
            # Reset performance tracking
            self.performance_tracker.reset()
            
            logger.debug("SearchEngine cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def _create_workload_overview_plot(self, timestamp: str) -> Optional[str]:
        """Create workload overview visualization."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create subplot grid
            gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
            
            # Plot 1: Layer counts
            plt.subplot(gs[0, 0])
            layer_names = [f"L{i+1}" for i in range(len(self.layers))]
            plt.bar(layer_names, self.layer_count, color='skyblue', alpha=0.7)
            plt.title('Layer Occurrence Counts')
            plt.xlabel('Layers')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Plot 2: Layer dimensions visualization
            plt.subplot(gs[0, 1])
            layer_complexities = []
            for layer in self.layers:
                # Calculate layer complexity (product of key dimensions)
                complexity = 1
                for dim_name, dim_value in layer.prob.items():
                    if dim_name in ['M', 'K', 'N']:  # Key matrix dimensions
                        complexity *= dim_value
                layer_complexities.append(complexity)
            
            plt.scatter(range(len(layer_complexities)), layer_complexities, 
                       color='coral', s=100, alpha=0.7)
            plt.title('Layer Computational Complexity')
            plt.xlabel('Layer Index')
            plt.ylabel('Complexity (ops)')
            plt.yscale('log')
            
            # Plot 3: Architecture info
            plt.subplot(gs[1, :])
            info_text = f"""
Workload: {self.workload_name.upper()}
Architecture: {self.arch_name.upper()}
Total Layers: {len(self.layers)}
Optimization Metric: {self.metric}
GPU Mode: {'Enabled' if self.gpu_id is not None else 'CPU Only'}
"""
            plt.text(0.1, 0.5, info_text, fontsize=12, 
                    verticalalignment='center', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            plt.axis('off')
            plt.title('Experiment Configuration', fontsize=14, fontweight='bold')
            
            plt.suptitle(f'DOSA Workload Overview: {self.workload_name.upper()}', 
                        fontsize=16, fontweight='bold')
            
            # Save plot
            plot_file = self.output_dir / f"workload_overview_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.debug(f"Created workload overview plot: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Failed to create workload overview plot: {e}")
            return None
    
    def _create_search_results_plot(self, result_files: List[pathlib.Path], timestamp: str) -> Optional[str]:
        """Create search results visualization."""
        try:
            plt.figure(figsize=(14, 10))
            gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
            
            # Load and analyze results
            all_results = []
            for file in result_files:
                if file.name.startswith(('network_search', 'random', 'bayesian')):
                    try:
                        result = FileHandler.load_json(file)
                        all_results.append((file.name, result))
                    except Exception as e:
                        logger.warning(f"Could not load result file {file}: {e}")
            
            if not all_results:
                logger.warning("No valid result files found for plotting")
                return None
            
            # Plot 1: Search timing
            plt.subplot(gs[0, 0])
            times = []
            strategies = []
            for filename, result in all_results:
                if 'search_results' in result:
                    for strategy, data in result['search_results'].items():
                        if isinstance(data, dict) and 'search_time' in data:
                            times.append(data['search_time'])
                            strategies.append(strategy)
            
            if times:
                plt.bar(strategies, times, color='lightgreen', alpha=0.7)
                plt.title('Search Strategy Performance')
                plt.xlabel('Strategy')
                plt.ylabel('Time (seconds)')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'No timing data available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Search Strategy Performance')
            
            # Plot 2: Call counts
            plt.subplot(gs[0, 1])
            calls = []
            call_strategies = []
            for filename, result in all_results:
                if 'search_results' in result:
                    for strategy, data in result['search_results'].items():
                        if isinstance(data, dict) and 'n_calls' in data:
                            calls.append(data['n_calls'])
                            call_strategies.append(strategy)
            
            if calls:
                plt.bar(call_strategies, calls, color='lightcoral', alpha=0.7)
                plt.title('Search Calls per Strategy')
                plt.xlabel('Strategy')
                plt.ylabel('Number of Calls')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'No call data available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Search Calls per Strategy')
            
            # Plot 3: Results summary
            plt.subplot(gs[1, :])
            summary_text = f"""
Search Results Summary:
• Total result files: {len(all_results)}
• Strategies executed: {len(set(strategies)) if strategies else 0}
• Total search time: {sum(times):.6f}s
• Average calls per strategy: {np.mean(calls):.1f} (if available)
• Output directory: {self.output_dir}
"""
            plt.text(0.1, 0.5, summary_text, fontsize=12,
                    verticalalignment='center', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
            plt.axis('off')
            plt.title('Search Execution Summary', fontsize=14, fontweight='bold')
            
            plt.suptitle('DOSA Search Results Analysis', fontsize=16, fontweight='bold')
            
            # Save plot
            plot_file = self.output_dir / f"search_results_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.debug(f"Created search results plot: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Failed to create search results plot: {e}")
            return None
    
    def _create_layer_analysis_plot(self, timestamp: str) -> Optional[str]:
        """Create layer analysis visualization."""
        try:
            plt.figure(figsize=(15, 10))
            gs = GridSpec(3, 2, hspace=0.4, wspace=0.3)
            
            # Extract layer information
            layer_info = []
            for i, layer in enumerate(self.layers):
                info = {
                    'index': i,
                    'name': layer.config_str(),
                    'count': self.layer_count[i] if i < len(self.layer_count) else 1,
                    'dimensions': {}
                }
                
                # Extract key dimensions
                for dim_name, dim_value in layer.prob.items():
                    info['dimensions'][dim_name] = dim_value
                
                layer_info.append(info)
            
            # Plot 1: Layer dimension heatmap
            plt.subplot(gs[0, :])
            dim_names = ['M', 'K', 'N', 'P', 'Q', 'R', 'S']
            available_dims = []
            heatmap_data = []
            
            for dim in dim_names:
                dim_values = []
                for layer in layer_info:
                    if dim in layer['dimensions']:
                        dim_values.append(layer['dimensions'][dim])
                    else:
                        dim_values.append(0)
                
                if any(v > 0 for v in dim_values):
                    available_dims.append(dim)
                    heatmap_data.append(dim_values)
            
            if heatmap_data:
                heatmap_data = np.array(heatmap_data)
                im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
                plt.colorbar(im, label='Dimension Value')
                plt.yticks(range(len(available_dims)), available_dims)
                plt.xticks(range(len(layer_info)), [f'L{i+1}' for i in range(len(layer_info))])
                plt.title('Layer Dimension Heatmap')
                plt.xlabel('Layers')
                plt.ylabel('Dimensions')
            
            # Plot 2: Memory requirements
            plt.subplot(gs[1, 0])
            memory_reqs = []
            for layer in layer_info:
                # Estimate memory requirement
                mem = 1
                for dim_name in ['M', 'K', 'N']:
                    if dim_name in layer['dimensions']:
                        mem *= layer['dimensions'][dim_name]
                memory_reqs.append(mem)
            
            plt.bar(range(len(memory_reqs)), memory_reqs, color='orange', alpha=0.7)
            plt.title('Estimated Memory Requirements')
            plt.xlabel('Layer Index')
            plt.ylabel('Memory (relative)')
            plt.yscale('log')
            
            # Plot 3: Computational intensity
            plt.subplot(gs[1, 1])
            intensities = []
            for layer in layer_info:
                # Calculate computational intensity
                intensity = 1
                for dim_value in layer['dimensions'].values():
                    intensity *= dim_value
                intensities.append(intensity)
            
            plt.scatter(range(len(intensities)), intensities, 
                       color='purple', s=100, alpha=0.7)
            plt.title('Computational Intensity')
            plt.xlabel('Layer Index')
            plt.ylabel('Operations (relative)')
            plt.yscale('log')
            
            # Plot 4: Layer details table
            plt.subplot(gs[2, :])
            table_data = []
            for i, layer in enumerate(layer_info):
                row = [
                    f"L{i+1}",
                    f"Count: {layer['count']}",
                    f"Dims: {len(layer['dimensions'])}",
                    layer['name'][:50] + "..." if len(layer['name']) > 50 else layer['name']
                ]
                table_data.append(row)
            
            table = plt.table(cellText=table_data,
                            colLabels=['Layer', 'Count', 'Dims', 'Configuration'],
                            cellLoc='left',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            plt.axis('off')
            plt.title('Layer Details', fontsize=12, fontweight='bold')
            
            plt.suptitle(f'Layer Analysis: {self.workload_name.upper()}', 
                        fontsize=16, fontweight='bold')
            
            # Save plot
            plot_file = self.output_dir / f"layer_analysis_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.debug(f"Created layer analysis plot: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Failed to create layer analysis plot: {e}")
            return None
    
    def _create_comprehensive_dashboard(self, result_files: List[pathlib.Path], timestamp: str) -> Optional[str]:
        """Create comprehensive dashboard visualization."""
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, hspace=0.4, wspace=0.3)
            
            # Title
            fig.suptitle(f'DOSA Comprehensive Dashboard - {self.workload_name.upper()} on {self.arch_name.upper()}',
                        fontsize=20, fontweight='bold', y=0.95)
            
            # Load results
            all_data = {}
            for file in result_files:
                try:
                    data = FileHandler.load_json(file)
                    all_data[file.name] = data
                except:
                    continue
            
            # Dashboard sections
            
            # 1. Experiment Overview (top-left)
            ax1 = plt.subplot(gs[0, 0:2])
            overview_text = f"""
🎯 EXPERIMENT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Workload: {self.workload_name.upper()}
🏗️  Architecture: {self.arch_name.upper()}
🔧 Optimization Metric: {self.metric}
📁 Layers: {len(self.layers)}
🖥️  Processing Mode: {'GPU' if self.gpu_id is not None else 'CPU'}
📋 Result Files: {len(result_files)}

🔍 LAYER CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            for i, layer in enumerate(self.layers[:3]):  # Show first 3 layers
                overview_text += f"L{i+1}: {layer.config_str()[:50]}...\n"
            if len(self.layers) > 3:
                overview_text += f"... and {len(self.layers)-3} more layers"
            
            ax1.text(0.05, 0.95, overview_text, fontsize=10, fontfamily='monospace',
                    verticalalignment='top', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
            ax1.axis('off')
            
            # 2. Performance Metrics (top-right)
            ax2 = plt.subplot(gs[0, 2:])
            perf_metrics = []
            for filename, data in all_data.items():
                if 'search_results' in data:
                    for strategy, results in data['search_results'].items():
                        if isinstance(results, dict):
                            perf_metrics.append({
                                'strategy': strategy,
                                'time': results.get('search_time', 0),
                                'calls': results.get('n_calls', 0)
                            })
            
            if perf_metrics:
                strategies = [m['strategy'] for m in perf_metrics]
                times = [m['time'] for m in perf_metrics]
                calls = [m['calls'] for m in perf_metrics]
                
                ax2_twin = ax2.twinx()
                bars1 = ax2.bar([s + ' (T)' for s in strategies], times, 
                               alpha=0.7, color='skyblue', label='Time (s)')
                bars2 = ax2_twin.bar([s + ' (C)' for s in strategies], calls, 
                                    alpha=0.7, color='lightcoral', label='Calls')
                
                ax2.set_ylabel('Time (seconds)', color='blue')
                ax2_twin.set_ylabel('Number of Calls', color='red')
                ax2.tick_params(axis='x', rotation=45)
                ax2.set_title('Performance Metrics')
            else:
                ax2.text(0.5, 0.5, 'No performance data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Performance Metrics')
            
            # 3. Layer Statistics (middle-left)
            ax3 = plt.subplot(gs[1, 0:2])
            layer_complexities = []
            for layer in self.layers:
                complexity = 1
                for dim_name, dim_value in layer.prob.items():
                    if dim_name in ['M', 'K', 'N']:
                        complexity *= dim_value
                layer_complexities.append(complexity)
            
            x_pos = range(len(layer_complexities))
            bars = ax3.bar(x_pos, layer_complexities, color='green', alpha=0.6)
            ax3.set_yscale('log')
            ax3.set_xlabel('Layer Index')
            ax3.set_ylabel('Computational Complexity')
            ax3.set_title('Layer Complexity Distribution')
            
            # Add layer counts as text on bars
            for i, (bar, count) in enumerate(zip(bars, self.layer_count)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height*1.1,
                        f'×{count}', ha='center', va='bottom', fontsize=8)
            
            # 4. Search Strategy Comparison (middle-right)
            ax4 = plt.subplot(gs[1, 2:])
            strategy_summary = {}
            for filename, data in all_data.items():
                if 'search_results' in data:
                    for strategy, results in data['search_results'].items():
                        if strategy not in strategy_summary:
                            strategy_summary[strategy] = {
                                'executions': 0, 'total_time': 0, 'total_calls': 0}
                        
                        strategy_summary[strategy]['executions'] += 1
                        if isinstance(results, dict):
                            strategy_summary[strategy]['total_time'] += results.get('search_time', 0)
                            strategy_summary[strategy]['total_calls'] += results.get('n_calls', 0)
            
            if strategy_summary:
                strategies = list(strategy_summary.keys())
                avg_times = [strategy_summary[s]['total_time']/strategy_summary[s]['executions'] 
                           for s in strategies]
                
                wedges, texts, autotexts = ax4.pie(avg_times, labels=strategies, autopct='%1.1f%%',
                                                  startangle=90, colors=['gold', 'lightcoral', 'lightblue'])
                ax4.set_title('Search Time Distribution')
            else:
                ax4.text(0.5, 0.5, 'No strategy data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Search Time Distribution')
            
            # 5. Summary Statistics (bottom)
            ax5 = plt.subplot(gs[2, :])
            total_time = sum(perf_metrics[i]['time'] for i in range(len(perf_metrics))) if perf_metrics else 0
            total_calls = sum(perf_metrics[i]['calls'] for i in range(len(perf_metrics))) if perf_metrics else 0
            
            summary_stats = f"""
📈 EXECUTION STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 Total Execution Time: {total_time:.6f} seconds            🎯 Total Search Calls: {total_calls}
📊 Average Time per Call: {(total_time/total_calls if total_calls > 0 else 0):.8f} seconds     🔄 Strategies Executed: {len(set(m['strategy'] for m in perf_metrics)) if perf_metrics else 0}
📁 Output Directory: {self.output_dir}
🏷️  Timestamp: {timestamp}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            ax5.text(0.02, 0.5, summary_stats, fontsize=11, fontfamily='monospace',
                    verticalalignment='center', transform=ax5.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))
            ax5.axis('off')
            
            # Save dashboard
            plot_file = self.output_dir / f"comprehensive_dashboard_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.debug(f"Created comprehensive dashboard: {plot_file}")
            return str(plot_file)
            
        except Exception as e:
            logger.error(f"Failed to create comprehensive dashboard: {e}")
            return None 