"""
Performance analyzer for DOSA.

This module provides comprehensive performance analysis capabilities
for mapping and architecture evaluation.
"""

import pathlib
from typing import Dict, List, Tuple, Optional
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from dataset.common import utils, logger
from dataset.dse import DlaDatasetCreator
from .utils import get_matching_rows, theoretical_min_cycles
from .mlp_predictor import run_mlp_experiment


class PerformanceAnalyzer:
    """Comprehensive performance analyzer for mapping and architecture evaluation."""
    
    def __init__(self, dataset_path: pathlib.Path, output_dir: pathlib.Path):
        """
        Initialize performance analyzer.
        
        Args:
            dataset_path: Path to dataset
            output_dir: Output directory for results
        """
        self.dataset_path = pathlib.Path(dataset_path)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        self.data_creator = DlaDatasetCreator(
            dataset_path=dataset_path, 
            shuffle=False, 
            total_samples=0, 
            split_ratios={"train": 1}, 
            process_mappings="split",
            target_log=False, 
            target_norm=None
        )
        self.df = self.data_creator.train_data.df
        
    def analyze_prediction_accuracy(self, target_key: str) -> Dict[str, float]:
        """
        Analyze prediction accuracy with different feature sets.
        
        Args:
            target_key: Target metric to predict
            
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info(f"Analyzing prediction accuracy for {target_key}")
        
        # Create train/test split
        split_creator = DlaDatasetCreator(
            dataset_path=self.dataset_path,
            total_samples=1000,
            split_ratios={"train": 0.8, "test": 0.2},
            process_mappings="split"
        )
        
        train_data = split_creator.train_data
        test_data = split_creator.test_data
        
        results = {}
        
        # Test different feature combinations
        feature_sets = {
            'prob_only': ["prob"],
            'mapping_only': ["mapping"], 
            'prob_mapping': ["prob", "mapping"],
            'arch_prob': ["arch", "prob"],
            'all_features': ["arch", "prob", "mapping"]
        }
        
        for set_name, features in feature_sets.items():
            logger.info(f"Testing feature set: {set_name}")
            
            try:
                y_true, y_pred = run_mlp_experiment(train_data, test_data, features, target_key)
                
                mse = mean_squared_error(y_true, y_pred)
                mape = mean_absolute_percentage_error(y_true, y_pred)
                
                results[set_name] = {
                    'mse': float(mse),
                    'mape': float(mape),
                    'rmse': float(np.sqrt(mse)),
                    'features': features
                }
                
                logger.info(f"{set_name}: MSE={mse:.6f}, MAPE={mape:.6f}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {set_name}: {e}")
                results[set_name] = {'error': str(e)}
        
        # Save results
        utils.store_json(self.output_dir / "prediction_accuracy.json", results, indent=4)
        
        # Create visualization
        self._plot_accuracy_comparison(results, target_key)
        
        return results
    
    def _plot_accuracy_comparison(self, results: Dict, target_key: str) -> None:
        """Plot comparison of prediction accuracy."""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            logger.warning("No valid results to plot")
            return
            
        feature_names = list(valid_results.keys())
        mse_values = [results[name]['mse'] for name in feature_names]
        mape_values = [results[name]['mape'] for name in feature_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MSE comparison
        bars1 = ax1.bar(feature_names, mse_values, alpha=0.7)
        ax1.set_title(f'Mean Squared Error - {target_key}')
        ax1.set_ylabel('MSE')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, mse_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom')
        
        # MAPE comparison
        bars2 = ax2.bar(feature_names, mape_values, alpha=0.7, color='orange')
        ax2.set_title(f'Mean Absolute Percentage Error - {target_key}')
        ax2.set_ylabel('MAPE')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, mape_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"accuracy_comparison_{target_key}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_mapping_efficiency(self, target_key: str, num_layers: int = 10) -> Dict:
        """
        Analyze mapping efficiency across different problems.
        
        Args:
            target_key: Target metric to analyze
            num_layers: Number of layers to analyze
            
        Returns:
            Dictionary with efficiency metrics
        """
        logger.info(f"Analyzing mapping efficiency for {target_key}")
        
        arch_keys = utils.keys_by_type(self.df, "arch", scalar_only=True)
        prob_keys = utils.keys_by_type(self.df, "prob", scalar_only=True)
        matching_keys = arch_keys + prob_keys
        
        efficiency_results = []
        
        # Group by architecture and problem
        grouped = self.df.groupby(matching_keys)
        
        for i, (group_name, group_df) in enumerate(grouped):
            if i >= num_layers:
                break
                
            if len(group_df) < 10:  # Skip small groups
                continue
                
            arch_vals = group_name[:len(arch_keys)]
            prob_vals = group_name[len(arch_keys):]
            
            # Get performance values
            performance_vals = group_df[target_key].values
            
            # Calculate theoretical minimum
            row = group_df.iloc[0]
            theoretical_min = theoretical_min_cycles(row)
            
            # Calculate efficiency metrics
            best_actual = performance_vals.min()
            worst_actual = performance_vals.max()
            mean_actual = performance_vals.mean()
            
            efficiency_best = theoretical_min / best_actual if best_actual > 0 else 0
            efficiency_mean = theoretical_min / mean_actual if mean_actual > 0 else 0
            
            # Calculate search efficiency (how quickly good solutions are found)
            sorted_vals = sorted(performance_vals)
            search_curve = utils.search_curve(performance_vals)
            
            # Find 90th percentile performance threshold
            threshold_90 = np.percentile(sorted_vals, 10)  # 10th percentile = top 90%
            steps_to_90 = next((i for i, val in enumerate(search_curve) if val <= threshold_90), len(search_curve))
            search_efficiency = steps_to_90 / len(search_curve)
            
            result = {
                'layer_id': i,
                'arch_params': arch_vals,
                'prob_params': prob_vals,
                'prob_shape': "_".join([str(int(v)) for v in prob_vals]),
                'num_mappings': len(performance_vals),
                'theoretical_min': float(theoretical_min),
                'best_actual': float(best_actual),
                'worst_actual': float(worst_actual),
                'mean_actual': float(mean_actual),
                'efficiency_best': float(efficiency_best),
                'efficiency_mean': float(efficiency_mean),
                'search_efficiency': float(search_efficiency),
                'performance_range': float(worst_actual - best_actual),
                'performance_cv': float(performance_vals.std() / mean_actual) if mean_actual > 0 else 0
            }
            
            efficiency_results.append(result)
        
        # Convert to DataFrame for analysis
        efficiency_df = pd.DataFrame(efficiency_results)
        
        # Save detailed results
        efficiency_df.to_csv(self.output_dir / f"mapping_efficiency_{target_key}.csv", index=False)
        
        # Calculate summary statistics
        summary = {
            'mean_efficiency_best': float(efficiency_df['efficiency_best'].mean()),
            'mean_efficiency_mean': float(efficiency_df['efficiency_mean'].mean()),
            'mean_search_efficiency': float(efficiency_df['search_efficiency'].mean()),
            'layers_analyzed': len(efficiency_results),
            'target_key': target_key
        }
        
        # Create visualizations
        self._plot_efficiency_analysis(efficiency_df, target_key)
        
        logger.info(f"Efficiency analysis complete. Mean efficiency: {summary['mean_efficiency_best']:.3f}")
        
        return summary
    
    def _plot_efficiency_analysis(self, efficiency_df: pd.DataFrame, target_key: str) -> None:
        """Plot efficiency analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Mapping Efficiency Analysis - {target_key}', fontsize=16)
        
        # Efficiency distribution
        axes[0, 0].hist(efficiency_df['efficiency_best'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Best Mapping Efficiency Distribution')
        axes[0, 0].set_xlabel('Efficiency (Theoretical/Actual)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Search efficiency vs performance efficiency
        axes[0, 1].scatter(efficiency_df['efficiency_best'], efficiency_df['search_efficiency'], alpha=0.6)
        axes[0, 1].set_title('Performance vs Search Efficiency')
        axes[0, 1].set_xlabel('Best Mapping Efficiency')
        axes[0, 1].set_ylabel('Search Efficiency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance range analysis
        axes[1, 0].scatter(efficiency_df['num_mappings'], efficiency_df['performance_range'], alpha=0.6)
        axes[1, 0].set_title('Mapping Count vs Performance Range')
        axes[1, 0].set_xlabel('Number of Mappings')
        axes[1, 0].set_ylabel('Performance Range')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency by layer
        layer_ids = efficiency_df['layer_id']
        efficiencies = efficiency_df['efficiency_best']
        axes[1, 1].bar(layer_ids, efficiencies, alpha=0.7)
        axes[1, 1].set_title('Efficiency by Layer')
        axes[1, 1].set_xlabel('Layer ID')
        axes[1, 1].set_ylabel('Best Mapping Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"efficiency_analysis_{target_key}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_architectures(self, target_key: str) -> Dict:
        """
        Compare performance across different architectures.
        
        Args:
            target_key: Target metric to analyze
            
        Returns:
            Dictionary with architecture comparison results
        """
        logger.info(f"Comparing architectures for {target_key}")
        
        arch_keys = utils.keys_by_type(self.df, "arch", scalar_only=True)
        
        # Group by architecture
        arch_results = {}
        
        for group_name, group_df in self.df.groupby(arch_keys):
            if len(group_df) < 10:  # Skip architectures with too few samples
                continue
                
            performance_vals = group_df[target_key].values
            
            arch_results[group_name] = {
                'num_samples': len(performance_vals),
                'best': float(performance_vals.min()),
                'worst': float(performance_vals.max()),
                'mean': float(performance_vals.mean()),
                'std': float(performance_vals.std()),
                'median': float(np.median(performance_vals)),
                'q25': float(np.percentile(performance_vals, 25)),
                'q75': float(np.percentile(performance_vals, 75))
            }
        
        # Find best and worst architectures
        if target_key in ['cycle', 'energy', 'edp']:  # Lower is better
            best_arch = min(arch_results.keys(), key=lambda x: arch_results[x]['mean'])
            worst_arch = max(arch_results.keys(), key=lambda x: arch_results[x]['mean'])
        else:  # Higher is better
            best_arch = max(arch_results.keys(), key=lambda x: arch_results[x]['mean'])
            worst_arch = min(arch_results.keys(), key=lambda x: arch_results[x]['mean'])
        
        comparison_summary = {
            'num_architectures': len(arch_results),
            'best_architecture': {
                'params': best_arch,
                'metrics': arch_results[best_arch]
            },
            'worst_architecture': {
                'params': worst_arch, 
                'metrics': arch_results[worst_arch]
            },
            'target_key': target_key
        }
        
        # Save detailed results
        utils.store_json(self.output_dir / f"architecture_comparison_{target_key}.json", 
                        {str(k): v for k, v in arch_results.items()}, indent=4)
        
        # Create visualization
        self._plot_architecture_comparison(arch_results, target_key)
        
        logger.info(f"Architecture comparison complete. Best: {best_arch}")
        
        return comparison_summary
    
    def _plot_architecture_comparison(self, arch_results: Dict, target_key: str) -> None:
        """Plot architecture comparison results."""
        if len(arch_results) == 0:
            logger.warning("No architecture results to plot")
            return
            
        # Prepare data
        arch_names = [str(arch)[:50] for arch in arch_results.keys()]  # Truncate long names
        mean_values = [results['mean'] for results in arch_results.values()]
        std_values = [results['std'] for results in arch_results.values()]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Mean performance comparison
        bars = ax1.bar(range(len(arch_names)), mean_values, yerr=std_values, 
                      alpha=0.7, capsize=5)
        ax1.set_title(f'Architecture Performance Comparison - {target_key}')
        ax1.set_xlabel('Architecture')
        ax1.set_ylabel(f'Mean {target_key}')
        ax1.set_xticks(range(len(arch_names)))
        ax1.set_xticklabels(arch_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Box plot for distribution comparison
        performance_data = []
        for arch, results in arch_results.items():
            # Create synthetic data points for box plot
            mean_val = results['mean']
            std_val = results['std']
            synthetic_data = np.random.normal(mean_val, std_val, 100)
            performance_data.append(synthetic_data)
        
        if performance_data:
            box_plot = ax2.boxplot(performance_data, labels=arch_names, patch_artist=True)
            ax2.set_title(f'Architecture Performance Distribution - {target_key}')
            ax2.set_xlabel('Architecture')
            ax2.set_ylabel(f'{target_key}')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Color boxes
            colors = plt.cm.viridis(np.linspace(0, 1, len(box_plot['boxes'])))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"architecture_comparison_{target_key}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_performance_report(self, target_keys: List[str] = None) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Args:
            target_keys: List of target metrics to analyze
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if target_keys is None:
            target_keys = ["target.cycle", "target.energy", "target.edp"]
        
        logger.info("Generating comprehensive performance report...")
        
        report = {
            'dataset_path': str(self.dataset_path),
            'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': len(self.df),
            'results': {}
        }
        
        for target_key in target_keys:
            logger.info(f"Analyzing {target_key}...")
            
            target_results = {
                'prediction_accuracy': {},
                'mapping_efficiency': {},
                'architecture_comparison': {}
            }
            
            try:
                # Prediction accuracy analysis
                target_results['prediction_accuracy'] = self.analyze_prediction_accuracy(target_key)
            except Exception as e:
                logger.error(f"Prediction accuracy analysis failed for {target_key}: {e}")
                target_results['prediction_accuracy'] = {'error': str(e)}
            
            try:
                # Mapping efficiency analysis
                target_results['mapping_efficiency'] = self.analyze_mapping_efficiency(target_key)
            except Exception as e:
                logger.error(f"Mapping efficiency analysis failed for {target_key}: {e}")
                target_results['mapping_efficiency'] = {'error': str(e)}
            
            try:
                # Architecture comparison
                target_results['architecture_comparison'] = self.compare_architectures(target_key)
            except Exception as e:
                logger.error(f"Architecture comparison failed for {target_key}: {e}")
                target_results['architecture_comparison'] = {'error': str(e)}
            
            report['results'][target_key] = target_results
        
        # Save comprehensive report
        utils.store_json(self.output_dir / "performance_report.json", report, indent=4)
        
        logger.info(f"Performance report generated and saved to {self.output_dir}")
        
        return report 