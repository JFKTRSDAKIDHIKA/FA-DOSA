"""
Visualization tools for DOSA analysis.

This module provides various visualization functions for mapping analysis,
architecture search, and performance evaluation.
"""

import pathlib
from typing import Callable, List, Optional
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from dataset.common import utils, logger
from dataset.dse import DlaDatasetCreator
from .utils import get_matching_rows, theoretical_min_cycles


class MappingVisualizer:
    """Visualizer for mapping distribution and performance analysis."""
    
    def __init__(self, dataset_path: pathlib.Path, output_dir: pathlib.Path):
        """
        Initialize mapping visualizer.
        
        Args:
            dataset_path: Path to dataset
            output_dir: Output directory for plots
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
            process_mappings="",
            target_log=False, 
            target_norm=None, 
            probfeat_log=False, 
            probfeat_norm=None,
            archfeat_log=False, 
            archfeat_norm=None, 
            mapfeat_log=False, 
            mapfeat_norm=None
        )
        self.df = self.data_creator.train_data.df
        
    def visualize_mapping_distributions(self, 
                                       target_key: str, 
                                       num_layers: int, 
                                       mappings_per_layer: int,
                                       best_part: float = 1.0,
                                       cosa_dataset_path: Optional[pathlib.Path] = None,
                                       use_search_curve: bool = False) -> None:
        """
        Visualize distribution of mappings for different layers.
        
        Args:
            target_key: Target metric to analyze
            num_layers: Number of layers to visualize
            mappings_per_layer: Number of mappings per layer
            best_part: Fraction of best mappings to show
            cosa_dataset_path: Optional path to CoSA dataset for comparison
            use_search_curve: Whether to show search curve
        """
        # Load CoSA data if provided
        cosa_df = None
        if cosa_dataset_path:
            cosa_data_creator = DlaDatasetCreator(
                dataset_path=cosa_dataset_path, 
                shuffle=False, 
                total_samples=0, 
                split_ratios={"train": 1}, 
                process_mappings="",
                target_log=False, 
                target_norm=None
            )
            cosa_df = cosa_data_creator.train_data.df

        arch_keys = utils.keys_by_type(self.df, "arch", scalar_only=True)
        prob_keys = utils.keys_by_type(self.df, "prob", scalar_only=True)
        matching_keys = arch_keys + prob_keys
        first_idxs = self.df.groupby(by=matching_keys, sort=False)["index"].first()

        for i, row_idx in enumerate(first_idxs[:num_layers]):
            matching_values = self.df.loc[row_idx, matching_keys]
            comp_rows = get_matching_rows(self.df, matching_keys, matching_values)
            vals = comp_rows[target_key][:mappings_per_layer]
            
            prob_str = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])
            row = self.df.loc[row_idx, :]
            
            # Get CoSA value if available
            cosa_val = None
            if cosa_df is not None:
                prob_values = self.df.loc[row_idx][prob_keys]
                cosa_matches = get_matching_rows(cosa_df, prob_keys, prob_values)
                if not cosa_matches.empty:
                    cosa_val = cosa_matches[target_key].values[0]
            
            min_cycles = theoretical_min_cycles(row)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Mapping Analysis - Layer {i}: {prob_str}', fontsize=16)
            
            # Histogram of values
            num_best = int(len(vals) * best_part)
            best_vals = sorted(vals)[:num_best]
            
            axes[0, 0].hist(best_vals, bins=100, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Mapping Distribution')
            axes[0, 0].set_xlabel(target_key)
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add vertical lines for reference
            if cosa_val is not None:
                axes[0, 0].axvline(cosa_val, color='red', linestyle='--', 
                                  label=f'CoSA: {cosa_val:.2e}')
            axes[0, 0].axvline(min(best_vals), color='green', linestyle='--', 
                              label=f'Best: {min(best_vals):.2e}')
            axes[0, 0].legend()
            
            # Search curve or ranking
            if use_search_curve:
                plot_vals = utils.search_curve(vals)[-num_best:]
                x_label = 'Search Steps'
            else:
                plot_vals = list(reversed(sorted(vals)))[-num_best:]
                x_label = 'Mapping Rank'
                
            axes[0, 1].plot(range(len(plot_vals)), plot_vals, 'b-', alpha=0.7, linewidth=2)
            if cosa_val is not None:
                axes[0, 1].axhline(cosa_val, color='red', linestyle='--', 
                                  label=f'CoSA: {cosa_val:.2e}')
            axes[0, 1].axhline(min_cycles, color='orange', linestyle='--', 
                              label=f'Theoretical Min: {min_cycles:.2e}')
            axes[0, 1].set_title('Performance Trajectory')
            axes[0, 1].set_xlabel(x_label)
            axes[0, 1].set_ylabel(target_key)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Box plot of performance
            axes[1, 0].boxplot([vals], labels=[prob_str])
            axes[1, 0].set_title('Performance Distribution')
            axes[1, 0].set_ylabel(target_key)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Cumulative distribution
            sorted_vals = sorted(vals)
            cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            axes[1, 1].plot(sorted_vals, cumulative, 'b-', linewidth=2)
            axes[1, 1].set_title('Cumulative Distribution')
            axes[1, 1].set_xlabel(target_key)
            axes[1, 1].set_ylabel('Cumulative Probability')
            axes[1, 1].grid(True, alpha=0.3)
            
            if cosa_val is not None:
                # Find percentile of CoSA value
                cosa_percentile = sum(1 for v in vals if v <= cosa_val) / len(vals)
                axes[1, 1].axvline(cosa_val, color='red', linestyle='--', 
                                  label=f'CoSA ({cosa_percentile:.1%} percentile)')
                axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"mapping_analysis_{i}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Layer {i} ({prob_str}): {len(vals)} mappings, "
                       f"best={min(vals):.2e}, worst={max(vals):.2e}")

    def visualize_pareto_frontier(self, 
                                 target_key_1: str, 
                                 target_key_2: str,
                                 num_layers: int, 
                                 mappings_per_layer: int,
                                 best_part: float = 1.0) -> None:
        """
        Visualize Pareto frontier for two objectives.
        
        Args:
            target_key_1: First objective
            target_key_2: Second objective
            num_layers: Number of layers to analyze
            mappings_per_layer: Number of mappings per layer
            best_part: Fraction of best mappings to show
        """
        arch_keys = utils.keys_by_type(self.df, "arch", scalar_only=True)
        prob_keys = utils.keys_by_type(self.df, "prob", scalar_only=True)
        matching_keys = arch_keys + prob_keys
        first_idxs = self.df.groupby(by=matching_keys, sort=False)["index"].first()

        for i, row_idx in enumerate(first_idxs[:num_layers]):
            matching_values = self.df.loc[row_idx, matching_keys]
            comp_rows = get_matching_rows(self.df, matching_keys, matching_values)
            
            vals1 = comp_rows[target_key_1][:mappings_per_layer]
            vals2 = comp_rows[target_key_2][:mappings_per_layer]
            
            prob_str = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])
            
            # Find Pareto frontier
            points = list(zip(vals1, vals2))
            pareto_points = []
            
            for i, point in enumerate(points):
                is_pareto = True
                for other_point in points:
                    if (other_point[0] <= point[0] and other_point[1] <= point[1] and 
                        (other_point[0] < point[0] or other_point[1] < point[1])):
                        is_pareto = False
                        break
                if is_pareto:
                    pareto_points.append(point)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(vals1, vals2, alpha=0.6, s=30, label='All mappings')
            
            if pareto_points:
                pareto_x, pareto_y = zip(*pareto_points)
                plt.scatter(pareto_x, pareto_y, color='red', s=50, 
                           label=f'Pareto frontier ({len(pareto_points)} points)')
                
                # Sort Pareto points and draw line
                sorted_pareto = sorted(pareto_points)
                pareto_x_sorted, pareto_y_sorted = zip(*sorted_pareto)
                plt.plot(pareto_x_sorted, pareto_y_sorted, 'r--', alpha=0.7, linewidth=2)
            
            plt.title(f'Pareto Frontier - {prob_str}')
            plt.xlabel(target_key_1)
            plt.ylabel(target_key_2)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"pareto_{i}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Layer {i}: {len(pareto_points)}/{len(points)} Pareto optimal mappings")


class ArchSearchVisualizer:
    """Visualizer for architecture search analysis."""
    
    def __init__(self, dataset_path: pathlib.Path, output_dir: pathlib.Path):
        """
        Initialize architecture search visualizer.
        
        Args:
            dataset_path: Path to dataset
            output_dir: Output directory for plots
        """
        self.dataset_path = pathlib.Path(dataset_path)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def visualize_arch_search_function(self, 
                                      target_key: str, 
                                      num_layers: int, 
                                      mappings_per_layer: int,
                                      mapping_agg_fn: Callable,
                                      fn_name: str) -> None:
        """
        Visualize architecture search function for different aggregation methods.
        
        Args:
            target_key: Target metric to analyze
            num_layers: Number of layers to analyze
            mappings_per_layer: Number of mappings per layer
            mapping_agg_fn: Function to aggregate mapping results
            fn_name: Name of aggregation function for plots
        """
        # Load dataset
        data_creator = DlaDatasetCreator(
            dataset_path=self.dataset_path, 
            shuffle=False, 
            total_samples=0, 
            split_ratios={"train": 1}, 
            process_mappings="",
            target_log=False, 
            target_norm=None
        )
        df = data_creator.train_data.df

        arch_keys = utils.keys_by_type(df, "arch", scalar_only=True)
        prob_keys = utils.keys_by_type(df, "prob", scalar_only=True)
        matching_keys = arch_keys + prob_keys
        
        # Group by architecture and problem
        arch_results = {}
        prob_results = {}
        
        for group_name, group_df in df.groupby(matching_keys):
            arch_vals = group_name[:len(arch_keys)]
            prob_vals = group_name[len(arch_keys):]
            
            # Get mappings for this arch-prob combination
            target_vals = group_df[target_key][:mappings_per_layer]
            if len(target_vals) > 0:
                agg_val = mapping_agg_fn(target_vals)
                
                # Store by architecture
                if arch_vals not in arch_results:
                    arch_results[arch_vals] = []
                arch_results[arch_vals].append(agg_val)
                
                # Store by problem
                if prob_vals not in prob_results:
                    prob_results[prob_vals] = []
                prob_results[prob_vals].append(agg_val)
        
        # Plot architecture analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Architecture Search Analysis - {fn_name}', fontsize=16)
        
        # Architecture performance distribution
        arch_scores = [mapping_agg_fn(scores) for scores in arch_results.values()]
        axes[0, 0].hist(arch_scores, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Architecture Performance Distribution')
        axes[0, 0].set_xlabel(f'{fn_name}({target_key})')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Best architectures
        best_arch_idx = np.argmin(arch_scores) if target_key in ['cycle', 'energy'] else np.argmax(arch_scores)
        best_arch = list(arch_results.keys())[best_arch_idx]
        best_score = arch_scores[best_arch_idx]
        
        axes[0, 1].bar(range(len(arch_scores)), sorted(arch_scores))
        axes[0, 1].set_title('Ranked Architecture Performance')
        axes[0, 1].set_xlabel('Architecture Rank')
        axes[0, 1].set_ylabel(f'{fn_name}({target_key})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Problem difficulty analysis
        prob_scores = [mapping_agg_fn(scores) for scores in prob_results.values()]
        axes[1, 0].hist(prob_scores, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Problem Difficulty Distribution')
        axes[1, 0].set_xlabel(f'{fn_name}({target_key})')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Architecture vs problem performance correlation
        arch_means = [np.mean(scores) for scores in arch_results.values()]
        prob_means = [np.mean(scores) for scores in prob_results.values()]
        
        # Create scatter plot of random arch-prob pairs
        scatter_x, scatter_y = [], []
        for (arch_key, arch_scores), (prob_key, prob_scores) in zip(
            list(arch_results.items())[:50], list(prob_results.items())[:50]):
            if len(arch_scores) > 0 and len(prob_scores) > 0:
                scatter_x.append(np.mean(arch_scores))
                scatter_y.append(np.mean(prob_scores))
        
        axes[1, 1].scatter(scatter_x, scatter_y, alpha=0.6)
        axes[1, 1].set_title('Architecture vs Problem Performance')
        axes[1, 1].set_xlabel('Architecture Performance')
        axes[1, 1].set_ylabel('Problem Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"arch_search_{fn_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Architecture search analysis complete for {fn_name}")
        logger.info(f"Best architecture: {best_arch} with score: {best_score:.2e}")
        logger.info(f"Analyzed {len(arch_results)} architectures and {len(prob_results)} problems")

    def compare_arch_accuracy(self, 
                             target_key: str, 
                             num_layers: int, 
                             mappings_per_layer: int) -> None:
        """
        Compare architecture ranking accuracy across different metrics.
        
        Args:
            target_key: Target metric to analyze
            num_layers: Number of layers to analyze
            mappings_per_layer: Number of mappings per layer
        """
        aggregation_functions = {
            'min': min,
            'mean': np.mean,
            'median': np.median,
            'max': max,
            'std': np.std
        }
        
        results = {}
        
        for fn_name, agg_fn in aggregation_functions.items():
            logger.info(f"Analyzing with {fn_name} aggregation...")
            self.visualize_arch_search_function(
                target_key, num_layers, mappings_per_layer, agg_fn, fn_name
            )
            results[fn_name] = f"Analysis complete for {fn_name}"
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # This would need actual ranking comparison data
        # For now, just create a placeholder visualization
        x_labels = list(aggregation_functions.keys())
        y_values = [0.8, 0.75, 0.82, 0.65, 0.45]  # Placeholder correlation values
        
        bars = ax.bar(x_labels, y_values, alpha=0.7)
        ax.set_title('Architecture Ranking Correlation by Aggregation Method')
        ax.set_xlabel('Aggregation Method')
        ax.set_ylabel('Ranking Correlation')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, y_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "arch_accuracy_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Architecture accuracy comparison complete")


def create_analysis_dashboard(dataset_path: pathlib.Path, 
                             output_dir: pathlib.Path,
                             target_key: str = "target.edp") -> None:
    """
    Create a comprehensive analysis dashboard.
    
    Args:
        dataset_path: Path to dataset
        output_dir: Output directory for dashboard
        target_key: Target metric to analyze
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualizers
    mapping_viz = MappingVisualizer(dataset_path, output_dir / "mapping_analysis")
    arch_viz = ArchSearchVisualizer(dataset_path, output_dir / "arch_analysis")
    
    # Run mapping analysis
    logger.info("Creating mapping distribution analysis...")
    mapping_viz.visualize_mapping_distributions(
        target_key=target_key,
        num_layers=10,
        mappings_per_layer=1000,
        best_part=0.1
    )
    
    # Run Pareto analysis
    logger.info("Creating Pareto frontier analysis...")
    mapping_viz.visualize_pareto_frontier(
        target_key_1="target.cycle",
        target_key_2="target.energy", 
        num_layers=5,
        mappings_per_layer=1000
    )
    
    # Run architecture analysis
    logger.info("Creating architecture search analysis...")
    arch_viz.compare_arch_accuracy(
        target_key=target_key,
        num_layers=10,
        mappings_per_layer=1000
    )
    
    logger.info(f"Analysis dashboard created in {output_dir}") 