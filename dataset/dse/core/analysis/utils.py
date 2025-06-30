"""
Analysis utilities for DOSA.

This module provides common utility functions for analysis tasks.
"""

import math
import pathlib
from typing import Iterable, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset.common import utils, logger
from dataset.dse import DlaDatasetCreator


def theoretical_min_cycles(row: pd.Series) -> int:
    """
    Calculate theoretical minimum cycles for a given workload.
    
    Args:
        row: DataFrame row containing problem and architecture parameters
        
    Returns:
        Theoretical minimum cycles
    """
    output_size = row["prob.P"] * row["prob.Q"]
    weight_size = row["prob.R"] * row["prob.S"]
    in_channel = row["prob.C"]
    out_channel = row["prob.K"]
    batch = row["prob.N"]
    total_macs = output_size * weight_size * in_channel * out_channel * batch

    hw_macs = row["arch.pe_dim"] ** 2  # TODO: update for new gemmini repr
    min_cycles = math.ceil(total_macs / hw_macs)
    return min_cycles


def get_matching_rows(df: pd.DataFrame, 
                     matching_keys: Iterable, 
                     matching_values: Iterable) -> pd.DataFrame:
    """
    Get DataFrame rows that match specific key-value pairs.
    
    Args:
        df: Input DataFrame
        matching_keys: Keys to match
        matching_values: Values to match
        
    Returns:
        Filtered DataFrame with matching rows
    """
    idxs_per_key = [df[matching_keys[i]] == matching_values[i] for i in range(len(matching_values))]
    idxs = idxs_per_key[0]
    for key_idxs in idxs_per_key:
        idxs = np.bitwise_and(idxs, key_idxs)
    comp_df = df.loc[idxs]
    return comp_df


def random_mapping_trajectories(dataset_path: pathlib.Path, 
                               output_dir: pathlib.Path, 
                               target_key: str, 
                               num_layers: int, 
                               mappings_per_layer: int) -> None:
    """
    Analyze random mapping trajectories for different layers.
    
    Args:
        dataset_path: Path to dataset
        output_dir: Output directory for plots
        target_key: Target metric to analyze
        num_layers: Number of layers to analyze
        mappings_per_layer: Number of mappings per layer
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    data_creator = DlaDatasetCreator(
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
    df = data_creator.train_data.df

    arch_keys = utils.keys_by_type(df, "arch", scalar_only=True)
    prob_keys = utils.keys_by_type(df, "prob", scalar_only=True)
    matching_keys = arch_keys + prob_keys
    first_idxs = df.groupby(by=matching_keys, sort=False)["index"].first()

    for i, row_idx in enumerate(first_idxs[:num_layers]):
        matching_values = df.loc[row_idx, matching_keys]
        comp_rows = get_matching_rows(df, matching_keys, matching_values)
        vals = comp_rows[target_key][:mappings_per_layer]
        
        prob_str = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])
        
        # Plot random trajectory
        plt.figure(figsize=(10, 6))
        plt.title(f'Random Mapping Trajectory - Layer {i}: {prob_str}')
        plt.xlabel('Mapping Index')
        plt.ylabel(target_key)
        plt.plot(range(len(vals)), vals, 'b-', alpha=0.7, linewidth=1)
        plt.scatter(range(len(vals)), vals, c='red', s=10, alpha=0.6)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"random_trajectory_layer_{i}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Layer {i}: {len(vals)} mappings, "
                   f"min={min(vals):.2e}, max={max(vals):.2e}, "
                   f"mean={np.mean(vals):.2e}")


def compare_search_strategies(dataset_path: pathlib.Path,
                             output_dir: pathlib.Path,
                             target_key: str,
                             num_layers: int,
                             mappings_per_layer: int) -> None:
    """
    Compare different search strategies (random vs best).
    
    Args:
        dataset_path: Path to dataset
        output_dir: Output directory for plots
        target_key: Target metric to analyze
        num_layers: Number of layers to analyze
        mappings_per_layer: Number of mappings per layer
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    data_creator = DlaDatasetCreator(
        dataset_path=dataset_path, 
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
    first_idxs = df.groupby(by=matching_keys, sort=False)["index"].first()

    for i, row_idx in enumerate(first_idxs[:num_layers]):
        matching_values = df.loc[row_idx, matching_keys]
        comp_rows = get_matching_rows(df, matching_keys, matching_values)
        vals = comp_rows[target_key][:mappings_per_layer]
        
        prob_str = "_".join([str(int(v)) for v in matching_values[-len(prob_keys):]])
        
        # Generate search curves
        random_curve = utils.search_curve(vals)
        best_curve = list(reversed(sorted(vals)))
        
        plt.figure(figsize=(12, 8))
        
        # Plot search curves
        plt.subplot(2, 2, 1)
        plt.title(f'Search Curves - {prob_str}')
        plt.plot(range(len(random_curve)), random_curve, label='Random Search', alpha=0.7)
        plt.plot(range(len(best_curve)), best_curve, label='Best Order', alpha=0.7)
        plt.xlabel('Search Steps')
        plt.ylabel(target_key)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot histogram
        plt.subplot(2, 2, 2)
        plt.title(f'Value Distribution - {prob_str}')
        plt.hist(vals, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel(target_key)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_vals = sorted(vals)
        cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        plt.plot(sorted_vals, cumulative)
        plt.title(f'Cumulative Distribution - {prob_str}')
        plt.xlabel(target_key)
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        
        # Plot search efficiency
        plt.subplot(2, 2, 4)
        efficiency = [random_curve[j] / best_curve[j] if best_curve[j] != 0 else 1 
                     for j in range(min(len(random_curve), len(best_curve)))]
        plt.plot(range(len(efficiency)), efficiency)
        plt.title(f'Search Efficiency - {prob_str}')
        plt.xlabel('Search Steps')
        plt.ylabel('Random/Best Ratio')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"search_comparison_layer_{i}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Layer {i} analysis complete: {prob_str}")


def analyze_mapping_diversity(dataset_path: pathlib.Path,
                             output_dir: pathlib.Path,
                             target_key: str) -> None:
    """
    Analyze diversity of mapping strategies across different problems.
    
    Args:
        dataset_path: Path to dataset
        output_dir: Output directory for results
        target_key: Target metric to analyze
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    data_creator = DlaDatasetCreator(
        dataset_path=dataset_path, 
        shuffle=False, 
        total_samples=0, 
        split_ratios={"train": 1}, 
        process_mappings="split"
    )
    df = data_creator.train_data.df
    
    # Get mapping features
    mapping_keys = utils.keys_by_type(df, "mapping", scalar_only=True)
    prob_keys = utils.keys_by_type(df, "prob", scalar_only=True)
    arch_keys = utils.keys_by_type(df, "arch", scalar_only=True)
    
    # Group by problem and architecture
    grouping_keys = arch_keys + prob_keys
    diversity_results = []
    
    for group_name, group_df in df.groupby(grouping_keys):
        if len(group_df) < 10:  # Skip groups with too few samples
            continue
            
        # Calculate mapping diversity metrics
        mapping_data = group_df[mapping_keys].values
        
        # Coefficient of variation for each mapping dimension
        cv_metrics = []
        for i in range(mapping_data.shape[1]):
            if mapping_data[:, i].std() > 0:
                cv = mapping_data[:, i].std() / mapping_data[:, i].mean()
                cv_metrics.append(cv)
        
        avg_cv = np.mean(cv_metrics) if cv_metrics else 0
        
        # Performance spread
        performance_vals = group_df[target_key].values
        perf_cv = performance_vals.std() / performance_vals.mean() if performance_vals.mean() > 0 else 0
        
        diversity_results.append({
            'group': group_name,
            'num_mappings': len(group_df),
            'mapping_diversity': avg_cv,
            'performance_diversity': perf_cv,
            'best_performance': performance_vals.min(),
            'worst_performance': performance_vals.max()
        })
    
    # Convert to DataFrame and save
    diversity_df = pd.DataFrame(diversity_results)
    diversity_df.to_csv(output_dir / "mapping_diversity_analysis.csv", index=False)
    
    # Plot diversity analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mapping diversity vs performance diversity
    axes[0, 0].scatter(diversity_df['mapping_diversity'], diversity_df['performance_diversity'])
    axes[0, 0].set_xlabel('Mapping Diversity (CV)')
    axes[0, 0].set_ylabel('Performance Diversity (CV)')
    axes[0, 0].set_title('Mapping vs Performance Diversity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Number of mappings vs diversity
    axes[0, 1].scatter(diversity_df['num_mappings'], diversity_df['mapping_diversity'])
    axes[0, 1].set_xlabel('Number of Mappings')
    axes[0, 1].set_ylabel('Mapping Diversity (CV)')
    axes[0, 1].set_title('Sample Size vs Mapping Diversity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance range analysis
    perf_range = diversity_df['worst_performance'] - diversity_df['best_performance']
    axes[1, 0].scatter(diversity_df['mapping_diversity'], perf_range)
    axes[1, 0].set_xlabel('Mapping Diversity (CV)')
    axes[1, 0].set_ylabel('Performance Range')
    axes[1, 0].set_title('Mapping Diversity vs Performance Range')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution of diversity metrics
    axes[1, 1].hist(diversity_df['mapping_diversity'], bins=30, alpha=0.7, label='Mapping Diversity')
    axes[1, 1].hist(diversity_df['performance_diversity'], bins=30, alpha=0.7, label='Performance Diversity')
    axes[1, 1].set_xlabel('Diversity (CV)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Diversity Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mapping_diversity_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Diversity analysis complete. Analyzed {len(diversity_results)} problem groups.")
    logger.info(f"Average mapping diversity: {diversity_df['mapping_diversity'].mean():.4f}")
    logger.info(f"Average performance diversity: {diversity_df['performance_diversity'].mean():.4f}") 