"""
Mathematical utilities for DOSA with improved organization.
"""

import math
import random
import string
import time
from typing import List, Tuple, Union

import numpy as np

from ..logger import logger


class MathUtils:
    """Mathematical operations and calculations."""
    
    @staticmethod
    def get_prime_factors(value: int) -> List[int]:
        """
        Get prime factors of a number from small to large.
        
        Args:
            value: Number to factorize
            
        Returns:
            List of prime factors
        """
        if value <= 0:
            logger.warning(f"Invalid value for prime factorization: {value}")
            return [1]
        
        factors = []
        
        # Handle factor 2
        while value % 2 == 0:
            factors.append(2)
            value = value // 2
        
        # Handle odd factors
        for i in range(3, int(math.sqrt(value)) + 1, 2):
            while value % i == 0:
                factors.append(i)
                value = value // i
        
        # If value is a prime number greater than 2
        if value > 2:
            factors.append(value)
        
        if len(factors) == 0:
            factors.append(1)
        
        logger.debug(f"Prime factors of {value}: {factors}")
        return factors
    
    @staticmethod
    def get_divisors(value: int) -> List[int]:
        """
        Get all divisors of a number in ascending order.
        
        Args:
            value: Number to find divisors for
            
        Returns:
            List of divisors
        """
        if value <= 0:
            logger.warning(f"Invalid value for divisors: {value}")
            return [1]
        
        divisors = []
        for i in range(1, value + 1):
            if value % i == 0:
                divisors.append(i)
        
        logger.debug(f"Divisors of {value}: {divisors}")
        return divisors
    
    @staticmethod
    def get_nearest_choice(value: float, sorted_choices: List[int]) -> int:
        """
        Find the closest choice to a given value.
        
        Args:
            value: Target value
            sorted_choices: List of choices in ascending order
            
        Returns:
            Closest choice to the value
        """
        if not sorted_choices:
            logger.warning("Empty choices list provided")
            return 1
        
        # Find the closest choice
        for idx, choice in enumerate(sorted_choices):
            if value <= choice:
                if idx == 0:
                    return choice
                
                # Compare distances to previous and current choice
                prev_choice = sorted_choices[idx - 1]
                if (choice - value) > (value - prev_choice):
                    return prev_choice
                return choice
        
        # If value is larger than all choices, return the largest
        return sorted_choices[-1]
    
    @staticmethod
    def round_down_choices(value: float, sorted_choices: List[int]) -> int:
        """
        Find the largest choice that is not greater than the value.
        
        Args:
            value: Target value
            sorted_choices: List of choices in ascending order
            
        Returns:
            Largest choice <= value
        """
        if not sorted_choices:
            logger.warning("Empty choices list provided")
            return 1
        
        for idx, choice in enumerate(sorted_choices):
            if value < choice:
                if idx == 0:
                    return choice  # Return first choice if value is smaller than all
                return sorted_choices[idx - 1]
        
        # If value is >= all choices, return the largest
        return sorted_choices[-1]
    
    @staticmethod
    def get_whole_ceil(n: float, near: float) -> int:
        """
        Get the smallest integer that divides n and is >= near.
        
        Args:
            n: Number to divide
            near: Target value
            
        Returns:
            Ceiling divisor
        """
        try:
            divisor_count = int(np.ceil(n / near))
            candidates = np.divide(n, np.linspace(1, divisor_count, divisor_count))
            whole_candidates = candidates[candidates % 1 == 0]
            
            if len(whole_candidates) > 0:
                return int(whole_candidates[-1])
            else:
                return int(near)
                
        except Exception as e:
            logger.error(f"Error in get_whole_ceil({n}, {near}): {e}")
            return int(near)
    
    @staticmethod
    def get_whole_floor(n: float, near: float) -> int:
        """
        Get the largest integer that divides n and is <= near.
        
        Args:
            n: Number to divide
            near: Target value
            
        Returns:
            Floor divisor
        """
        try:
            start_divisor = int(np.floor(n / near))
            end_divisor = int(n)
            divisor_range = end_divisor - start_divisor + 1
            
            candidates = np.divide(n, np.linspace(start_divisor, end_divisor, divisor_range))
            whole_candidates = candidates[candidates % 1 == 0]
            
            if len(whole_candidates) > 0:
                return int(whole_candidates[0])
            else:
                return int(near)
                
        except Exception as e:
            logger.error(f"Error in get_whole_floor({n}, {near}): {e}")
            return int(near)
    
    @staticmethod
    def update_prime_factors(prob_factors: List[List[int]], prob_idx: int, factor: int) -> None:
        """
        Update prime factors list by removing a specific factor.
        
        Args:
            prob_factors: List of prime factor lists (modified in place)
            prob_idx: Index of the problem dimension
            factor: Factor to remove
        """
        logger.debug(f"Before update: prob_factors[{prob_idx}] = {prob_factors[prob_idx]}")
        
        if prob_idx >= len(prob_factors):
            logger.error(f"Invalid prob_idx {prob_idx}, max is {len(prob_factors) - 1}")
            return
        
        orig_factor = factor
        rm_indices = []
        
        # Find indices of factors that divide the given factor
        for i, val in enumerate(prob_factors[prob_idx]):
            if factor % val == 0:
                rm_indices.append(i)
                factor = factor // val
                if factor == 1:
                    break
        
        # Remove factors in reverse order to maintain indices
        removed_vals = []
        for i in reversed(rm_indices):
            removed_vals.append(prob_factors[prob_idx].pop(i))
        
        # If no factors left, add 1
        if len(prob_factors[prob_idx]) == 0:
            prob_factors[prob_idx] = [1]
        
        # Verify the removal was correct
        removed_product = 1
        for val in removed_vals:
            removed_product *= val
        
        if removed_product != orig_factor:
            logger.error(f"Factor removal verification failed: {removed_product} != {orig_factor}")
        
        logger.debug(f"After update: prob_factors[{prob_idx}] = {prob_factors[prob_idx]}")
    
    @staticmethod
    def shrink_factor_space(prob_factors: List[List[int]]) -> None:
        """
        Remove trivial factor spaces (those containing only [1]).
        
        Args:
            prob_factors: List of prime factor lists (modified in place)
        """
        for prob_idx in range(len(prob_factors)):
            if len(prob_factors[prob_idx]) == 1 and prob_factors[prob_idx][0] == 1:
                prob_factors[prob_idx] = []
        
        logger.debug("Shrunk factor space")
    
    @staticmethod
    def get_correlation(a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> float:
        """
        Calculate Pearson correlation coefficient between two arrays.
        
        Args:
            a: First array
            b: Second array
            
        Returns:
            Correlation coefficient
        """
        try:
            a = np.array(a)
            b = np.array(b)
            
            if len(a) != len(b):
                logger.error(f"Array length mismatch: {len(a)} vs {len(b)}")
                return 0.0
            
            if len(a) < 2:
                logger.warning("Need at least 2 data points for correlation")
                return 0.0
            
            correlation_matrix = np.corrcoef(a, b)
            correlation = correlation_matrix[0, 1]
            
            # Handle NaN case
            if np.isnan(correlation):
                logger.warning("Correlation calculation returned NaN")
                return 0.0
            
            return float(correlation)
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    @staticmethod
    def search_curve(values: List[float]) -> List[float]:
        """
        Generate a search curve showing best value found so far at each point.
        
        Args:
            values: List of values
            
        Returns:
            List of best values found so far
        """
        if not values:
            return []
        
        best_per_point = []
        best_so_far = float("inf")
        
        for val in values:
            if val < best_so_far:
                best_so_far = val
            best_per_point.append(best_so_far)
        
        return best_per_point


class RandomUtils:
    """Random number generation and utilities."""
    
    @staticmethod
    def unique_filename(extension: str = "", prefix: str = "") -> str:
        """
        Generate a unique filename with timestamp and random characters.
        
        Args:
            extension: File extension (without dot)
            prefix: Optional prefix
            
        Returns:
            Unique filename
        """
        timeline = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())
        randname = ''.join(random.choice(string.ascii_uppercase + string.digits) 
                          for _ in range(16))
        
        filename = f"{timeline}-{randname}"
        
        if extension:
            filename = f"{filename}.{extension}"
        
        if prefix:
            filename = f"{prefix}-{filename}"
        
        return filename
    
    @staticmethod
    def get_current_seed() -> int:
        """
        Get a reasonable default random seed.
        
        Returns:
            Default random seed value
        """
        # Return a constant seed for testing and consistency
        return 42
    
    @staticmethod
    def set_global_seed(seed: int) -> None:
        """
        Set random seed for reproducibility across multiple libraries.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            logger.info(f"Set global random seed to {seed} (including PyTorch)")
        except ImportError:
            logger.info(f"Set global random seed to {seed} (PyTorch not available)")
    
    @staticmethod
    def generate_random_string(length: int = 16, 
                              charset: str = string.ascii_uppercase + string.digits) -> str:
        """
        Generate random string of specified length.
        
        Args:
            length: Length of random string
            charset: Character set to choose from
            
        Returns:
            Random string
        """
        return ''.join(random.choice(charset) for _ in range(length))


class GeometryUtils:
    """Geometric and spatial calculations."""
    
    @staticmethod
    def get_perm_arr_from_val(val: int, perm_arr_shape: Tuple[int, ...], 
                             prob_levels: int) -> List[int]:
        """
        Convert a flat index to permutation array.
        
        Args:
            val: Flat index value
            perm_arr_shape: Shape of permutation array
            prob_levels: Number of problem levels
            
        Returns:
            Permutation array
        """
        try:
            order_idx = np.unravel_index(val, perm_arr_shape)
            perm_idx = list(range(prob_levels))
            perm_arr = [-1] * len(perm_arr_shape)
            
            for i, idx in enumerate(order_idx):
                if idx < len(perm_idx):
                    perm_arr[i] = perm_idx[idx]
                    del perm_idx[idx]
                else:
                    logger.warning(f"Invalid index {idx} in permutation conversion")
                    perm_arr[i] = 0
            
            return perm_arr
            
        except Exception as e:
            logger.error(f"Error in get_perm_arr_from_val: {e}")
            return list(range(len(perm_arr_shape)))
    
    @staticmethod
    def get_val_from_perm_arr(perm_arr: List[int], perm_arr_shape: Tuple[int, ...], 
                             prob_levels: int) -> int:
        """
        Convert permutation array to flat index.
        
        Args:
            perm_arr: Permutation array
            perm_arr_shape: Shape of permutation array
            prob_levels: Number of problem levels
            
        Returns:
            Flat index value
        """
        try:
            if len(perm_arr) != len(perm_arr_shape):
                logger.error(f"Permutation array length mismatch: {len(perm_arr)} vs {len(perm_arr_shape)}")
                return 0
            
            order_idx = [0] * len(perm_arr)
            perm_idx = list(range(prob_levels))
            
            for i, idx in enumerate(perm_arr):
                for j, val in enumerate(perm_idx):
                    if val == idx:
                        order_idx[i] = j
                        del perm_idx[j]
                        break
            
            val = np.ravel_multi_index(tuple(order_idx), perm_arr_shape)
            return int(val)
            
        except Exception as e:
            logger.error(f"Error in get_val_from_perm_arr: {e}")
            return 0 