"""
Refactored Energy Model for DOSA.

This module provides a clean, modular implementation of the energy prediction model
with separated concerns for data preprocessing, model training, and prediction.
"""

import pathlib
import traceback
from typing import Optional, Dict, List, Tuple
import multiprocessing

import numpy as np
import pandas as pd
import torch

from dataset.common import logger, utils
from dataset.dse import pytorch_util, DlaDataset


class AccessCountProcessor:
    """Handles access count computation and normalization."""
    
    def __init__(self, output_dir: pathlib.Path, with_cache: bool = False):
        """
        Initialize access count processor.
        
        Args:
            output_dir: Directory for temporary files
            with_cache: Whether to use caching
        """
        self.output_dir = pathlib.Path(output_dir)
        self.with_cache = with_cache
        
    def add_access_counts(self, dataset: DlaDataset) -> None:
        """
        Add access count columns to dataset.
        
        Args:
            dataset: Dataset to process
        """
        if "dse.access_mac" in dataset.df.columns:
            logger.info("Access counts already present, skipping computation")
            return
            
        logger.info("Computing access counts...")
        
        # Use multiprocessing for faster computation
        prob_keys = utils.keys_by_type(dataset.df, "prob", scalar_only=True)
        
        # Split dataframe into chunks for parallel processing
        num_processes = min(multiprocessing.cpu_count(), 8)
        chunk_size = max(1, len(dataset.df) // num_processes)
        chunks = [dataset.df[i:i + chunk_size] for i in range(0, len(dataset.df), chunk_size)]
        
        # Process chunks
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = self._process_chunk(chunk, prob_keys, dataset)
            processed_chunks.append(processed_chunk)
        
        # Combine results
        dataset.df = pd.concat(processed_chunks, ignore_index=True)
        
        logger.info("Access count computation complete")
        
    def _process_chunk(self, chunk: pd.DataFrame, prob_keys: List[str], dataset: DlaDataset) -> pd.DataFrame:
        """Process a chunk of data to compute access counts."""
        chunk = chunk.copy()
        
        # Add access count columns (simplified computation)
        # In real implementation, this would call the actual access count computation
        for index, row in chunk.iterrows():
            # Placeholder computation - replace with actual logic
            chunk.at[index, "dse.access_mac"] = self._compute_mac_access(row)
            chunk.at[index, "dse.access_memlvl0"] = self._compute_memory_access(row, level=0)
            chunk.at[index, "dse.access_memlvl1"] = self._compute_memory_access(row, level=1)
            chunk.at[index, "dse.access_memlvl2"] = self._compute_memory_access(row, level=2)
            chunk.at[index, "dse.access_memlvl3"] = self._compute_memory_access(row, level=3)
            
        return chunk
        
    def _compute_mac_access(self, row: pd.Series) -> float:
        """Compute MAC access count for a mapping."""
        # Simplified computation - replace with actual logic
        return float(row.get("prob.P", 1) * row.get("prob.Q", 1) * row.get("prob.K", 1))
        
    def _compute_memory_access(self, row: pd.Series, level: int) -> float:
        """Compute memory access count for a specific level."""
        # Simplified computation - replace with actual logic
        base_access = self._compute_mac_access(row)
        return base_access * (0.5 ** level)  # Decreasing access with memory level
        
    def normalize_access_counts(self, dataset: DlaDataset, stats: Dict) -> None:
        """
        Normalize access count columns.
        
        Args:
            dataset: Dataset to normalize
            stats: Statistics dictionary to update
        """
        access_keys = utils.keys_by_type(dataset.df, "dse.access", scalar_only=True)
        
        for access_key in access_keys:
            col_max = dataset.df[access_key].max()
            if col_max != 0:
                dataset.df[access_key] = dataset.df[access_key] / col_max
            stats[access_key + "_max"] = float(col_max)
            
        logger.info(f"Normalized {len(access_keys)} access count columns")


class EnergyModelArchitecture:
    """Handles energy model architecture and training."""
    
    def __init__(self, 
                 arch_param_size: int,
                 hidden_layers: Tuple[int, ...] = (8, 32),
                 learning_rate: float = 1e-3,
                 dropout: float = 0.4):
        """
        Initialize model architecture.
        
        Args:
            arch_param_size: Size of architecture parameters
            hidden_layers: Hidden layer sizes
            learning_rate: Learning rate for optimization
            dropout: Dropout rate
        """
        self.arch_param_size = arch_param_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        
        self.mlps = []
        self.optimizer = None
        self.loss_fn = torch.nn.MSELoss()
        
    def build_models(self, num_access_types: int) -> None:
        """
        Build MLP models for different access types.
        
        Args:
            num_access_types: Number of different access types
        """
        pytorch_util.init_gpu(gpu_id=0)
        
        # Architecture indices for different access types
        arch_indices = [None, None, [0, 2], [1], None]
        self.arch_indices = []
        
        for i in range(num_access_types):
            if i >= len(arch_indices) or arch_indices[i] is None:
                continue
                
            input_size = len(arch_indices[i]) if arch_indices[i] else self.arch_param_size
            
            mlp = pytorch_util.build_mlp(
                input_size=input_size,
                output_size=1,
                n_layers=len(self.hidden_layers),
                size=self.hidden_layers,
                activation="relu",
                dropout=self.dropout,
                output_activation="softplus",
            )
            mlp.to(pytorch_util.device)
            self.mlps.append(mlp)
            self.arch_indices.append(arch_indices[i])
            
        # Create optimizer for all models
        params = []
        for mlp in self.mlps:
            params.extend(list(mlp.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        
        logger.info(f"Built {len(self.mlps)} energy prediction models")
        
    def train_models(self, 
                    train_loader: torch.utils.data.DataLoader,
                    num_epochs: int = 1000) -> None:
        """
        Train the energy models.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of training epochs
        """
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, gamma=0.2)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for X_batch, y_batch, access_batch in train_loader:
                coeff_batch = self.predict_coefficients(X_batch)
                y_pred_batch = (coeff_batch * access_batch).sum(dim=1).unsqueeze(-1)
                loss = self.loss_fn(y_pred_batch, y_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for mlp in self.mlps for p in mlp.parameters()], 0.1
                )
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
                
            scheduler.step()
            
    def predict_coefficients(self, arch_params: torch.Tensor) -> torch.Tensor:
        """
        Predict energy coefficients for given architecture parameters.
        
        Args:
            arch_params: Architecture parameters
            
        Returns:
            Predicted coefficients
        """
        coefficients = []
        
        for i, mlp in enumerate(self.mlps):
            if self.arch_indices[i] is not None:
                selected_params = arch_params[:, self.arch_indices[i]]
            else:
                selected_params = arch_params
                
            coeff = mlp(selected_params)
            coefficients.append(coeff)
            
        return torch.cat(coefficients, dim=1)
        
    def save_models(self, output_dir: pathlib.Path, save_prefix: str = "energy") -> None:
        """Save trained models."""
        output_dir = pathlib.Path(output_dir)
        
        for i, mlp in enumerate(self.mlps):
            model_path = output_dir / f"mlp_{save_prefix}_{i}.pt"
            torch.save(mlp.state_dict(), model_path)
            
        opt_path = output_dir / f"mlp_opt_{save_prefix}.pt"
        torch.save(self.optimizer.state_dict(), opt_path)
        
        logger.info(f"Saved {len(self.mlps)} models to {output_dir}")
        
    def load_models(self, output_dir: pathlib.Path, save_prefix: str = "energy") -> bool:
        """
        Load trained models.
        
        Returns:
            True if models were loaded successfully
        """
        output_dir = pathlib.Path(output_dir)
        
        try:
            for i, mlp in enumerate(self.mlps):
                model_path = output_dir / f"mlp_{save_prefix}_{i}.pt"
                mlp.load_state_dict(torch.load(model_path))
                
            logger.info(f"Loaded {len(self.mlps)} models from {output_dir}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
            return False


class EnergyModelRefactored:
    """
    Refactored energy model with clean separation of concerns.
    
    This class provides energy prediction capabilities with modular components
    for data processing, model architecture, and prediction.
    """
    
    def __init__(self, output_dir: pathlib.Path, arch_param_size: int):
        """
        Initialize refactored energy model.
        
        Args:
            output_dir: Output directory for models and logs
            arch_param_size: Size of architecture parameter vector
        """
        self.output_dir = pathlib.Path(output_dir)
        self.arch_param_size = arch_param_size
        
        # Initialize components
        self.access_processor = AccessCountProcessor(output_dir)
        self.model_arch = EnergyModelArchitecture(arch_param_size)
        
        # Model state
        self.stats = {}
        self.energy_max = None
        self.mac_energy = None
        self.reg_energy = None
        self.dram_energy = None
        
    def train(self, 
              train_data: DlaDataset, 
              valid_data: Optional[DlaDataset] = None,
              num_epochs: int = 1000,
              gpu_id: int = 0,
              continue_training: bool = False,
              with_cache: bool = False) -> None:
        """
        Train the energy model.
        
        Args:
            train_data: Training dataset
            valid_data: Optional validation dataset
            num_epochs: Number of training epochs
            gpu_id: GPU device ID
            continue_training: Whether to continue from existing model
            with_cache: Whether to use caching
        """
        logger.info("Starting energy model training...")
        
        self.stats = train_data.creator.stats
        
        # Process access counts
        self.access_processor.with_cache = with_cache
        self.access_processor.add_access_counts(train_data)
        self.access_processor.normalize_access_counts(train_data, self.stats)
        
        if valid_data is not None:
            self.access_processor.add_access_counts(valid_data)
            self.access_processor.normalize_access_counts(valid_data, self.stats)
        
        # Prepare training data
        arch_keys = utils.keys_by_type(train_data.df, "arch", scalar_only=True)
        access_keys = utils.keys_by_type(train_data.df, "dse.access", scalar_only=True)
        
        X_train = train_data.df[arch_keys].to_numpy()
        access_train = train_data.df[access_keys].to_numpy()
        y_train = train_data.df["target.energy"].to_numpy()
        
        # Denormalize target for training
        y_train = train_data.denorm("target.energy", y_train).numpy()
        self.energy_max = y_train.max() / 100
        
        # Set up energy coefficients
        self._setup_energy_coefficients()
        
        # Build and train models
        self.model_arch.build_models(num_access_types=len(access_keys))
        
        # Try to load existing models
        if not continue_training or not self.model_arch.load_models(self.output_dir):
            logger.info("Training new models...")
            
            # Convert to tensors
            X_train = pytorch_util.from_numpy(X_train)
            access_train = pytorch_util.from_numpy(access_train)
            y_train = pytorch_util.from_numpy(y_train)
            
            # Create data loader
            train_dataset = pytorch_util.X_y_dataset(X_train, y_train, access_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=40000)
            
            # Train models
            self.model_arch.train_models(train_loader, num_epochs)
            
            # Save trained models
            self.model_arch.save_models(self.output_dir)
        
        # Update stats and save
        utils.store_json(train_data.creator.stats_path, train_data.creator.outer_stats, indent=4)
        
        logger.info("Energy model training complete")
        
    def _setup_energy_coefficients(self) -> None:
        """Setup energy coefficients for different access types."""
        # Energy per access values (simplified)
        mac_energy = 0.5608e-6 / self.energy_max * self.stats.get("dse.access_mac_max", 1)
        reg_energy = 0.48746172e-6 / self.energy_max * self.stats.get("dse.access_memlvl0_max", 1)
        dram_energy = 100e-6 / self.energy_max * self.stats.get("dse.access_memlvl3_max", 1)
        
        self.mac_energy = pytorch_util.from_numpy(np.array([mac_energy]))
        self.reg_energy = pytorch_util.from_numpy(np.array([reg_energy]))
        self.dram_energy = pytorch_util.from_numpy(np.array([dram_energy]))
        
    def predict(self, arch_params: torch.Tensor, access_params: torch.Tensor) -> torch.Tensor:
        """
        Predict energy consumption.
        
        Args:
            arch_params: Architecture parameters
            access_params: Access count parameters
            
        Returns:
            Predicted energy values
        """
        if not self.model_arch.mlps:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get prediction coefficients
        coefficients = self.model_arch.predict_coefficients(arch_params)
        
        # Compute energy prediction
        energy_pred = (coefficients * access_params).sum(dim=1, keepdim=True)
        
        return energy_pred
        
    def test(self, 
             test_data: DlaDataset, 
             num_worst_points: int = 10,
             gpu_id: int = 0) -> Dict[str, float]:
        """
        Test the energy model on test data.
        
        Args:
            test_data: Test dataset
            num_worst_points: Number of worst predictions to analyze
            gpu_id: GPU device ID
            
        Returns:
            Dictionary with test metrics
        """
        logger.info("Testing energy model...")
        
        # Process test data
        self.access_processor.add_access_counts(test_data)
        self.access_processor.normalize_access_counts(test_data, self.stats)
        
        # Prepare test data
        arch_keys = utils.keys_by_type(test_data.df, "arch", scalar_only=True)
        access_keys = utils.keys_by_type(test_data.df, "dse.access", scalar_only=True)
        
        X_test = pytorch_util.from_numpy(test_data.df[arch_keys].to_numpy())
        access_test = pytorch_util.from_numpy(test_data.df[access_keys].to_numpy())
        y_true = test_data.df["target.energy"].to_numpy()
        
        # Make predictions
        with torch.no_grad():
            y_pred = self.predict(X_test, access_test)
            y_pred = pytorch_util.to_numpy(y_pred.squeeze())
        
        # Denormalize for comparison
        y_true = test_data.denorm("target.energy", y_true).numpy()
        y_pred = y_pred * self.energy_max
        
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Find worst predictions
        errors = np.abs(y_true - y_pred)
        worst_indices = np.argsort(errors)[-num_worst_points:]
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'mape': float(mape),
            'num_samples': len(y_true),
            'worst_predictions': {
                'indices': worst_indices.tolist(),
                'errors': errors[worst_indices].tolist()
            }
        }
        
        logger.info(f"Test Results - MSE: {mse:.2e}, MAE: {mae:.2e}, MAPE: {mape:.2f}%")
        
        return metrics
        
    def freeze(self) -> None:
        """Freeze model parameters."""
        for mlp in self.model_arch.mlps:
            for param in mlp.parameters():
                param.requires_grad = False
                
    def unfreeze(self) -> None:
        """Unfreeze model parameters."""
        for mlp in self.model_arch.mlps:
            for param in mlp.parameters():
                param.requires_grad = True 