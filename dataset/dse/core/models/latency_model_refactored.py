"""
Refactored Latency Model for DOSA.

This module provides a clean, modular implementation of the latency prediction model
with separated analytical and ML-based prediction components.
"""

import pathlib
import traceback
import shutil
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch

from dataset import DATASET_ROOT_PATH
from dataset.common import logger, utils
from dataset.dse import pytorch_util, DlaDataset, predictors


class AnalyticalLatencyPredictor:
    """Handles analytical latency prediction using roofline models."""
    
    def __init__(self, with_roofline: bool = True):
        """
        Initialize analytical predictor.
        
        Args:
            with_roofline: Whether to use roofline model
        """
        self.with_roofline = with_roofline
        self.roofline_max = None
        
    def predict_analytical(self, 
                          hw_config: torch.Tensor, 
                          mapping: torch.Tensor, 
                          access_counts: torch.Tensor) -> torch.Tensor:
        """
        Predict latency using analytical model.
        
        Args:
            hw_config: Hardware configuration parameters
            mapping: Mapping parameters
            access_counts: Access count statistics
            
        Returns:
            Predicted latency values
        """
        # Simplified analytical model
        # In practice, this would implement detailed analytical formulas
        
        batch_size = hw_config.shape[0]
        
        # Basic computation: cycles = operations / throughput
        operations = access_counts[:, 0]  # MAC operations
        pe_utilization = torch.clamp(mapping[:, 0] * mapping[:, 1], 0.1, 1.0)  # Spatial utilization
        throughput = hw_config[:, 0] * pe_utilization  # Effective throughput
        
        cycles = operations / torch.clamp(throughput, min=1.0)
        
        return cycles.unsqueeze(-1)
        
    def generate_rooflines(self, 
                          hw_config: torch.Tensor, 
                          mapping: torch.Tensor, 
                          access_counts: torch.Tensor) -> torch.Tensor:
        """
        Generate roofline model predictions.
        
        Args:
            hw_config: Hardware configuration
            mapping: Mapping parameters
            access_counts: Access counts
            
        Returns:
            Roofline predictions
        """
        if not self.with_roofline:
            return torch.zeros(hw_config.shape[0], 5)
            
        # Compute different roofline metrics
        compute_roof = hw_config[:, 0]  # Peak compute
        memory_roof = access_counts[:, -1] * 0.1  # Memory bandwidth limit
        
        # Additional roofline components
        l1_roof = access_counts[:, 1] * 0.01
        l2_roof = access_counts[:, 2] * 0.05
        dram_roof = access_counts[:, 3] * 0.2
        
        rooflines = torch.stack([compute_roof, memory_roof, l1_roof, l2_roof, dram_roof], dim=1)
        
        return rooflines


class MLLatencyPredictor:
    """Machine learning-based latency predictor."""
    
    def __init__(self, 
                 output_dir: pathlib.Path,
                 input_size: int,
                 hidden_layers: Tuple[int, ...] = (256, 512, 2048, 2048, 512, 256, 64),
                 learning_rate: float = 1e-5,
                 dropout: float = 0.3):
        """
        Initialize ML predictor.
        
        Args:
            output_dir: Output directory for model files
            input_size: Input feature size
            hidden_layers: Hidden layer sizes
            learning_rate: Learning rate
            dropout: Dropout rate
        """
        self.output_dir = pathlib.Path(output_dir)
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        
        self.mlp = None
        self.optimizer = None
        self.loss_fn = torch.nn.L1Loss()
        
    def build_model(self) -> None:
        """Build the MLP model."""
        self.mlp = pytorch_util.build_mlp(
            input_size=self.input_size,
            output_size=1,
            n_layers=len(self.hidden_layers),
            size=self.hidden_layers,
            activation="gelu",
            dropout=self.dropout,
            output_activation="softplus",
        )
        self.mlp.to(pytorch_util.device)
        
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.learning_rate)
        
        logger.info(f"Built MLP model with {sum(p.numel() for p in self.mlp.parameters())} parameters")
        
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              num_epochs: int = 1000) -> None:
        """
        Train the ML model.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of training epochs
        """
        if self.mlp is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        self.mlp.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for features, targets in train_loader:
                predictions = self.mlp(features).squeeze()
                loss = self.loss_fn(predictions, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            if epoch % 100 == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
                
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the ML model.
        
        Args:
            features: Input features
            
        Returns:
            Predicted latency values
        """
        if self.mlp is None:
            raise ValueError("Model not built or trained.")
            
        self.mlp.eval()
        with torch.no_grad():
            predictions = self.mlp(features)
            
        return predictions
        
    def save_model(self, model_path: pathlib.Path) -> None:
        """Save the trained model."""
        if self.mlp is None:
            raise ValueError("No model to save.")
            
        # Create backup if model exists
        if model_path.exists():
            backup_path = model_path.parent / (model_path.name + ".bak")
            shutil.copy(model_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
            
        torch.save(self.mlp.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path: pathlib.Path) -> bool:
        """
        Load a trained model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if loaded successfully
        """
        try:
            if self.mlp is None:
                self.build_model()
                
            self.mlp.load_state_dict(torch.load(model_path, map_location=pytorch_util.device))
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class FeatureExtractor:
    """Extracts and prepares features for latency prediction."""
    
    def __init__(self, 
                 relevant_mapping_keys: List[str],
                 with_analytical: bool = True,
                 with_roofline: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            relevant_mapping_keys: List of relevant mapping feature keys
            with_analytical: Whether to include analytical features
            with_roofline: Whether to include roofline features
        """
        self.relevant_mapping_keys = relevant_mapping_keys
        self.with_analytical = with_analytical
        self.with_roofline = with_roofline
        
        self.internal_relevant_idxs = []
        self.internal_relevant_keys = []
        
    def extract_features(self, 
                        data: DlaDataset,
                        hw_config: torch.Tensor,
                        mapping: torch.Tensor,
                        prob: torch.Tensor,
                        access_counts: torch.Tensor,
                        rooflines: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract features for latency prediction.
        
        Args:
            data: Dataset containing statistics
            hw_config: Hardware configuration
            mapping: Mapping parameters
            prob: Problem parameters
            access_counts: Access count features
            rooflines: Optional roofline features
            
        Returns:
            Concatenated feature tensor
        """
        # Filter relevant mapping features
        self._update_relevant_indices(data)
        
        relevant_mapping = mapping[:, self.internal_relevant_idxs]
        
        # Start with basic features
        features = [relevant_mapping, prob, access_counts]
        
        # Add roofline features if enabled
        if self.with_roofline and rooflines is not None:
            features.append(rooflines)
            
        # Concatenate all features
        combined_features = torch.cat(features, dim=1)
        
        return combined_features
        
    def _update_relevant_indices(self, data: DlaDataset) -> None:
        """Update indices of relevant mapping features."""
        if self.internal_relevant_idxs:  # Already computed
            return
            
        self.internal_relevant_idxs = []
        self.internal_relevant_keys = []
        
        for idx, key in enumerate(self.relevant_mapping_keys):
            if data.creator.stats.get(key + "_std", 0) != 0:
                self.internal_relevant_idxs.append(idx)
                self.internal_relevant_keys.append(key)
                
        logger.info(f"Using {len(self.internal_relevant_idxs)} relevant mapping features")


class LatencyModelRefactored:
    """
    Refactored latency model with clean separation of analytical and ML components.
    
    This class combines analytical modeling with machine learning for accurate
    latency prediction across different hardware configurations and mappings.
    """
    
    def __init__(self, 
                 output_dir: pathlib.Path, 
                 relevant_mapping_keys: List[str]):
        """
        Initialize refactored latency model.
        
        Args:
            output_dir: Output directory for model files
            relevant_mapping_keys: List of relevant mapping feature keys
        """
        self.output_dir = pathlib.Path(output_dir)
        self.relevant_mapping_keys = relevant_mapping_keys
        self.target_key = "target.cycle"
        
        # Model configuration
        self.with_analytical = True
        self.with_roofline = True
        self.train_model = False
        self.with_cache = False
        
        # Initialize components
        self.analytical_predictor = AnalyticalLatencyPredictor(with_roofline=self.with_roofline)
        self.feature_extractor = FeatureExtractor(
            relevant_mapping_keys, 
            with_analytical=self.with_analytical,
            with_roofline=self.with_roofline
        )
        self.ml_predictor = None
        
    def train(self, 
              train_data: DlaDataset, 
              valid_data: Optional[DlaDataset] = None,
              train_model: bool = False,
              with_analytical: bool = True,
              with_roofline: bool = True,
              continue_training: bool = False,
              num_epochs: int = 1000,
              gpu_id: int = 0,
              with_cache: bool = False) -> None:
        """
        Train the latency model.
        
        Args:
            train_data: Training dataset
            valid_data: Optional validation dataset
            train_model: Whether to train ML model
            with_analytical: Whether to use analytical features
            with_roofline: Whether to use roofline model
            continue_training: Whether to continue from existing model
            num_epochs: Number of training epochs
            gpu_id: GPU device ID
            with_cache: Whether to use caching
        """
        logger.info("Starting latency model training...")
        
        # Update configuration
        self.train_model = train_model
        self.with_analytical = with_analytical
        self.with_roofline = with_roofline and with_analytical
        self.with_cache = with_cache
        
        # Update components
        self.analytical_predictor.with_roofline = self.with_roofline
        self.feature_extractor.with_analytical = self.with_analytical
        self.feature_extractor.with_roofline = self.with_roofline
        
        if not train_model:
            logger.info("Training disabled, using analytical model only")
            return
            
        # Prepare training data
        arch_keys = utils.keys_by_type(train_data.df, "arch")
        prob_keys = utils.keys_by_type(train_data.df, "prob", scalar_only=True)
        access_keys = utils.keys_by_type(train_data.df, "dse.access")
        mapping_keys = self.relevant_mapping_keys
        
        # Extract tensors
        arch_train = pytorch_util.from_numpy(train_data.df[arch_keys].to_numpy())
        mapping_train = pytorch_util.from_numpy(train_data.df[mapping_keys].to_numpy())
        prob_train = pytorch_util.from_numpy(train_data.df[prob_keys].to_numpy())
        access_train = pytorch_util.from_numpy(train_data.df[access_keys].to_numpy())
        y_train = pytorch_util.from_numpy(train_data.df[self.target_key].to_numpy())
        
        # Generate rooflines
        rooflines = self.analytical_predictor.generate_rooflines(
            arch_train, mapping_train, access_train
        )
        self.analytical_predictor.roofline_max = rooflines.max()
        
        # Extract features
        features = self.feature_extractor.extract_features(
            train_data, arch_train, mapping_train, prob_train, access_train, rooflines
        )
        
        # Initialize ML predictor
        input_size = features.shape[1]
        self.ml_predictor = MLLatencyPredictor(
            output_dir=self.output_dir,
            input_size=input_size
        )
        
        # Try to load existing model
        model_path = self._get_model_path()
        if continue_training or not self.ml_predictor.load_model(model_path):
            logger.info("Training new ML model...")
            
            # Build and train model
            self.ml_predictor.build_model()
            
            # Create data loader
            train_dataset = pytorch_util.X_y_dataset(features, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100000)
            
            # Train model
            self.ml_predictor.train(train_loader, num_epochs)
            
            # Save trained model
            self.ml_predictor.save_model(model_path)
            
        logger.info("Latency model training complete")
        
    def _get_model_path(self) -> pathlib.Path:
        """Get model file path based on configuration."""
        if self.with_analytical:
            pred_type = "both"
        else:
            pred_type = "dnn"
            
        # Use artifact path if available
        artifact_path = DATASET_ROOT_PATH / "dse" / "trained_models" / "artifact" / pred_type
        if artifact_path.exists():
            return artifact_path / "mlp_latency_0.pt"
        else:
            return self.output_dir / f"mlp_{pred_type}_latency_0.pt"
            
    def predict(self, 
                hw_config: torch.Tensor, 
                mapping: torch.Tensor, 
                access_counts: torch.Tensor, 
                probs: torch.Tensor) -> torch.Tensor:
        """
        Predict latency using the trained model.
        
        Args:
            hw_config: Hardware configuration
            mapping: Mapping parameters
            access_counts: Access count statistics
            probs: Problem parameters
            
        Returns:
            Predicted latency values
        """
        if not self.train_model:
            # Use analytical model only
            return self.analytical_predictor.predict_analytical(hw_config, mapping, access_counts)
            
        if self.ml_predictor is None:
            raise ValueError("ML model not trained. Call train() first.")
            
        # Generate features
        rooflines = None
        if self.with_roofline:
            rooflines = self.analytical_predictor.generate_rooflines(
                hw_config, mapping, access_counts
            )
            
        # Create dummy dataset for feature extraction
        dummy_data = type('DummyData', (), {
            'creator': type('DummyCreator', (), {
                'stats': {key + "_std": 1.0 for key in self.relevant_mapping_keys}
            })()
        })()
        
        features = self.feature_extractor.extract_features(
            dummy_data, hw_config, mapping, probs, access_counts, rooflines
        )
        
        # Make prediction
        predictions = self.ml_predictor.predict(features)
        
        return predictions
        
    def test(self, 
             test_data: DlaDataset, 
             num_worst_points: int = 10,
             gpu_id: int = 0) -> Dict[str, float]:
        """
        Test the latency model on test data.
        
        Args:
            test_data: Test dataset
            num_worst_points: Number of worst predictions to analyze
            gpu_id: GPU device ID
            
        Returns:
            Dictionary with test metrics
        """
        logger.info("Testing latency model...")
        
        # Prepare test data
        arch_keys = utils.keys_by_type(test_data.df, "arch")
        prob_keys = utils.keys_by_type(test_data.df, "prob", scalar_only=True)
        access_keys = utils.keys_by_type(test_data.df, "dse.access")
        
        arch_test = pytorch_util.from_numpy(test_data.df[arch_keys].to_numpy())
        mapping_test = pytorch_util.from_numpy(test_data.df[self.relevant_mapping_keys].to_numpy())
        prob_test = pytorch_util.from_numpy(test_data.df[prob_keys].to_numpy())
        access_test = pytorch_util.from_numpy(test_data.df[access_keys].to_numpy())
        y_true = test_data.df[self.target_key].to_numpy()
        
        # Make predictions
        with torch.no_grad():
            y_pred = self.predict(arch_test, mapping_test, access_test, prob_test)
            y_pred = pytorch_util.to_numpy(y_pred.squeeze())
        
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
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
        if self.ml_predictor and self.ml_predictor.mlp:
            for param in self.ml_predictor.mlp.parameters():
                param.requires_grad = False
                
    def unfreeze(self) -> None:
        """Unfreeze model parameters."""
        if self.ml_predictor and self.ml_predictor.mlp:
            for param in self.ml_predictor.mlp.parameters():
                param.requires_grad = True 