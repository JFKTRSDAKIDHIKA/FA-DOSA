"""
MLP predictor for performance analysis.

This module provides neural network-based prediction capabilities
for mapping performance evaluation.
"""

import pathlib
from typing import List, Tuple
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from dataset.common import utils, logger
from dataset.dse import pytorch_util, DlaDatasetCreator


class MLPPredictor:
    """Neural network predictor for performance metrics."""
    
    def __init__(self, input_size: int, hidden_layer_sizes: Tuple[int, ...] = (128, 256, 256, 32)):
        """
        Initialize MLP predictor.
        
        Args:
            input_size: Number of input features
            hidden_layer_sizes: Tuple of hidden layer sizes
        """
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.mlp = None
        self.optimizer = None
        self.loss_fn = torch.nn.MSELoss()
        
    def build_model(self, gpu_id: int = 0) -> None:
        """
        Build the MLP model.
        
        Args:
            gpu_id: GPU device ID to use
        """
        pytorch_util.init_gpu(gpu_id=gpu_id)
        
        self.mlp = pytorch_util.build_mlp(
            input_size=self.input_size,
            output_size=1,
            n_layers=len(self.hidden_layer_sizes),
            size=self.hidden_layer_sizes,
            activation="relu"
        )
        self.mlp.to(pytorch_util.device)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-3)
        
    def train(self, 
              X_train: torch.Tensor, 
              y_train: torch.Tensor,
              epochs: int = 200,
              batch_size: int = 20000) -> None:
        """
        Train the MLP model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.mlp is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        train_dataset = pytorch_util.X_y_dataset(X_train, y_train)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        
        for epoch in range(epochs):
            for X_batch, y_batch in train_data_loader:
                y_pred_batch = self.mlp(X_batch).squeeze()
                loss = self.loss_fn(y_pred_batch, y_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, loss: {loss.item():.6f}")
                
    def predict(self, X_test: torch.Tensor) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions as numpy array
        """
        if self.mlp is None:
            raise ValueError("Model not built or trained.")
            
        self.mlp.eval()
        with torch.no_grad():
            test_dataset = pytorch_util.X_y_dataset(X_test, torch.zeros(len(X_test)))
            test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20000)
            
            y_pred = np.array([])
            for X_batch, _ in test_data_loader:
                y_batch_pred = self.mlp(X_batch).squeeze()
                y_pred = np.concatenate((y_pred, pytorch_util.to_numpy(y_batch_pred)))
                
        return y_pred
        
    def save_model(self, model_path: pathlib.Path) -> None:
        """Save the trained model."""
        if self.mlp is None:
            raise ValueError("No model to save.")
        torch.save(self.mlp.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path: pathlib.Path) -> None:
        """Load a trained model."""
        if self.mlp is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.mlp.load_state_dict(torch.load(model_path))
        logger.info(f"Model loaded from {model_path}")


def run_mlp_experiment(train_data, test_data, x_key_types: List[str], y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a complete MLP prediction experiment.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset  
        x_key_types: List of input feature types
        y_key: Target variable key
        
    Returns:
        Tuple of (true_values, predictions)
    """
    train_df = train_data.df
    
    # Get feature keys
    x_keys = []
    for key_type in x_key_types:
        type_keys = utils.keys_by_type(train_df, key_type, scalar_only=True)
        x_keys.extend(type_keys)
        
    # Prepare training data
    X_train = train_df[x_keys]
    y_train = train_df[y_key]
    
    # Prepare test data
    test_df = test_data.df
    X_test = test_df[x_keys]
    y_test = test_df[y_key]
    
    # Initialize predictor
    predictor = MLPPredictor(input_size=len(x_keys))
    predictor.build_model(gpu_id=0)
    
    # Convert to tensors
    X_train_tensor = pytorch_util.from_numpy(X_train.to_numpy())
    y_train_tensor = pytorch_util.from_numpy(y_train.to_numpy())
    X_test_tensor = pytorch_util.from_numpy(X_test.to_numpy())
    
    # Train model
    predictor.train(X_train_tensor, y_train_tensor)
    
    # Make predictions
    y_pred = predictor.predict(X_test_tensor)
    y_true = pytorch_util.to_numpy(pytorch_util.from_numpy(y_test.to_numpy()))
    
    return y_true, y_pred


def evaluate_prediction_performance(dataset_path: pathlib.Path, 
                                   output_dir: pathlib.Path, 
                                   target_key: str) -> None:
    """
    Evaluate prediction performance with and without architecture features.
    
    Args:
        dataset_path: Path to dataset
        output_dir: Output directory for results
        target_key: Target metric to predict
    """
    import matplotlib.pyplot as plt
    
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create dataset
    data_creator = DlaDatasetCreator(
        dataset_path=dataset_path, 
        total_samples=100, 
        split_ratios={"train": 0.8, "test": 0.2}, 
        process_mappings="split"
    )
    
    train_data = data_creator.train_data
    test_data = data_creator.test_data
    
    # Test without architecture features
    logger.info("Testing prediction without architecture features...")
    y_test, y_pred = run_mlp_experiment(train_data, test_data, ("prob", "mapping"), target_key)
    
    mse_without_arch = mean_squared_error(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.title(f"{target_key} prediction without arch")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.savefig(output_dir / "pred_without_arch.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test with architecture features
    logger.info("Testing prediction with architecture features...")
    y_test, y_pred = run_mlp_experiment(train_data, test_data, ("arch", "prob", "mapping"), target_key)
    
    mse_with_arch = mean_squared_error(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.title(f"{target_key} prediction with arch")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.savefig(output_dir / "pred_with_arch.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Report results
    improvement = (1 - mse_with_arch / mse_without_arch) * 100
    logger.info(f"MSE without arch: {mse_without_arch:.6f}")
    logger.info(f"MSE with arch: {mse_with_arch:.6f}")
    logger.info(f"MSE improvement: {improvement:.1f}%")
    
    # Save results
    results = {
        'mse_without_arch': float(mse_without_arch),
        'mse_with_arch': float(mse_with_arch),
        'improvement_percent': float(improvement)
    }
    utils.store_json(output_dir / "prediction_results.json", results, indent=4) 