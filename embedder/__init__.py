"""
Embedder Module

This module provides fingerprint embedding functionality including:
- EmbeddingNet: Neural network for extracting fingerprint embeddings
- TripletLoss: Loss function for training embeddings
- Training utilities and dataset handling
- Configuration management

Example usage:
    from embedder import EmbeddingNet, Config, train_model
    
    # Create and train embedder
    config = Config()
    model = EmbeddingNet(embedding_dim=512)
    train_model(config)
    
    # Load trained model
    model = EmbeddingNet.load_from_checkpoint("path/to/checkpoint.pth")
"""

# Import main model classes
from .model import EmbeddingNet, TripletLoss

# Import configuration
from .config import Config, get_default_config

# Import training functionality
from .train import train_model, train_epoch, validate_embeddings, visualize_embeddings

# Import dataset handling
from .dataset import FingerprintDataset, create_data_loaders, split_ids_for_validation

# Import utilities
from .utils import (
    EarlyStopping, 
    set_random_seed, 
    count_parameters, 
    save_model_summary
)

# Version information
__version__ = "1.0.0"
__author__ = "Yaniv Hacmon, Keren Gorelik, Yisroel Mirsky"

# Export main classes and functions
__all__ = [
    # Model classes
    'EmbeddingNet',
    'TripletLoss',
    
    # Configuration
    'Config',
    'get_default_config',
    
    # Training
    'train_model',
    'train_epoch', 
    'validate_embeddings',
    'visualize_embeddings',
    
    # Dataset
    'FingerprintDataset',
    'create_data_loaders',
    'split_ids_for_validation',
    
    # Utilities
    'EarlyStopping',
    'set_random_seed',
    'count_parameters',
    'save_model_summary'
]

# Module-level convenience functions
def create_model(embedding_dim=512, depth=9, filters_base=16, device=None):
    """
    Create a new EmbeddingNet model with specified parameters.
    
    Args:
        embedding_dim: Dimension of output embeddings
        depth: Number of convolutional layers
        filters_base: Base number of filters in first layer
        device: Device to place model on (auto-detected if None)
    
    Returns:
        EmbeddingNet: Initialized model
    """
    import torch
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EmbeddingNet(
        embedding_dim=embedding_dim,
        depth=depth, 
        filters_base=filters_base
    ).to(device)
    
    return model


def load_model(checkpoint_path, embed_config=None, device=None):
    """
    Load a trained EmbeddingNet model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        embed_config: Optional config dict for model architecture
        device: Device to load model on (auto-detected if None)
    
    Returns:
        EmbeddingNet: Loaded model
    """
    return EmbeddingNet.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        embed_config=embed_config,
        device=device
    )


def get_model_info():
    """Print information about the embedder module and its components."""
    print("Embedder Module Information")
    print("=" * 40)
    print(f"Version: {__version__}")
    print(f"Authors: {__author__}")
    print()
    print("Main Components:")
    print("- EmbeddingNet: Neural network for fingerprint embedding extraction")
    print("- TripletLoss: Loss function for training with triplet learning")
    print("- Config: Configuration management for training parameters")
    print("- Training utilities: Functions for model training and validation")
    print("- Dataset handling: Data loading and preprocessing for fingerprint images")
    print()
    print("Quick Start:")
    print("  from embedder import EmbeddingNet, Config")
    print("  config = Config()")
    print("  model = EmbeddingNet(embedding_dim=512)")
    print()
    print("For detailed usage, see individual module documentation.")
