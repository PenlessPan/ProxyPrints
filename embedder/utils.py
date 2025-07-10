"""
Utility functions for embedder training.
"""
import logging
import numpy as np


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""
    
    def __init__(self, patience=7, min_epochs=10, delta=0.001):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_epochs: Minimum epochs before early stopping activates
            delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = -1
        
    def __call__(self, val_score, epoch):
        """
        Call this function after each epoch with the validation score.
        
        Args:
            val_score: Current validation score (higher is better)
            epoch: Current epoch number
            
        Returns:
            Tuple of (should_stop, is_best)
        """
        # Don't consider early stopping until min_epochs is reached
        if epoch < self.min_epochs:
            if self.best_score is None or val_score > self.best_score:
                self.best_score = val_score
                self.best_epoch = epoch
                return False, True
            return False, False
        
        # Check if this is a new best score
        if self.best_score is None or val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            return False, True
        else:
            # Score didn't improve enough
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
            
            return self.early_stop, False


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_summary(model, config):
    """Save a summary of the model architecture."""
    import os
    
    summary_path = os.path.join(config.log_dir, 'model_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("Model Architecture Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Embedding Dimension: {model.embedding_dim}\n")
        f.write(f"Depth: {model.depth}\n")
        f.write(f"Filters Base: {model.filters_base}\n")
        f.write(f"Total Parameters: {count_parameters(model):,}\n\n")
        
        f.write("Layer Details:\n")
        f.write("-" * 20 + "\n")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters())
                f.write(f"{name}: {module} ({param_count:,} parameters)\n")
    
    return summary_path
