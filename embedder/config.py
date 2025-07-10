"""
Configuration settings for embedder training.
"""
import os
from pathlib import Path


class Config:
    """Configuration class for embedder training parameters."""
    
    def __init__(self, output_dir=None):
        # Dataset parameters
        self.data_path = "data/fingerprints"
        self.img_size = 256
        self.batch_size = 128
        self.num_workers = 4
        
        # Model parameters
        self.embedding_dim = 512
        self.conv_layers_depth = 9
        self.filters_base = 16
        
        # Training parameters
        self.learning_rate = 0.00000521043136294799
        self.weight_decay = 0.00001
        self.epochs = 200
        self.margin = 10.140036474483678
        
        # Augmentation parameters
        self.rotation_degrees = 7
        self.crop_percent = 11
        
        # Validation parameters
        self.val_id_percent = 20
        self.early_stopping_patience = 10
        self.min_epochs_before_stopping = 15
        
        # Output directories
        if output_dir is None:
            output_dir = "models/embedder"
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir
        self.log_dir = os.path.join(output_dir, 'logs')
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def update_from_dict(self, config_dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Recreate directories after update
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


def get_default_config():
    """Get default configuration instance."""
    return Config()
