"""
Configuration settings for fingerprint preprocessing pipeline.
"""
import os
from pathlib import Path


class Config:
    """Configuration class for preprocessing parameters."""
    
    # External tool paths
    NFIQ_PATH = os.path.join(os.path.dirname(__file__), "external_tools", "nfiq")
    
    # Image processing parameters
    NFIQ_THRESHOLD = 3  # Filter images with NFIQ score >= this value (1=best, 5=worst)
    CROPPING_MARGIN = 75  # Margin for cropping in pixels
    TARGET_ASPECT_RATIO = 1.0  # 1:1 aspect ratio (set to 0.75 for 3:4)
    BRIGHTNESS_THRESHOLD = 225  # Threshold for determining if cropping is needed
    
    # File format settings
    SUPPORTED_IMAGE_FORMATS = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff']
    OUTPUT_FORMAT = 'PNG'  # Output format for processed images
    
    # Enhancement settings
    ENHANCEMENT_ENABLED = True  # Whether enhancement functionality is available


def get_default_config():
    """Get default configuration instance."""
    return Config()