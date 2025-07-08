"""
Fingerprint Preprocessing Pipeline

A preprocessing pipeline for fingerprint images including:
- Quality filtering using NFIQ scores
- Image cropping and centering 
- Fingerprint enhancement
- Format standardization

Example usage:
    from preprocessing import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline()
    pipeline.process_folder(
        input_dir="raw_fingerprints/",
        output_dir="processed_fingerprints/"
    )
"""

from .config import Config, get_default_config
from .utils import setup_logging, compute_nfiq_score
from .fingerprint_processor import (
    preprocess_fingerprint_image,
    process_folder,
    crop_and_center_image,
    enhance_fingerprint
)

import os
import logging
from typing import Optional

__version__ = "1.0.0"
__author__ = "Yaniv Hacmon, Keren Gorelik, Yisroel Mirsky"

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Main preprocessing pipeline class for fingerprint image processing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config: Configuration object (optional, uses default if None)
        """
        self.config = config if config is not None else get_default_config()
        logger.info("Preprocessing pipeline initialized")
    
    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        filter_quality: bool = True,
        enhance: bool = False,
        crop_center: bool = True
    ) -> dict:
        """
        Process all fingerprint images in a folder.
        
        Args:
            input_dir: Directory containing raw fingerprint images
            output_dir: Output directory for processed images
            filter_quality: Whether to apply NFIQ quality filtering
            enhance: Whether to apply fingerprint enhancement
            crop_center: Whether to apply cropping and centering
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Starting preprocessing: {input_dir} -> {output_dir}")
        
        processed_count, total_images = process_folder(
            input_dir, output_dir, 
            filter_nfiq_score=filter_quality,
            enhance=enhance,
            crop_center=crop_center,
            config=self.config
        )
        
        stats = {
            'processed': processed_count,
            'total': total_images,
            'success_rate': processed_count / total_images if total_images > 0 else 0
        }
        
        logger.info(f"Processing complete: {processed_count}/{total_images} images processed successfully")
        return stats
    
    def process_single_image(
        self,
        input_path: str,
        output_path: str,
        filter_quality: bool = True,
        enhance: bool = False,
        crop_center: bool = True
    ) -> bool:
        """
        Process a single fingerprint image.
        
        Args:
            input_path: Path to input image
            output_path: Path for output image
            filter_quality: Whether to apply NFIQ quality filtering
            enhance: Whether to apply fingerprint enhancement
            crop_center: Whether to apply cropping and centering
        
        Returns:
            True if processing succeeded, False otherwise
        """
        return preprocess_fingerprint_image(
            input_path, output_path,
            filter_nfiq_score=filter_quality,
            crop_center=crop_center,
            enhance=enhance,
            config=self.config
        )


# Convenience function for quick access
def create_pipeline(config: Optional[Config] = None) -> PreprocessingPipeline:
    """Create a preprocessing pipeline instance."""
    return PreprocessingPipeline(config)


# Export main classes and functions
__all__ = [
    'PreprocessingPipeline',
    'Config',
    'get_default_config',
    'create_pipeline',
    'preprocess_fingerprint_image',
    'process_folder',
    'crop_and_center_image',
    'enhance_fingerprint',
    'setup_logging'
]