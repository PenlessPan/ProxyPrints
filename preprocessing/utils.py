"""
Utility functions for fingerprint preprocessing.
"""
import os
import subprocess
from PIL import Image, ImageFile
from collections import Counter
from typing import Optional, Dict, List
import logging

from .config import get_default_config

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def compute_nfiq_score(image_file: str, config=None) -> Optional[int]:
    """
    Compute NFIQ quality score for a fingerprint image.
    
    Args:
        image_file: Path to the fingerprint image
        config: Configuration object (optional)
    
    Returns:
        NFIQ score (1-5, where 1 is highest quality) or None if error
    """
    if config is None:
        config = get_default_config()
    
    try:
        cmd = f'"{config.NFIQ_PATH}" "{image_file}"'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()
        
        stdout, stderr = process.communicate()
        
        if stderr:
            logger.debug(f"NFIQ stderr: {stderr.decode('utf-8')}")
        
        if stdout:
            output = stdout.decode('utf-8').strip()
            return int(output[0])
        
        return None
        
    except Exception as e:
        logger.error(f"Error computing NFIQ score for {image_file}: {e}")
        return None


def convert_image_to_png(image_file: str) -> Optional[str]:
    """
    Convert image to 8-bit grayscale PNG format.
    
    Args:
        image_file: Path to input image
    
    Returns:
        Path to converted PNG file or None if error
    """
    try:
        with Image.open(image_file) as image:
            # Convert to 8-bit grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Create output filename
            file_name, _ = os.path.splitext(image_file)
            output_file = f"{file_name}_temp.png"
            
            # Save as PNG
            image.save(output_file, 'PNG')
            
        return output_file
        
    except Exception as e:
        logger.error(f"Error converting {image_file} to PNG: {e}")
        return None


def compute_nfiq_scores_batch(target_dir: str, config=None) -> Dict[int, int]:
    """
    Compute NFIQ scores for all images in a directory.
    
    Args:
        target_dir: Directory containing fingerprint images
        config: Configuration object (optional)
    
    Returns:
        Counter dictionary with NFIQ scores and their frequencies
    """
    if config is None:
        config = get_default_config()
    
    counter = Counter()
    processed = 0
    
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in config.SUPPORTED_IMAGE_FORMATS):
                processed += 1
                
                if processed % 100 == 0:
                    logger.info(f"Processed {processed} images. Current distribution: {dict(counter)}")
                
                image_path = os.path.join(root, file)
                
                # Convert to PNG for NFIQ processing
                converted_image = convert_image_to_png(image_path)
                if converted_image is None:
                    continue
                
                # Compute NFIQ score
                nfiq_score = compute_nfiq_score(converted_image, config)
                if nfiq_score is not None:
                    counter[nfiq_score] += 1
                
                # Clean up temporary PNG
                if os.path.exists(converted_image):
                    os.remove(converted_image)
    
    logger.info(f"Final NFIQ distribution: {dict(counter)}")
    return counter


def get_file_name_and_ext(image_file: str) -> tuple:
    """
    Extract filename components.
    
    Args:
        image_file: Path to image file
    
    Returns:
        Tuple of (full_filename, name_without_ext, extension)
    """
    full_file_name = os.path.basename(image_file)
    file_name, file_ext = os.path.splitext(full_file_name)
    return full_file_name, file_name, file_ext[1:] if file_ext else ""


def list_image_files(folder: str, with_path: bool = True, config=None) -> List[str]:
    """
    List all image files in a folder (supports nested folders).
    
    Args:
        folder: Target folder path
        with_path: Whether to return full paths or just filenames
        config: Configuration object (optional)
    
    Returns:
        List of image file paths or filenames
    """
    if config is None:
        config = get_default_config()
    
    image_files = []
    
    # Check if folder contains subfolders or files directly
    items = [os.path.join(folder, item) for item in os.listdir(folder)]
    
    if items and os.path.isdir(items[0]):
        # Folder contains subfolders
        for subfolder in items:
            if os.path.isdir(subfolder):
                for file in os.listdir(subfolder):
                    if any(file.lower().endswith(ext) for ext in config.SUPPORTED_IMAGE_FORMATS):
                        path = os.path.join(subfolder, file) if with_path else file
                        image_files.append(path)
    else:
        # Folder contains files directly
        for file in os.listdir(folder):
            if any(file.lower().endswith(ext) for ext in config.SUPPORTED_IMAGE_FORMATS):
                path = os.path.join(folder, file) if with_path else file
                image_files.append(path)
    
    return image_files


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )