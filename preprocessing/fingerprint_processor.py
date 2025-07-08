"""
Main fingerprint preprocessing functions.
Handles quality filtering, cropping, centering, and enhancement.
"""
import os
import cv2
import numpy as np
import logging
from tqdm import tqdm
from typing import Optional, Tuple

try:
    import fingerprint_enhancer
    ENHANCER_AVAILABLE = True
except ImportError:
    ENHANCER_AVAILABLE = False
    logging.warning("fingerprint_enhancer not available. Enhancement will be skipped.")

from .utils import compute_nfiq_score, convert_image_to_png
from .config import get_default_config

logger = logging.getLogger(__name__)


def crop_and_center_image(img: np.ndarray, config=None) -> Optional[np.ndarray]:
    """
    Crop and center a fingerprint image based on contour detection.
    
    Args:
        img: Input image as numpy array
        config: Configuration object (optional)
    
    Returns:
        Cropped and centered image or None if processing failed
    """
    if config is None:
        config = get_default_config()
    
    try:
        h_img, w_img = img.shape[:2]
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Binarize the image
        _, thresh = cv2.threshold(gray, 192, 255, cv2.THRESH_BINARY_INV)
        
        # Erode to remove noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Calculate bounding rectangles
        rect_areas = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            rect_areas.append(w * h)
        
        if not rect_areas:
            logger.warning("No contours found in image")
            return None
        
        # Determine average area threshold
        try:
            avg_area = sorted(rect_areas, reverse=True)[5] if len(rect_areas) > 5 else rect_areas[-1]
        except IndexError:
            avg_area = rect_areas[-1]
        
        # Filter small contours
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 0.5 * avg_area:
                thresh[y:y + h, x:x + w] = 0
        
        # Get overall bounding box
        bx, by, bw, bh = cv2.boundingRect(thresh)
        
        # Set margins
        x_margin = config.CROPPING_MARGIN
        y_margin = int(config.CROPPING_MARGIN * 1.5) if config.TARGET_ASPECT_RATIO == 0.75 else config.CROPPING_MARGIN
        
        # Pad image if necessary
        if bx - x_margin < 0:
            pad_left = x_margin - bx
            gray = np.pad(gray, ((0, 0), (pad_left, 0)), mode='constant', constant_values=255)
            bx = x_margin
        
        if bx + bw + x_margin > gray.shape[1]:
            pad_right = bx + bw + x_margin - gray.shape[1]
            gray = np.pad(gray, ((0, 0), (0, pad_right)), mode='constant', constant_values=255)
        
        if by - y_margin < 0:
            pad_top = y_margin - by
            gray = np.pad(gray, ((pad_top, 0), (0, 0)), mode='constant', constant_values=255)
            by = y_margin
        
        if by + bh + y_margin > gray.shape[0]:
            pad_bottom = by + bh + y_margin - gray.shape[0]
            gray = np.pad(gray, ((0, pad_bottom), (0, 0)), mode='constant', constant_values=255)
        
        # Crop the image
        cropped = gray[by - y_margin:by + bh + y_margin, bx - x_margin:bx + bw + x_margin]
        
        return cropped
        
    except Exception as e:
        logger.error(f"Error in crop_and_center_image: {e}")
        return None


def fit_aspect_ratio(img: np.ndarray, target_ratio: float = 1.0) -> np.ndarray:
    """
    Fit image to target aspect ratio by padding with white pixels.
    
    Args:
        img: Input image
        target_ratio: Target width/height ratio (1.0 = square, 0.75 = 3:4)
    
    Returns:
        Image with target aspect ratio
    """
    h, w = img.shape[:2]
    current_ratio = w / h
    
    if abs(current_ratio - target_ratio) < 0.01:  # Already close to target
        return img
    
    if current_ratio < target_ratio:
        # Need to increase width
        new_w = int(h * target_ratio)
        pad = (new_w - w) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        # Need to increase height
        new_h = int(w / target_ratio)
        pad = (new_h - h) // 2
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    return img


def enhance_fingerprint(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Enhance fingerprint image using fingerprint_enhancer library.
    
    Args:
        img: Input grayscale fingerprint image
    
    Returns:
        Enhanced image or original if enhancement fails/unavailable
    """
    if not ENHANCER_AVAILABLE:
        logger.warning("Fingerprint enhancer not available, returning original image")
        return img
    
    try:
        enhanced = fingerprint_enhancer.enhance_Fingerprint(img)
        return np.invert(enhanced)  # Invert to match expected format
    except Exception as e:
        logger.error(f"Error enhancing fingerprint: {e}")
        return img


def preprocess_fingerprint_image(
    img_path: str, 
    output_path: str, 
    filter_nfiq_score: bool = True,
    crop_center: bool = True,
    enhance: bool = False,
    config=None
) -> bool:
    """
    Main preprocessing function for a single fingerprint image.
    
    Args:
        img_path: Path to input image
        output_path: Path for output image
        filter_nfiq_score: Whether to filter by NFIQ quality score
        crop_center: Whether to apply cropping and centering
        enhance: Whether to apply fingerprint enhancement
        config: Configuration object (optional)
    
    Returns:
        True if processing succeeded, False otherwise
    """
    if config is None:
        config = get_default_config()
    
    try:
        # Convert to PNG for processing
        png_path = convert_image_to_png(img_path)
        if png_path is None:
            return False
        
        # Read the image
        img = cv2.imread(png_path)
        if img is None:
            logger.error(f"Could not read image: {png_path}")
            if os.path.exists(png_path):
                os.remove(png_path)
            return False
        
        # Quality filtering
        if filter_nfiq_score:
            nfiq_score = compute_nfiq_score(png_path, config)
            if nfiq_score is None or nfiq_score >= config.NFIQ_THRESHOLD:
                logger.info(f"Filtering out low quality image (NFIQ: {nfiq_score}): {img_path}")
                if os.path.exists(png_path):
                    os.remove(png_path)
                return False
        
        # Clean up temporary PNG
        if os.path.exists(png_path):
            os.remove(png_path)
        
        # Crop and center if needed
        if crop_center and img.mean() > config.BRIGHTNESS_THRESHOLD:
            img = crop_and_center_image(img, config)
            if img is None:
                logger.warning(f"Cropping failed for: {img_path}")
                return False
        
        # Fit to aspect ratio
        img = fit_aspect_ratio(img, config.TARGET_ASPECT_RATIO)
        
        # Enhancement (optional)
        if enhance:
            img = enhance_fingerprint(img)
        
        # Save processed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, img)
        
        if not success:
            logger.error(f"Failed to save image: {output_path}")
            return False
        
        logger.debug(f"Successfully processed: {img_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return False


def process_folder(
    input_dir: str, 
    output_dir: str, 
    filter_nfiq_score: bool = True,
    crop_center: bool = True,
    enhance: bool = False,
    config=None
) -> Tuple[int, int]:
    """
    Process all fingerprint images in a folder.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        filter_nfiq_score: Whether to filter by NFIQ quality score
        crop_center: Whether to apply cropping and centering
        enhance: Whether to apply fingerprint enhancement
        config: Configuration object (optional)
    
    Returns:
        Tuple of (successful_count, total_count)
    """
    if config is None:
        config = get_default_config()
    
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    total = 0
    
    files = [f for f in os.listdir(input_dir) 
             if any(f.lower().endswith(ext) for ext in config.SUPPORTED_IMAGE_FORMATS)]
    
    for img_name in tqdm(files, desc="Processing fingerprints"):
        total += 1
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.png')
        
        if preprocess_fingerprint_image(img_path, output_path, filter_nfiq_score, crop_center, enhance, config):
            successful += 1
    
    logger.info(f"Processing complete: {successful}/{total} images processed successfully")
    return successful, total