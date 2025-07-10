import re
import csv
import logging
import subprocess
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
from PIL import Image
import os
from .config import get_default_config, Config  # Import your config system

class FingerprintMatcher:
    """
    A utility class for fingerprint matching and minutiae extraction.
    
    This class provides methods to process fingerprint images, extract minutiae,
    and perform matching using NBIS tools (mindtct, bozorth3).
    """

    # Error handling modes
    ERROR_RAISE = "raise"
    ERROR_PRINT = "print"
    ERROR_IGNORE = "ignore"

    def __init__(self, 
             config: Optional[Config] = None,
             temp_dir: str = "./TempDir",
             error_handling: str = ERROR_PRINT,
             log_level: int = logging.INFO):
        """
        Initialize the FingerprintMatcher.
        
        Args:
            mindtct_path: Path to the mindtct executable
            bozorth3_path: Path to the bozorth3 executable
            temp_dir: Directory for temporary files
            error_handling: How to handle errors (raise, print, ignore)
            log_level: Logging level
        """
        # Set up logging
        self.logger = logging.getLogger("FingerprintMatcher")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize paths


        self.config = config if config is not None else get_default_config()
        self.mindtct_path = self.config.MINDTCT_PATH
        self.bozorth3_path = self.config.BOZORTH3_PATH
        self.temp_dir = temp_dir
        self.error_handling = error_handling
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Validate executables
        self._validate_executables()
    
    def _validate_executables(self) -> None:
        """Verify that the required executables exist and are executable."""
        for exe_path, name in [(self.mindtct_path, "mindtct"), (self.bozorth3_path, "bozorth3")]:
            if not os.path.exists(exe_path):
                self.logger.warning(f"{name} executable not found at: {exe_path}")
            elif not os.access(exe_path, os.X_OK):
                self.logger.warning(f"{name} executable is not executable: {exe_path}")
    
    def _handle_error(self, message: str, exception: Exception = None) -> None:
        """
        Handle errors according to the error_handling setting.
        
        Args:
            message: Error message
            exception: Exception that was raised (if any)
        """
        if self.error_handling == self.ERROR_RAISE:
            if exception:
                raise exception
            else:
                raise RuntimeError(message)
        elif self.error_handling == self.ERROR_PRINT:
            if exception:
                self.logger.error(f"{message}: {str(exception)}")
            else:
                self.logger.error(message)
        # If ERROR_IGNORE, do nothing
    
    def extract_id_and_imp(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the fingerprint ID and impression number from a filename.
        
        Args:
            filename: Filename to extract ID from (format: ID_impression.ext)
            
        Returns:
            Tuple of (ID, impression) or (None, None) if no match
        """
        # Extract basename without extension
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        # Match ID_impression pattern
        match = re.search(r'(\d+)_(\d+)', base_name)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def grayscale_image(self, input_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Convert a fingerprint image to 8-bit grayscale format suitable for processing.
        
        Args:
            input_path: Path to input image
            output_path: Path to save processed image (if None, a temp path is used)
            
        Returns:
            Path to the processed image or None if processing failed
        """
        if output_path is None:
            output_path = os.path.join(self.temp_dir, f"temp_{os.path.basename(input_path)}")
        
        try:
            with Image.open(input_path) as img:
                gray_img = img.convert('L')
                gray_img = gray_img.point(lambda x: x)
                gray_img.save(output_path, 'PNG')
            return output_path
        except Exception as e:
            self._handle_error(f"Error converting image to grayscale", e)
            return None
    
    def grayscale_dir(self, input_dir: str, output_dir: Optional[str] = None, 
                     show_progress: bool = True) -> Dict[str, str]:
        """
        Convert all images in a directory to 8-bit grayscale.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images (if None, uses temp_dir)
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary mapping original filenames to processed filenames
        """
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, "grayscale")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files in the directory
        image_files = [f for f in os.listdir(input_dir) 
                      if os.path.isfile(os.path.join(input_dir, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        result_map = {}
        
        # Use tqdm for progress tracking if requested
        iterator = tqdm(image_files) if show_progress else image_files
        
        for filename in iterator:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            
            processed_path = self.grayscale_image(input_path, output_path)
            if processed_path:
                result_map[filename] = processed_path
        
        return result_map
    
    def call_mindtct(self, image_path: str, output_base: Optional[str] = None) -> Optional[str]:
        """
        Extract minutiae from a fingerprint image using mindtct.
        
        Args:
            image_path: Path to preprocessed fingerprint image
            output_base: Base filename for output files (if None, a temp name is used)
            
        Returns:
            Base path for output files or None if extraction failed
        """
        if output_base is None:
            output_base = os.path.join(self.temp_dir, f"minutiae_{os.path.basename(image_path).split('.')[0]}")
        
        try:
            result = subprocess.run(
                [self.mindtct_path, '-m1', image_path, output_base],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return output_base
            else:
                self._handle_error(f"Mindtct failed with code {result.returncode}: {result.stderr}")
                return None
        except Exception as e:
            self._handle_error(f"Error calling mindtct for {image_path}", e)
            return None
    
    def extract_xyt(self, image_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Process an image and extract just the XYT file.
        
        Args:
            image_path: Path to fingerprint image
            output_dir: Directory to save the XYT file (if None, uses temp_dir)
            
        Returns:
            Path to the XYT file or None if processing failed
        """
        # Create output directory if needed
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, "xyt")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Process image to grayscale
        processed_path = self.grayscale_image(image_path)
        if not processed_path:
            return None
        
        # Run mindtct on processed image
        output_base = os.path.join(self.temp_dir, f"temp_minutiae_{base_name}")
        minutiae_base = self.call_mindtct(processed_path, output_base)
        if not minutiae_base:
            return None
        
        # Copy XYT file to output directory
        xyt_source = f"{minutiae_base}.xyt"
        xyt_dest = os.path.join(output_dir, f"{base_name}.xyt")
        
        try:
            shutil.copy2(xyt_source, xyt_dest)
            
            # Clean up all other minutiae files
            for ext in ['.brw', '.dm', '.hcm', '.lcm', '.lfm', '.min', '.qm']:
                if os.path.exists(f"{minutiae_base}{ext}"):
                    os.remove(f"{minutiae_base}{ext}")
            
            return xyt_dest
        except Exception as e:
            self._handle_error(f"Error copying XYT file", e)
            return None
    
    def extract_xyt_dir(self, input_dir: str, output_dir: Optional[str] = None,
                       show_progress: bool = True) -> Dict[str, str]:
        """
        Process all images in a directory and extract XYT files.
        
        Args:
            input_dir: Directory containing fingerprint images
            output_dir: Directory to save XYT files (if None, uses temp_dir/xyt)
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary mapping original filenames to XYT filenames
        """
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, "xyt")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files in the directory
        image_files = [f for f in os.listdir(input_dir) 
                      if os.path.isfile(os.path.join(input_dir, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        result_map = {}
        
        # Use tqdm for progress tracking if requested
        iterator = tqdm(image_files) if show_progress else image_files
        
        for filename in iterator:
            input_path = os.path.join(input_dir, filename)
            xyt_path = self.extract_xyt(input_path, output_dir)
            
            if xyt_path:
                result_map[filename] = xyt_path
        
        return result_map
    
    def extract_min(self, image_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Process an image and extract just the MIN file.
        
        Args:
            image_path: Path to fingerprint image
            output_dir: Directory to save the MIN file (if None, uses temp_dir)
            
        Returns:
            Path to the MIN file or None if processing failed
        """
        # Create output directory if needed
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, "min")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Process image to grayscale
        processed_path = self.grayscale_image(image_path)
        if not processed_path:
            return None
        
        # Run mindtct on processed image
        output_base = os.path.join(self.temp_dir, f"temp_minutiae_{base_name}")
        minutiae_base = self.call_mindtct(processed_path, output_base)
        if not minutiae_base:
            return None
        
        # Copy MIN file to output directory
        min_source = f"{minutiae_base}.min"
        min_dest = os.path.join(output_dir, f"{base_name}.min")
        
        try:
            shutil.copy2(min_source, min_dest)
            
            # Clean up all other minutiae files
            for ext in ['.brw', '.dm', '.hcm', '.lcm', '.lfm', '.xyt', '.qm']:
                if os.path.exists(f"{minutiae_base}{ext}"):
                    os.remove(f"{minutiae_base}{ext}")
            
            return min_dest
        except Exception as e:
            self._handle_error(f"Error copying MIN file", e)
            return None
    
    def extract_min_dir(self, input_dir: str, output_dir: Optional[str] = None,
                       show_progress: bool = True) -> Dict[str, str]:
        """
        Process all images in a directory and extract MIN files.
        
        Args:
            input_dir: Directory containing fingerprint images
            output_dir: Directory to save MIN files (if None, uses temp_dir/min)
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary mapping original filenames to MIN filenames
        """
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, "min")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files in the directory
        image_files = [f for f in os.listdir(input_dir) 
                      if os.path.isfile(os.path.join(input_dir, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        result_map = {}
        
        # Use tqdm for progress tracking if requested
        iterator = tqdm(image_files) if show_progress else image_files
        
        for filename in iterator:
            input_path = os.path.join(input_dir, filename)
            min_path = self.extract_min(input_path, output_dir)
            
            if min_path:
                result_map[filename] = min_path
        
        return result_map
    
    def count_minutiae(self, xyt_file: str) -> int:
        """
        Count the number of minutiae points in an XYT file.
        
        Args:
            xyt_file: Path to .xyt file
            
        Returns:
            Number of minutiae points
        """
        try:
            if not os.path.exists(xyt_file):
                return 0
            
            with open(xyt_file, 'r') as f:
                minutiae_count = sum(1 for line in f)
            return minutiae_count
        except Exception as e:
            self._handle_error(f"Error counting minutiae in {xyt_file}", e)
            return 0
    
    def call_bozorth(self, probe_xyt: str, gallery_xyt: str) -> Optional[float]:
        """
        Match two fingerprints using bozorth3.
        
        Args:
            probe_xyt: Path to probe fingerprint .xyt file
            gallery_xyt: Path to gallery fingerprint .xyt file
            
        Returns:
            Match score or None if matching failed
        """
        try:
            result = subprocess.run(
                [self.bozorth3_path, probe_xyt, gallery_xyt],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                self._handle_error(f"Bozorth3 failed with code {result.returncode}: {result.stderr}")
                return None
        except Exception as e:
            self._handle_error(f"Error running bozorth3", e)
            return None
    
    def match_pair(self, probe_image: str, gallery_image: str) -> Optional[float]:
        """
        Match two fingerprint images directly.
        
        Args:
            probe_image: Path to probe fingerprint image
            gallery_image: Path to gallery fingerprint image
            
        Returns:
            Match score or None if matching failed
        """
        # Extract XYT files for both images
        probe_xyt = self.extract_xyt(probe_image)
        if not probe_xyt:
            return None
        
        gallery_xyt = self.extract_xyt(gallery_image)
        if not gallery_xyt:
            return None
        
        # Match the XYT files
        return self.call_bozorth(probe_xyt, gallery_xyt)
    
    def match_pair_dir(self, probe_dir: str, gallery_dir: str, 
                      show_progress: bool = True) -> Dict[str, float]:
        """
        Match pairs of fingerprint images with the same name from two directories.
        
        Args:
            probe_dir: Directory containing probe fingerprint images
            gallery_dir: Directory containing gallery fingerprint images
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary mapping filenames to match scores
        """
        # Get common filenames (without extension)
        probe_files = {os.path.splitext(f)[0]: f for f in os.listdir(probe_dir) 
                     if os.path.isfile(os.path.join(probe_dir, f))}
        gallery_files = {os.path.splitext(f)[0]: f for f in os.listdir(gallery_dir) 
                       if os.path.isfile(os.path.join(gallery_dir, f))}
        
        common_names = set(probe_files.keys()).intersection(set(gallery_files.keys()))
        
        result_map = {}
        
        # Use tqdm for progress tracking if requested
        iterator = tqdm(common_names) if show_progress else common_names
        
        for name in iterator:
            probe_path = os.path.join(probe_dir, probe_files[name])
            gallery_path = os.path.join(gallery_dir, gallery_files[name])
            
            score = self.match_pair(probe_path, gallery_path)
            if score is not None:
                result_map[name] = score
        
        return result_map
    
    def match_all(self, probe_dir: str, gallery_dir: str, output_csv: str,
                 min_score: float = 0, show_progress: bool = True) -> None:
        """
        Match all fingerprint images between two directories and save results to CSV.
        
        Args:
            probe_dir: Directory containing probe fingerprint images
            gallery_dir: Directory containing gallery fingerprint images
            output_csv: Path to save CSV results
            min_score: Minimum score to include in CSV (0 for all scores)
            show_progress: Whether to show a progress bar
        """
        # Get all image files
        probe_files = [f for f in os.listdir(probe_dir) 
                     if os.path.isfile(os.path.join(probe_dir, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        gallery_files = [f for f in os.listdir(gallery_dir) 
                       if os.path.isfile(os.path.join(gallery_dir, f)) and 
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        # Extract XYT files for all probe images
        self.logger.info(f"Extracting XYT files for {len(probe_files)} probe images...")
        probe_xyt_map = self.extract_xyt_dir(probe_dir, show_progress=show_progress)
        
        # Extract XYT files for all gallery images
        self.logger.info(f"Extracting XYT files for {len(gallery_files)} gallery images...")
        gallery_xyt_map = self.extract_xyt_dir(gallery_dir, show_progress=show_progress)
        
        # Prepare CSV file
        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header row
            header = ['probe_file'] + [os.path.splitext(f)[0] for f in gallery_files]
            csv_writer.writerow(header)
            
            total_comparisons = len(probe_files) * len(gallery_files)
            self.logger.info(f"Performing {total_comparisons} comparisons...")
            
            # Set up progress tracking for the outer loop
            probe_iterator = tqdm(probe_files) if show_progress else probe_files
            
            # Perform matching
            for probe_file in probe_iterator:
                probe_name = os.path.splitext(probe_file)[0]
                probe_xyt = probe_xyt_map.get(probe_file)
                
                if not probe_xyt:
                    continue
                
                row = [probe_name]
                
                for gallery_file in gallery_files:
                    gallery_name = os.path.splitext(gallery_file)[0]
                    gallery_xyt = gallery_xyt_map.get(gallery_file)
                    
                    if not gallery_xyt:
                        row.append('')
                        continue
                    
                    score = self.call_bozorth(probe_xyt, gallery_xyt)
                    
                    if score is not None and score >= min_score:
                        row.append(score)
                    else:
                        row.append('')
                
                csv_writer.writerow(row)
        
        self.logger.info(f"Results saved to {output_csv}")
    
    def process_fingerprint(self, image_path: str) -> Tuple[Optional[str], Optional[str], int]:
        """
        Process a fingerprint image: preprocess, extract minutiae, and count them.
        
        Args:
            image_path: Path to fingerprint image
            
        Returns:
            Tuple of (processed image path, xyt file path, minutiae count)
        """
        # Process the image
        processed_path = self.grayscale_image(image_path)
        if not processed_path:
            return None, None, 0
        
        # Extract minutiae
        minutiae_base = self.call_mindtct(processed_path)
        if not minutiae_base:
            return processed_path, None, 0
        
        # Get XYT path
        xyt_path = f"{minutiae_base}.xyt"
        
        # Count minutiae
        minutiae_count = self.count_minutiae(xyt_path)
        
        return processed_path, xyt_path, minutiae_count
    
    def match_against_targets(self, 
                             probe_image: str, 
                             target_ids: List[str], 
                             gallery_dir: str) -> List[Tuple[str, str, float, int]]:
        """
        Match a probe fingerprint against specific target IDs in the gallery.
        
        Args:
            probe_image: Path to probe fingerprint image
            target_ids: List of target IDs to match against
            gallery_dir: Directory containing gallery fingerprint images
            
        Returns:
            List of tuples (ID, impression, score, minutiae_count) for best matches
        """
        self.logger.info(f"Testing {probe_image} against impressions from specified IDs...")
        
        # Process the probe image
        _, probe_xyt, probe_minutiae_count = self.process_fingerprint(probe_image)
        if not probe_xyt:
            self._handle_error("Failed to process probe image")
            return []
        
        self.logger.info(f"Probe image has {probe_minutiae_count} minutiae points")
        
        # Get all matches
        all_results = []
        
        for target_id in target_ids:
            self.logger.info(f"Matching against impressions of ID {target_id}")
            
            # Find all gallery images with this ID
            gallery_images = [f for f in os.listdir(gallery_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')) and 
                           f.split('_')[0] == target_id]
            
            if not gallery_images:
                self.logger.info(f"No impressions found for ID {target_id}")
                continue
            
            # Compare against each gallery impression
            id_results = []
            for gallery_img in sorted(gallery_images):
                gallery_path = os.path.join(gallery_dir, gallery_img)
                
                # Process gallery image
                _, gallery_xyt, gallery_minutiae_count = self.process_fingerprint(gallery_path)
                if not gallery_xyt:
                    continue
                
                # Match fingerprints
                score = self.call_bozorth(probe_xyt, gallery_xyt)
                if score is not None:
                    self.logger.info(f"Against {gallery_img}: Score = {score}, Minutiae count = {gallery_minutiae_count}")
                    id_results.append((gallery_img, score, gallery_minutiae_count))
            
            # Get best match for this ID
            if id_results:
                best_match = max(id_results, key=lambda x: x[1])
                all_results.append((target_id, best_match[0], best_match[1], best_match[2]))
        
        # Print summary
        if all_results:
            self.logger.info("\nSummary of Best Matches:")
            sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
            for id_, impression, score, minutiae_count in sorted_results:
                self.logger.info(f"ID {id_}, impression {impression}: Score = {score}, Minutiae count = {minutiae_count}")
        
        return all_results
    
    def match_against_gallery(self, 
                             probe_image: str, 
                             gallery_dir: str, 
                             threshold_score: float = 40,
                             show_progress: bool = True) -> List[Tuple[str, str, float, int]]:
        """
        Match a probe fingerprint against all images in a gallery directory.
        
        Args:
            probe_image: Path to probe fingerprint image
            gallery_dir: Directory containing gallery fingerprint images
            threshold_score: Minimum score to consider a match
            show_progress: Whether to show a progress bar
            
        Returns:
            List of tuples (ID, impression, score, minutiae_count) for matches above threshold
        """
        self.logger.info(f"Testing {probe_image} against all fingerprints in {gallery_dir}...")
        self.logger.info(f"Reporting matches with score higher than {threshold_score}")
        
        # Process the probe image
        _, probe_xyt, probe_minutiae_count = self.process_fingerprint(probe_image)
        if not probe_xyt:
            self._handle_error("Failed to process probe image")
            return []
        
        self.logger.info(f"Probe image has {probe_minutiae_count} minutiae points")
        
        # Get all images from the directory
        all_images = [f for f in os.listdir(gallery_dir) 
                    if os.path.isfile(os.path.join(gallery_dir, f)) and
                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        high_score_matches = []
        
        self.logger.info(f"Processing {len(all_images)} fingerprint images...")
        
        # Use tqdm for progress tracking if requested
        iterator = tqdm(all_images) if show_progress else all_images
        
        for gallery_img in iterator:
            gallery_path = os.path.join(gallery_dir, gallery_img)
            
            # Process gallery image
            _, gallery_xyt, gallery_minutiae_count = self.process_fingerprint(gallery_path)
            if not gallery_xyt:
                continue
            
            # Match fingerprints
            score = self.call_bozorth(probe_xyt, gallery_xyt)
            if score is not None and score > threshold_score:
                # Extract ID from filename
                img_id, _ = self.extract_id_and_imp(gallery_img)
                high_score_matches.append((img_id, gallery_img, score, gallery_minutiae_count))
        
        # Print summary
        if high_score_matches:
            self.logger.info(f"\nFound {len(high_score_matches)} matches above threshold {threshold_score}")
            
            # Sort by score (highest first)
            sorted_matches = sorted(high_score_matches, key=lambda x: x[2], reverse=True)
            
            for img_id, impression, score, minutiae_count in sorted_matches[:10]:  # Show top 10
                self.logger.info(f"ID {img_id}, file {impression}: Score = {score}, Minutiae count = {minutiae_count}")
        else:
            self.logger.info(f"No fingerprints found with match score higher than {threshold_score}")
        
        return high_score_matches
    
    def get_match_summary_by_id(self, matches: List[Tuple[str, str, float, int]]) -> Dict[str, Tuple[int, float]]:
        """
        Group match results by ID and find the best match for each ID.
        
        Args:
            matches: List of match results (ID, impression, score, minutiae_count)
            
        Returns:
            Dictionary mapping IDs to (match_count, best_score)
        """
        id_groups = {}
        for img_id, _, score, _ in matches:
            if img_id not in id_groups:
                id_groups[img_id] = []
            id_groups[img_id].append(score)
        
        return {img_id: (len(scores), max(scores)) for img_id, scores in id_groups.items()}
    
    def cleanup(self, force: bool = True) -> None:
        """
        Clean up temporary files.
        
        Args:
            force: If True, remove the entire temp directory. If False, only remove files older than 1 day.
        """
        if force:
            try:
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir, exist_ok=True)
                self.logger.info(f"Removed temporary directory: {self.temp_dir}")
            except Exception as e:
                self._handle_error(f"Error removing temporary directory", e)
        else:
            import time
            current_time = time.time()
            one_day_in_seconds = 86400
            
            try:
                count = 0
                for filename in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > one_day_in_seconds:
                            os.remove(file_path)
                            count += 1
                
                self.logger.info(f"Removed {count} old temporary files")
            except Exception as e:
                self._handle_error(f"Error cleaning up temporary files", e)