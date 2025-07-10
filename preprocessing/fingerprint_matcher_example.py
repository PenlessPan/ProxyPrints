"""
Example usage of FingerprintMatcher integrated with the preprocessing pipeline.

This example shows how to use the FingerprintMatcher class with your existing
config system for fingerprint matching and analysis.
"""

import sys
from pathlib import Path

# Add preprocessing directory to path
sys.path.append(str(Path(__file__).parent))

from preprocessing import Config, get_default_config
from preprocessing.fingerprint_matcher import FingerprintMatcher


def example_basic_matching():
    """Example: Basic fingerprint matching using your config system."""
    print("=== Basic Fingerprint Matching ===")
    
    # Create matcher using your config system (no hardcoded paths!)
    matcher = FingerprintMatcher()
    
    # Match two fingerprint images
    probe_image = "path/to/probe_fingerprint.png"
    gallery_image = "path/to/gallery_fingerprint.png"
    
    score = matcher.match_pair(probe_image, gallery_image)
    
    if score is not None:
        print(f"Match score: {score}")
        if score > 40:  # Common threshold
            print("Strong match detected!")
        else:
            print("No strong match")
    else:
        print("Matching failed")
    
    # Clean up
    matcher.cleanup()


def example_custom_config():
    """Example: Using custom configuration."""
    print("\\n=== Custom Configuration ===")
    
    # Create custom config if needed
    config = Config()
    # Modify config settings if needed
    # config.MINDTCT_PATH = "/custom/path/to/mindtct"
    
    # Create matcher with custom config
    matcher = FingerprintMatcher(
        config=config,
        temp_dir="./custom_temp",
        error_handling=FingerprintMatcher.ERROR_PRINT
    )
    
    print("Matcher created with custom configuration")
    matcher.cleanup()


def example_gallery_matching():
    """Example: Match against entire gallery."""
    print("\\n=== Gallery Matching ===")
    
    matcher = FingerprintMatcher()
    
    # Match one fingerprint against entire gallery
    matches = matcher.match_against_gallery(
        probe_image="path/to/test_fingerprint.png",
        gallery_dir="path/to/gallery_folder/",
        threshold_score=40,
        show_progress=True
    )
    
    # Display results
    if matches:
        print(f"\\nFound {len(matches)} matches:")
        for match_id, filename, score, minutiae_count in matches[:5]:  # Top 5
            print(f"  ID {match_id}: Score={score:.1f}, Minutiae={minutiae_count}")
    else:
        print("No matches found above threshold")
    
    matcher.cleanup()


def example_minutiae_extraction():
    """Example: Extract minutiae files for analysis."""
    print("\\n=== Minutiae Extraction ===")
    
    matcher = FingerprintMatcher()
    
    # Extract XYT files from a directory
    input_dir = "path/to/fingerprint_images/"
    output_dir = "path/to/xyt_output/"
    
    xyt_map = matcher.extract_xyt_dir(
        input_dir=input_dir,
        output_dir=output_dir,
        show_progress=True
    )
    
    print(f"Extracted {len(xyt_map)} XYT files")
    
    # Count minutiae for each file
    for original_file, xyt_file in xyt_map.items():
        count = matcher.count_minutiae(xyt_file)
        print(f"  {original_file}: {count} minutiae")
    
    matcher.cleanup()


def example_integration_with_preprocessing():
    """Example: Using FingerprintMatcher with preprocessing pipeline."""
    print("\\n=== Integration with Preprocessing ===")
    
    from preprocessing import PreprocessingPipeline
    
    # First preprocess images
    preprocessor = PreprocessingPipeline()
    stats = preprocessor.process_folder(
        input_dir="raw_fingerprints/",
        output_dir="processed_fingerprints/",
        filter_quality=True,
        enhance=False,
        crop_center=True
    )
    
    print(f"Preprocessed {stats['processed']} images")
    
    # Then use matcher on processed images
    matcher = FingerprintMatcher()
    
    # Extract minutiae from preprocessed images
    xyt_map = matcher.extract_xyt_dir("processed_fingerprints/")
    print(f"Extracted minutiae from {len(xyt_map)} images")
    
    # Clean up
    matcher.cleanup()


if __name__ == "__main__":
    print("FingerprintMatcher Integration Examples")
    print("=" * 50)
    
    # Show configuration info
    config = get_default_config()
    print(f"NFIQ path: {config.NFIQ_PATH}")
    print(f"mindtct path: {config.MINDTCT_PATH}")
    print(f"bozorth3 path: {config.BOZORTH3_PATH}")
    print()
    
    print("Available examples:")
    print("1. example_basic_matching() - Basic two-image matching")
    print("2. example_custom_config() - Custom configuration usage")
    print("3. example_gallery_matching() - Match against gallery")
    print("4. example_minutiae_extraction() - Extract minutiae files")
    print("5. example_integration_with_preprocessing() - Full pipeline")
    print()
    print("Remember to:")
    print("- Install NBIS tools in external_tools/ directory")
    print("- Replace example paths with your actual data directories")
    print("- Set proper executable permissions (chmod +x)")
