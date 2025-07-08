"""
Example usage of the fingerprint preprocessing pipeline.

This script demonstrates how to use the preprocessing pipeline for:
1. Processing folders of fingerprint images
2. Processing individual images
3. Custom configuration options

Note: Replace the paths below with your actual data directories.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import preprocessing module
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing import PreprocessingPipeline, Config, setup_logging


def example_basic_usage():
    """Example: Basic preprocessing pipeline usage."""
    print("=== Basic Usage Example ===")
    
    # Set up paths (replace with your actual paths)
    input_dir = "path/to/your/raw_fingerprints"
    output_dir = "path/to/your/processed_output"
    
    # Create pipeline with default configuration
    pipeline = PreprocessingPipeline()
    
    # Process all images in folder
    stats = pipeline.process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        filter_quality=True,  # Apply NFIQ quality filtering
        enhance=False,        # Skip enhancement for faster processing
        crop_center=True      # Apply cropping and centering
    )
    
    # Print results
    print(f"Processing Results:")
    print(f"  Processed: {stats['processed']}/{stats['total']} images")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    
    return stats


def example_single_image():
    """Example: Process a single fingerprint image."""
    print("\\n=== Single Image Example ===")
    
    # Set up paths (replace with your actual paths)
    input_image = "path/to/your/fingerprint.png"
    output_image = "path/to/your/processed_fingerprint.png"
    
    pipeline = PreprocessingPipeline()
    
    # Process single image
    success = pipeline.process_single_image(
        input_path=input_image,
        output_path=output_image,
        filter_quality=True,
        enhance=False,
        crop_center=True
    )
    
    if success:
        print(f"✓ Successfully processed: {input_image}")
    else:
        print(f"✗ Failed to process: {input_image}")


def example_with_enhancement():
    """Example: Processing with fingerprint enhancement."""
    print("\\n=== Enhancement Example ===")
    
    # Set up paths (replace with your actual paths)
    input_dir = "path/to/your/raw_fingerprints"
    output_dir = "path/to/your/enhanced_output"
    
    pipeline = PreprocessingPipeline()
    
    # Process with enhancement enabled
    stats = pipeline.process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        filter_quality=True,
        enhance=True,  # Enable enhancement
        crop_center=True
    )
    
    print(f"Enhanced {stats['processed']} fingerprint images")


def example_custom_config():
    """Example: Using custom configuration."""
    print("\\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = Config()
    config.NFIQ_THRESHOLD = 2  # More lenient quality filtering
    config.TARGET_ASPECT_RATIO = 0.75  # 3:4 aspect ratio instead of 1:1
    config.CROPPING_MARGIN = 100  # Larger cropping margin
    
    # Create pipeline with custom config
    pipeline = PreprocessingPipeline(config)
    
    # Process with custom settings (replace with your actual paths)
    stats = pipeline.process_folder(
        input_dir="path/to/your/raw_fingerprints",
        output_dir="path/to/your/custom_output",
        filter_quality=True,
        enhance=False,
        crop_center=True
    )
    
    print(f"Custom config results: {stats['processed']} images processed")


def example_quality_analysis():
    """Example: Analyze image quality distribution."""
    print("\\n=== Quality Analysis Example ===")
    
    from preprocessing.utils import compute_nfiq_scores_batch
    
    # Analyze quality distribution in a folder
    target_dir = "path/to/your/fingerprint_folder"
    
    print("Analyzing image quality distribution...")
    quality_distribution = compute_nfiq_scores_batch(target_dir)
    
    print("NFIQ Score Distribution:")
    for score in sorted(quality_distribution.keys()):
        count = quality_distribution[score]
        print(f"  Score {score}: {count} images")


def show_basic_usage():
    """Show basic usage patterns without actual execution."""
    print("=== Basic Usage Patterns ===")
    print("""
# 1. Simple folder processing
from preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline()
stats = pipeline.process_folder("input_dir/", "output_dir/")

# 2. Single image processing
success = pipeline.process_single_image("input.png", "output.png")

# 3. With enhancement
stats = pipeline.process_folder(
    "input_dir/", "output_dir/", 
    enhance=True
)

# 4. Custom configuration
from preprocessing import Config

config = Config()
config.NFIQ_THRESHOLD = 2  # More lenient filtering
pipeline = PreprocessingPipeline(config)
""")


if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    print("Fingerprint Preprocessing Pipeline - Usage Examples")
    print("=" * 55)
    print()
    print("This file contains example code patterns for using the preprocessing pipeline.")
    print("Replace the placeholder paths with your actual data directories.")
    print()
    
    # Show usage patterns
    show_basic_usage()
    
    print("\\n" + "=" * 55)
    print("Example Functions Available:")
    print("- example_basic_usage()       : Basic preprocessing example")
    print("- example_single_image()      : Single image processing")  
    print("- example_with_enhancement()  : Processing with enhancement")
    print("- example_custom_config()     : Custom configuration example")
    print("- example_quality_analysis()  : Quality analysis example")
    print()
    print("To run an example:")
    print("  python -c \"from example_usage import example_basic_usage; example_basic_usage()\"")
    print()
    print("Remember to:")
    print("1. Install dependencies: pip install -r ../../requirements.txt")
    print("2. Set up external tools: see ../external_tools/README.md")
    print("3. Replace example paths with your actual data directories")