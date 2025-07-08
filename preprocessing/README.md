# Fingerprint Preprocessing Pipeline

A comprehensive preprocessing pipeline for fingerprint images, designed for quality assessment, enhancement, and standardization.

## Features

- **Quality Filtering**: Uses NIST NFIQ scores to filter low-quality fingerprint images
- **Image Enhancement**: Optional fingerprint enhancement using specialized algorithms
- **Automatic Cropping**: Intelligent cropping and centering based on contour detection
- **Format Standardization**: Converts images to consistent format and aspect ratio
- **Batch Processing**: Efficient processing of large image collections

## Quick Start

```python
from preprocessing import PreprocessingPipeline

# Create pipeline
pipeline = PreprocessingPipeline()

# Process a folder of fingerprint images
stats = pipeline.process_folder(
    input_dir="raw_fingerprints/",
    output_dir="processed_fingerprints/",
    filter_quality=True,  # Apply quality filtering
    enhance=False,        # Skip enhancement for speed
    crop_center=True      # Apply cropping and centering
)

print(f"Processed {stats['processed']}/{stats['total']} images")
```

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up external tools (NFIQ):
   ```bash
   cd external_tools
   chmod +x nfiq  # Make executable (Linux/macOS)
   ```
   
   For other platforms, see `external_tools/README.md`

## Usage Examples

### Basic Usage

```python
from preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline()

# Process folder
stats = pipeline.process_folder("input/", "output/")

# Process single image
success = pipeline.process_single_image("input.png", "output.png")
```

### With Enhancement

```python
# Enable fingerprint enhancement
stats = pipeline.process_folder(
    "input/", "output/", 
    enhance=True
)
```

### Custom Configuration

```python
from preprocessing import Config, PreprocessingPipeline

config = Config()
config.NFIQ_THRESHOLD = 2  # More lenient quality filtering
config.TARGET_ASPECT_RATIO = 0.75  # 3:4 aspect ratio

pipeline = PreprocessingPipeline(config)
stats = pipeline.process_folder("input/", "output/")
```

## Configuration Options

Key parameters in `Config` class:

- `NFIQ_THRESHOLD`: Quality threshold (1=best, 5=worst, default=3)
- `TARGET_ASPECT_RATIO`: Output aspect ratio (1.0=square, 0.75=3:4)
- `CROPPING_MARGIN`: Margin for automatic cropping (default=75 pixels)
- `BRIGHTNESS_THRESHOLD`: Threshold for determining if cropping is needed

## Directory Structure

```
preprocessing/
├── __init__.py              # Main pipeline class
├── config.py                # Configuration settings
├── fingerprint_processor.py # Core processing functions
├── utils.py                 # Utility functions
├── requirements.txt         # Dependencies
├── external_tools/          # External NIST tools
│   ├── README.md           # Setup instructions
│   └── nfiq               # NFIQ executable (Linux)
└── examples/
    └── example_usage.py    # Usage examples
```

## Quality Filtering

The pipeline uses NIST NFIQ (NIST Fingerprint Image Quality) for automatic quality assessment:

- **Score 1**: Excellent quality
- **Score 2**: Very good quality  
- **Score 3**: Good quality (default threshold)
- **Score 4**: Fair quality
- **Score 5**: Poor quality

Images with scores >= threshold are filtered out.

## Enhancement

Optional enhancement using the `fingerprint-enhancer` library:

```bash
# Install enhancement dependency
pip install fingerprint-enhancer
```

Then enable in processing:

```python
stats = pipeline.process_folder("input/", "output/", enhance=True)
```

## Error Handling

The pipeline includes comprehensive error handling:

- Skips corrupted or unreadable images
- Logs processing failures with details
- Continues processing remaining images on individual failures
- Returns detailed statistics on success/failure rates

## Performance

Typical processing speeds:
- **Basic preprocessing**: ~0.1-0.5 seconds per image
- **With enhancement**: ~1-3 seconds per image
- **With quality filtering**: Additional ~0.1 seconds per image

## License

This preprocessing pipeline is part of the fingerprint research project. See main repository for license details.

External tools (NFIQ) are provided by NIST and are public domain.