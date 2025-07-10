# ProxyPrints: From Database Breach to Spoof, A Plug-and-Play Defense for Biometric Systems

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/XXXX.XXXXX)

> **Note**: This paper is currently under review. Citation information is incomplete and will be updated upon acceptance.

## Overview

This repository contains the complete implementation of **ProxyPrints**, a novel defense mechanism for fingerprint biometric systems. The README is structured as follows:

1. **Introduction & Methodology** - Overview of the ProxyPrints approach
2. **Repository Structure** - Organization of codebase components  
3. **Installation & Setup** - Dependencies, external tools, and model weights
4. **Usage Examples** - Step-by-step implementation guide
5. **Components Documentation** - Detailed usage for each module
6. **Troubleshooting** - Common issues and solutions
7. **Citation & Acknowledgments** - References and credits

## Introduction

ProxyPrints addresses a critical vulnerability in fingerprint biometric systems: the ability to reconstruct realistic fingerprint images from stolen minutiae templates. Our system acts as a transparent middleware layer that transforms genuine fingerprints into consistent, unlinkable aliases while preserving matcher compatibility.

### Key Features

- **🔒 Cancellable Biometrics**: Generate revocable fingerprint aliases that can be updated via key rotation
- **🔌 Plug-and-Play**: Compatible with existing fingerprint matchers without software modifications
- **🛡️ Breach Detection**: Detect database compromises through direct alias matching attempts
- **⚡ Real-time Processing**: Fast transformation suitable for production deployment
- **🔑 Key-Based Security**: Protection relies on secret key management, not algorithm secrecy

### Methodology

ProxyPrints employs a three-stage transformation pipeline:

1. **Encoder (En)**: Maps fingerprint images to normalized embeddings on a unit hypersphere
2. **Aligner (Align)**: Applies key-dependent rotation to create unlinkable identity mappings  
3. **Decoder (De)**: Generates realistic fingerprint aliases from transformed embeddings

The complete transformation `T(x) = De(Align(En(x)))` produces deterministic, matcher-compatible aliases that break correspondence with original biometrics while maintaining intra-identity consistency.

### Related: Attack Pipeline

For research purposes, we also provide the complete attack pipeline demonstrated in our paper (fingerprint template reconstruction and physical spoofing) at: [Deepfake-Fingerprint-Generation-Pipeline](https://github.com/PenlessPan/Deepfake-Fingerprint-Generation-Pipeline)

## Repository Structure

```
ProxyPrints/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── ProxyPrints.py              # Main defense script
├── preprocessing/              # Fingerprint preprocessing pipeline
│   ├── __init__.py
│   ├── config.py
│   ├── fingerprint_processor.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── external_tools/         # NIST NBIS tools (nfiq, mindtct, bozorth3)
│   │   ├── README.md
│   │   ├── nfiq               # Quality assessment tool
│   │   ├── mindtct            # Minutiae extraction
│   │   └── bozorth3           # Fingerprint matching
│   └── examples/
│       └── example_usage.py
├── embedder/                   # Embedder training module
│   ├── __init__.py
│   ├── model.py               # EmbeddingNet architecture
│   ├── dataset.py             # Data loading and augmentation
│   ├── train.py               # Training logic
│   ├── config.py              # Training configuration
│   └── utils.py               # Training utilities
├── StyleGAN.pytorch/          # Generator training (external repo)
│   ├── README.md
│   ├── train.py
│   ├── models/
│   ├── configs/
│   │   └── fingerprints_config.yaml
│   └── requirements.txt
└── models/                    # Pre-trained model weights (user-created)
    ├── embedder.pth          # Embedder model weights
    └── GAN_GEN_SHADOW_7_24.pth  # StyleGAN generator weights
```

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ProxyPrints.git
cd ProxyPrints
```

### 2. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt
```

### 3. Setup External Tools

The preprocessing pipeline requires NIST NBIS tools. Pre-compiled binaries for Rocky Linux 9.5 are included.

**For other systems:**

```bash
cd preprocessing/external_tools
# Download NBIS from: https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis
# Copy nfiq, mindtct, and bozorth3 executables to this directory
chmod +x nfiq mindtct bozorth3
```

See `preprocessing/external_tools/README.md` for detailed setup instructions.

### 4. Download Model Weights

Download pre-trained model weights from the [GitHub Releases](https://github.com/PenlessPan/ProxyPrints/releases) page:

1. Create models directory: `mkdir models`
2. Download and place the following files in `models/`:
   - `embedder.pth` - Pre-trained embedder weights
   - `GAN_GEN_SHADOW_7_24.pth` - Pre-trained StyleGAN generator weights

```bash
# Create models directory
mkdir models

# Download weights (replace with actual release URLs)
wget -O models/embedder.pth "https://github.com/PenlessPan/ProxyPrints/releases/download/model-weights-v1/embedder.pth"
wget -O models/GAN_GEN_SHADOW_7_24.pth "https://github.com/PenlessPan/ProxyPrints/releases/download/model-weights-v1/GAN_GEN_SHADOW_7_24.pth"
```

## Usage Examples

### Quick Start

```python
from ProxyPrints import ProxyPrints

# Initialize ProxyPrints with default models
proxy = ProxyPrints()

# Transform a fingerprint image
alias_image = proxy.transform("path/to/fingerprint.png")
alias_image.save("alias_fingerprint.png")

# Transform with custom key
alias_image = proxy.transform("fingerprint.png", key="my_secret_key")
```

### Complete Pipeline Example

```python
from preprocessing import PreprocessingPipeline
from ProxyPrints import ProxyPrints

# Step 1: Preprocess raw fingerprint images
pipeline = PreprocessingPipeline()
stats = pipeline.process_folder(
    input_dir="raw_fingerprints/",
    output_dir="processed_fingerprints/",
    filter_quality=True,  # Remove low-quality images
    enhance=True,         # Enable fingerprint enhancement
    crop_center=True      # Apply automatic cropping
)

print(f"Processed {stats['processed']}/{stats['total']} images")

# Step 2: Generate aliases using ProxyPrints
proxy = ProxyPrints()

# Process individual images
for image_path in os.listdir("processed_fingerprints/"):
    if image_path.endswith(('.png', '.jpg', '.bmp')):
        input_path = os.path.join("processed_fingerprints/", image_path)
        alias = proxy.transform(input_path)
        
        # Save alias with modified filename
        alias_name = f"alias_{image_path}"
        alias.save(os.path.join("alias_fingerprints/", alias_name))
```

### Batch Processing Example

```python
import os
from ProxyPrints import ProxyPrints

def batch_transform_fingerprints(input_dir, output_dir, key=None):
    """Transform all fingerprints in a directory to aliases."""
    proxy = ProxyPrints()
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.bmp', '.tif')):
            input_path = os.path.join(input_dir, filename)
            
            try:
                # Transform to alias
                alias = proxy.transform(input_path, key=key)
                
                # Save with same filename
                output_path = os.path.join(output_dir, filename)
                alias.save(output_path)
                print(f"✓ Processed: {filename}")
                
            except Exception as e:
                print(f"✗ Failed to process {filename}: {e}")

# Usage
batch_transform_fingerprints("fingerprints/", "aliases/", key="organization_key")
```

## Components Documentation

### Fingerprint Preprocessing

Comprehensive preprocessing pipeline for quality assessment and standardization.

**File Naming Convention**: Fingerprint files must follow the format `XX_YY.ext` where:
- `XX` is the ID number
- `YY` is the impression number  
- `ext` is the file extension (png, jpg, bmp, tif)

```python
from preprocessing import PreprocessingPipeline, Config

# Custom configuration
config = Config()
config.NFIQ_THRESHOLD = 3  # Default quality filtering
config.TARGET_ASPECT_RATIO = 1.0  # Square aspect ratio

pipeline = PreprocessingPipeline(config)

# Process with custom settings
stats = pipeline.process_folder(
    "input/", "output/",
    filter_quality=True,
    enhance=True,  # Enable fingerprint enhancement
    crop_center=True
)
```

### Embedder Training

Train custom embedding models for fingerprint feature extraction.

```python
from embedder import Config, train_model

# Configure training
config = Config()
config.data_path = "data/fingerprints"
config.embedding_dim = 512
config.batch_size = 64
config.epochs = 100

# Train embedder
train_model(config)
```

### ProxyPrints Defense

Main defense implementation with key management.

```python
from ProxyPrints import ProxyPrints

# Initialize with downloaded models from step 4
proxy = ProxyPrints(
    embedder_name="embedder.pth",
    generator_name="GAN_GEN_SHADOW_7_24.pth",
    config_name="fingerprints_config.yaml"
)

# Transform without rotation (for testing)
reconstruction = proxy.trans_nr("fingerprint.png")

# Transform with rotation (full defense)
alias = proxy.transform("fingerprint.png", key="secret_key")

# List available models
ProxyPrints.list_available_models()
```

## Configuration

### External Tools Configuration

Modify paths in `preprocessing/config.py` if tools are installed elsewhere:

```python
class Config:
    NFIQ_PATH = "/usr/local/bin/nfiq"
    MINDTCT_PATH = "/usr/local/bin/mindtct"  
    BOZORTH3_PATH = "/usr/local/bin/bozorth3"
```

### StyleGAN Configuration

The StyleGAN generator uses `configs/fingerprints_config.yaml`. No modifications are typically needed, but ensure the config file exists in the StyleGAN directory.

## Troubleshooting

### Common Issues

**Permission Denied (External Tools)**
```bash
cd preprocessing/external_tools
chmod +x nfiq mindtct bozorth3
```

**Missing Model Weights**
```python
# Check available models
from ProxyPrints import ProxyPrints
ProxyPrints.list_available_models()
ProxyPrints.print_model_info()
```

**StyleGAN Import Errors**
- Ensure StyleGAN.pytorch directory exists
- Check that `configs/fingerprints_config.yaml` is present
- Verify StyleGAN dependencies are installed

**CUDA Out of Memory**
- Reduce batch size in embedder training
- Use CPU-only mode by setting `CUDA_VISIBLE_DEVICES=""`

## File Format Requirements

- **Input fingerprints**: PNG, JPG, BMP, TIF formats supported
- **Naming convention**: `ID_IMPRESSION.ext` (e.g., `001_01.png`, `025_03.jpg`)
- **Resolution**: 256x256 pixels recommended (automatic resizing applied)
- **Quality**: NFIQ scores 1-3 recommended (1=best, 5=worst)

## Performance Notes

- **Preprocessing**: ~0.1-0.5 seconds per image
- **ProxyPrints transformation**: ~0.2-1.0 seconds per image  
- **Memory usage**: ~2-4GB GPU memory for inference
- **Batch processing**: Recommended for large datasets

## Citation

> **Important**: This paper is currently under review. The citation information below is incomplete and will be updated upon acceptance.

```bibtex
@article{proxyprints2024,
  title={ProxyPrints: From Database Breach to Spoof, A Plug-and-Play Defense for Biometric Systems},
  author={[Authors to be revealed upon acceptance]},
  journal={[Journal/Conference to be determined]},
  year={2024},
  note={Under review}
}
```

## Acknowledgments

- **StyleGAN Implementation**: Based on [StyleGAN.pytorch](https://github.com/huangzh13/StyleGAN.pytorch) by huangzh13
- **NIST NBIS Tools**: Fingerprint processing tools provided by the National Institute of Standards and Technology

## License

- **Original code** (preprocessing, embedder, ProxyPrint): MIT License
- **StyleGAN.pytorch/**: CC BY-NC 4.0 (see directory licenses)
- **External tools**: Public domain (NIST NBIS)

---

**Disclaimer**: This research tool is provided for academic and research purposes. Users are responsible for compliance with applicable biometric privacy laws and regulations in their jurisdiction.

Yaniv Hacmon - yanivhac@post.bgu.ac.il