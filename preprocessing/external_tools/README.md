# External Tools Setup

This preprocessing pipeline requires **NBIS (NIST Biometric Image Software)** tools from NIST (National Institute of Standards and Technology) for fingerprint processing and matching operations.

## Required NBIS Tools

The following three executables are required:

- **`nfiq`** - NIST Fingerprint Image Quality assessment
- **`mindtct`** - Minutiae detection and extraction 
- **`bozorth3`** - Fingerprint matching using minutiae templates

## Pre-compiled Binaries Included

**For Rocky Linux 9.5 users**: The required executables are already provided in this directory and should work out of the box.

If you're using a different Linux distribution or operating system, please follow the installation instructions below.

## Installation Instructions (For Other Systems)

### Option 1: Download Pre-compiled Binary Package (Recommended)

1. Visit: https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis
2. Download the NBIS package appropriate for your system
3. Extract the package and locate the following executables:
   - `nfiq` (usually in `bin/` directory)
   - `mindtct` (usually in `bin/` directory) 
   - `bozorth3` (usually in `bin/` directory)
4. Copy all three executables to this directory (`external_tools/`)
5. Make sure the executables have proper permissions:
   ```bash
   chmod +x nfiq mindtct bozorth3
   ```

### Option 2: Compile from Source

```bash
# Download NBIS source from NIST website
# Follow the compilation instructions provided by NIST
# After compilation, copy the three binaries to external_tools/
cp /path/to/nbis/bin/nfiq ./
cp /path/to/nbis/bin/mindtct ./
cp /path/to/nbis/bin/bozorth3 ./
chmod +x nfiq mindtct bozorth3
```

## Directory Structure

After setup, your `external_tools/` directory should look like:

```
external_tools/
├── README.md         # This file
├── nfiq             # NFIQ quality assessment tool
├── mindtct          # Minutiae detection tool
└── bozorth3         # Fingerprint matching tool
```

## Configuration

The tool paths are configured in `config.py`. Update if you place the tools elsewhere:

```python
class Config:
    NFIQ_PATH = os.path.join(os.path.dirname(__file__), "external_tools", "nfiq")
    MINDTCT_PATH = os.path.join(os.path.dirname(__file__), "external_tools", "mindtct")
    BOZORTH3_PATH = os.path.join(os.path.dirname(__file__), "external_tools", "bozorth3")
```

## Testing

To verify the tools are working correctly:

```bash
# Test NFIQ (should output a quality score 1-5)
./nfiq path/to/fingerprint_image.png

# Test mindtct (should create minutiae files)
./mindtct -m1 path/to/fingerprint_image.png output_base

# Test bozorth3 (should output a match score)
./bozorth3 probe.xyt gallery.xyt
```

### Expected Outputs

- **NFIQ**: Single digit (1=excellent, 5=poor quality)
- **mindtct**: Creates multiple files (.xyt, .min, .qm, etc.) with minutiae data
- **bozorth3**: Numeric match score (higher = better match)

## Tool Descriptions

### NFIQ (Fingerprint Image Quality)
- **Purpose**: Assess fingerprint image quality
- **Input**: Fingerprint image (PNG, JPEG, etc.)
- **Output**: Quality score (1-5, where 1 is best quality)
- **Usage**: `nfiq <image_path>`

### mindtct (Minutiae Detection)
- **Purpose**: Extract minutiae points from fingerprint images
- **Input**: Fingerprint image
- **Output**: Multiple files including .xyt (minutiae coordinates) and .min files
- **Usage**: `mindtct -m1 <input_image> <output_base>`

### bozorth3 (Fingerprint Matching)
- **Purpose**: Compare two fingerprint minutiae templates
- **Input**: Two .xyt files (probe and gallery)
- **Output**: Match score (0-400+, higher indicates better match)
- **Usage**: `bozorth3 <probe.xyt> <gallery.xyt>`

## Troubleshooting

### Common Issues

1. **Permission denied**: Make sure executables have execute permissions
   ```bash
   chmod +x nfiq mindtct bozorth3
   ```

2. **Library dependencies**: Some systems may require additional libraries
   - On Ubuntu/Debian: `sudo apt-get install libjpeg-dev libpng-dev libtiff-dev`
   - On CentOS/RHEL: `sudo yum install libjpeg-devel libpng-devel libtiff-devel`

3. **Path issues**: Verify the executable paths in your config match the actual locations

4. **Missing files**: Ensure all three executables are present and in the correct directory

### Platform-specific Notes

#### Windows
- Use `.exe` extension for executables
- Consider using WSL (Windows Subsystem for Linux) for easier setup
- May need to install Visual C++ redistributables

#### macOS  
- May need to install additional dependencies via Homebrew
- Check security settings if macOS blocks unsigned executables
- Use `xattr -d com.apple.quarantine <executable>` if needed

#### Linux
- Most straightforward platform for these tools
- Ensure you have development libraries installed
- Different distributions may require different dependency packages

### Error Messages

**"No such file or directory"**
- Check that the executable exists and has proper permissions
- Verify the path configuration in `config.py`

**"Permission denied"**
- Run `chmod +x` on the executables
- Check that you have execution permissions in the directory

**"Shared library not found"**
- Install missing system dependencies (see library dependencies above)
- May need to set `LD_LIBRARY_PATH` on some systems

## Performance Notes

- **NFIQ**: Fast (~0.1 seconds per image)
- **mindtct**: Moderate (~0.5-2 seconds per image depending on size/quality)
- **bozorth3**: Very fast (~0.01 seconds per comparison)

## License and Terms

NBIS tools are developed by NIST and are public domain software. Please review the license terms and usage restrictions on the NIST website.

## Version Information

This README is compatible with NBIS version 5.0.0 and later. Earlier versions may have different command-line interfaces or requirements.

## Support

- **For NBIS tool issues**: Refer to the NIST website and official NBIS documentation
- **For preprocessing pipeline integration**: Check the main repository issues
- **For installation help**: See the NIST NBIS installation guide

## References

- [NIST Biometric Image Software (NBIS)](https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis)
- [NBIS User Guide](https://www.nist.gov/document/nbis-user-guide)
- [Fingerprint Minutiae Format for Data Interchange](https://www.nist.gov/publications/fingerprint-minutiae-format-data-interchange)