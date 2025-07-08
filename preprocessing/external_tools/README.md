# External Tools Setup

This preprocessing pipeline requires **NFIQ** (Fingerprint Image Quality assessment) from NIST (National Institute of Standards and Technology) for quality filtering.

## Pre-compiled Binary Included

**For Rocky Linux 9.5 users**: The required executable (`nfiq`) is already provided in this directory and should work out of the box.

If you're using a different Linux distribution or operating system, please follow the installation instructions below.

## Installation Instructions (For Other Systems)

### Option 1: Download Pre-compiled Binary (Recommended)

1. Visit: https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis
2. Download the appropriate binary for your system
3. Extract and place the `nfiq` executable in this directory (`external_tools/`)
4. Make sure the executable has proper permissions: `chmod +x nfiq`

### Option 2: Compile from Source

```bash
# Download NFIQ source from NIST website
# Follow the compilation instructions provided by NIST
# Copy the nfiq binary to external_tools/
```

## Directory Structure

After setup, your `external_tools/` directory should look like:

```
external_tools/
├── README.md (this file)
└── nfiq          # NFIQ executable
```

## Configuration

The tool path is configured in `config.py`. Update if you place the tool elsewhere:

```python
class Config:
    NFIQ_PATH = os.path.join(os.path.dirname(__file__), "external_tools", "nfiq")
```

## Testing

To verify the tool is working correctly:

```bash
# Test NFIQ
./nfiq path/to/fingerprint_image.png
```

The output should be a single digit (1-5) representing the quality score.

## Troubleshooting

### Common Issues

1. **Permission denied**: Make sure executable has execute permissions
   ```bash
   chmod +x nfiq
   ```

2. **Library dependencies**: Some systems may require additional libraries
   - On Ubuntu/Debian: `sudo apt-get install libjpeg-dev libpng-dev`
   - On CentOS/RHEL: `sudo yum install libjpeg-devel libpng-devel`

3. **Path issues**: Verify the executable path in your config matches the actual location

### Platform-specific Notes

#### Windows
- Use `.exe` extension for executables
- Consider using WSL (Windows Subsystem for Linux) for easier setup

#### macOS  
- May need to install additional dependencies via Homebrew
- Check security settings if macOS blocks unsigned executables

#### Linux
- Most straightforward platform for these tools
- Ensure you have development libraries installed

## License and Terms

NFIQ is developed by NIST and is public domain software. Please review the license terms and usage restrictions on the NIST website.

## Support

For issues with NFIQ itself, please refer to the NIST website and documentation.

For issues with the preprocessing pipeline integration, please check the main repository issues.