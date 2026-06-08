# HSI Analysis

A Python CLI tool to inspect and extract information from hyperspectral image cubes in ENVI standard format (combining `.hdr` and `.raw` files).

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

## Usage

To get information about a hyperspectral cube, run the following CLI command:

```bash
hsi-analysis cube-info --hdr data/sample.hdr --raw data/sample.raw
```

This will display:

- The dimensions of the cube (samples, lines, bands)
- The starting and ending wavelengths
- The data type and interleave format
- Validation details for the raw binary file size
