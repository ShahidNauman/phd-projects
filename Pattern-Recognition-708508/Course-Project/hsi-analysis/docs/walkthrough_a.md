# Walkthrough - HSI Analysis CLI Tool

The Python project has been successfully created, packaged, and verified.

## Changes Made

### 1. Project Packaging
- Created [pyproject.toml](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/pyproject.toml) which sets up the build system (`setuptools`) and defines the CLI entry point `hsi-analysis = hsi_analysis.cli:main`.
- Initialized the packages and version inside [__init__.py](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/src/hsi_analysis/__init__.py).

### 2. ENVI Parser Logic
- Created [parser.py](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/src/hsi_analysis/parser.py) containing:
  - `parse_envi_header(filepath)`: Parses `.hdr` files (including multi-line array declarations like `wavelength = { ... }`).
  - `validate_raw_file(hdr_metadata, raw_filepath)`: Calculates the expected byte size of the `.raw` file from dimensions (samples, lines, bands) and data type size, validating it against the actual file size on disk.

### 3. CLI Command Handler
- Created [cli.py](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/src/hsi_analysis/cli.py) implementing the subcommand `cube-info` with options `--hdr` and `--raw`. It prints a structured summary of the hyperspectral cube parameters.

### 4. Tests
- Created [test_cli.py](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/tests/test_cli.py) to test the parsing and validation logic with mock data.

---

## Verification Results

### 1. Automated Tests
Running unit tests:
```bash
python -m unittest discover -s tests
```
Output:
```
Ran 3 tests in 0.043s

OK
```

### 2. Manual Verification
Running the requested command:
```bash
python -m hsi_analysis.cli cube-info --hdr data/sample.hdr --raw data/sample.raw
```
Output:
```
============================================================
HYPERSPECTRAL CUBE METADATA
============================================================
Header File:      data/sample.hdr
Raw Data File:    data/sample.raw
------------------------------------------------------------
Dimensions:       512 (samples/width) x 650 (lines/height)
Total Bands:      149
Wavelength Range: 478.783 nm to 900.972 nm
Interleave:       BSQ
Data Type:        4 (32-bit Floating-Point)
------------------------------------------------------------
RAW FILE VALIDATION
------------------------------------------------------------
Expected Size:    198,348,800 bytes
Actual Size:      198,348,800 bytes
Status:           SUCCESS (File size matches dimensions and data type)
============================================================
```

> [!NOTE]
> The starting wavelength is **478.783 nm** and the ending wavelength is **900.972 nm**, spanning **149 bands** in total.
