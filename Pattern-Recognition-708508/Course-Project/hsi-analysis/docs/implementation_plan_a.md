# Implementation Plan - Hyperspectral Cube Info CLI Tool

Create a Python project `hsi-analysis` that exposes a CLI tool of the same name with a `cube-info` command. This command parses an ENVI header file (`.hdr`) and validates/describes the hyperspectral cube raw file (`.raw`), displaying the total bands, starting/ending wavelength range, and other metadata.

## Proposed Changes

We will create a structured Python project with a `pyproject.toml` file to manage packages and define the entry point. The project structure is as follows:

```
hsi-analysis/
├── pyproject.toml
├── README.md
├── src/
│   └── hsi_analysis/
│       ├── __init__.py
│       ├── cli.py
│       └── parser.py
└── tests/
    ├── __init__.py
    └── test_cli.py
```

### [NEW] [pyproject.toml](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/pyproject.toml)
Defines project metadata and exposes the `hsi-analysis` command line entry point using `setuptools`.

### [NEW] [README.md](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/README.md)
Project documentation explaining installation and usage.

### [NEW] [__init__.py](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/src/hsi_analysis/__init__.py)
Initializes the package and exposes the version.

### [NEW] [parser.py](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/src/hsi_analysis/parser.py)
Implements a clean, robust, and zero-dependency parser for ENVI header (`.hdr`) files. It extracts fields such as:
- `bands`
- `samples`
- `lines`
- `data type`
- `interleave`
- `wavelength`
- `wavelength units`

It will also calculate the expected size of the raw image data to validate it against the `.raw` file size.

### [NEW] [cli.py](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/src/hsi_analysis/cli.py)
Implements the command line interface using Python's built-in `argparse` module:
- Subcommand: `cube-info`
- Arguments:
  - `--hdr`: Path to the ENVI header file (required).
  - `--raw`: Path to the raw binary file (optional/required for validation).
- Displays output with a clean, formatted presentation.

### [NEW] [test_cli.py](file:///e:/Institute%20of%20Space%20Technology%20%28IST%29/phd-projects/Pattern-Recognition-708508/Course-Project/hsi-analysis/tests/test_cli.py)
Tests the parsing logic and the command line interface output.

---

## Verification Plan

### Automated Tests
We will write tests to:
1. Parse sample ENVI headers.
2. Verify calculation of starting and ending wavelength ranges.
3. Validate raw file size matching logic.
4. Run: `python -m unittest discover -s tests`

### Manual Verification
1. Install the package locally:
   `pip install -e .`
2. Run the requested CLI command:
   `hsi-analysis cube-info --hdr data/sample.hdr --raw data/sample.raw`
3. Verify that the output lists the total number of bands, starting/ending wavelength ranges, and valid raw size.
