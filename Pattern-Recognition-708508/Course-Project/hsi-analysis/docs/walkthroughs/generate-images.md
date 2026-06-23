# Walkthrough - HSI Analysis CLI Tool

The Python project has been successfully created, packaged, refactored, and verified. A new feature has been added to extract specific spectral bands into grayscale PNG images.

## Changes Made

### 1. Project Packaging

- Created [pyproject.toml](/pyproject.toml) which sets up the build system (`setuptools`) and defines the CLI entry point `hsi-analysis = hsi_analysis.cli:main`.
- Initialized the packages and version inside [**init**.py](/src/hsi_analysis/__init__.py).

### 2. ENVI Parser Logic

- Created [parser.py](/src/hsi_analysis/parser.py) containing:
  - `parse_envi_header(filepath)`: Parses `.hdr` files (including multi-line array declarations like `wavelength = { ... }`).
  - `validate_raw_file(hdr_metadata, raw_filepath)`: Calculates the expected byte size of the `.raw` file from dimensions (samples, lines, bands) and data type size, validating it against the actual file size on disk.

### 3. Separation of Concerns & Refactoring

- Created [cube_info.py](/src/hsi_analysis/cube_info.py) to house the presentation and CLI display logic (`print_cube_info`).
- Refactored [cli.py](/src/hsi_analysis/cli.py) to parse arguments and import `print_cube_info` from the newly created `cube_info.py` module.

### 4. Image Generation CLI & Normalization

- Created [image_generator.py](/src/hsi_analysis/image_generator.py) to parse hyperspectral cubes, filter out extreme background outliers (outside `[0.0, 10.0]`), apply min-max normalization, and save individual bands as 8-bit grayscale PNGs.
- Registered the `generate-images` subcommand in [cli.py](/src/hsi_analysis/cli.py) with options `--band` (repeatable), `--hdr`, and `--raw`.

### 5. Tests

- Created [test_cli.py](/tests/test_cli.py) to test the parsing and validation logic with mock data.

---

## Verification & Analysis Results

### 1. Automated Tests

Running unit tests:

```bash
.venv/Scripts/python -m unittest discover -s tests
```

Output:

```
Ran 3 tests in 0.013s

OK
```

### 2. Manual Verification

Running the requested command:

```bash
.venv/Scripts/hsi-analysis generate-images --band first --band 30 --band 60 --band last --hdr data/sample.hdr --raw data/sample.raw
```

Output:

```
============================================================
GENERATING GRAYSCALE BAND IMAGES
============================================================
Header:       data/sample.hdr
Raw Cube:     data/sample.raw
Dimensions:   512 x 650 x 149
------------------------------------------------------------
Saved:        output/images/band_1.png (Band 1 (first))
Saved:        output/images/band_30.png (Band 30)
Saved:        output/images/band_60.png (Band 60)
Saved:        output/images/band_149.png (Band 149 (last))
============================================================
```

---

## Image Carousel and Visual Analysis

```carousel
![Band 1 (478.783 nm - Visible Blue)](/output/images/band_1.png)
<!-- slide -->
![Band 30 (578.163 nm - Visible Green-Yellow)](/output/images/band_30.png)
<!-- slide -->
![Band 60 (661.815 nm - Visible Red)](/output/images/band_60.png)
<!-- slide -->
![Band 149 (900.972 nm - Near-Infrared)](/output/images/band_149.png)
```

### 1. Explanation of Image Content

The images depict a document sheet containing a structured table:

- **Header Metadata**: At the very top, the text fields label the writer details: `Writer #: 01`, `Gender: male`, `Age: 24`.
- **Table Structure**: A table with two main columns:
  - `pen #`: Indicates the pen index numbers 1 through 6.
  - `Text`: A box containing handwritten sentences of the pangram _"A quick brown fox jumps over the lazy dog"_ written twice for each pen index (yielding a total of 12 rows of handwritten text).
- **Writing Instruments**: Six different pens (likely of various brands and formulations) were used to write the rows.

### 2. Explanation of Visual Differences in Grayscale Bands

The differences observed in the bands demonstrate the changes in material light absorption and reflectance across visible and near-infrared (NIR) wavelengths:

- **Band 1 (478.783 nm - Blue Spectral Band)**:
  - **Lower Contrast Background**: The paper substrate appears relatively dark gray because standard paper has lower reflectance at short blue wavelengths.
  - **High Ink Legibility**: The handwritten text of all six pens and the printed table grid lines are dark black and highly legible. This is because standard ink dyes strongly absorb blue visible light.

- **Band 30 (578.163 nm - Green-Yellow Band)**:
  - **Brighter Background**: The paper substrate has higher reflectance in this visible region and appears light gray/off-white.
  - **Legible Text**: The handwritten text and printed table lines remain dark black and highly readable.

- **Band 60 (661.815 nm - Red Band)**:
  - **High Contrast**: The paper background reaches near-maximum brightness, making it appear very clean white.
  - **Legibility**: The printed text and handwritten text remain clearly legible, though subtle differences in the relative grey level of different inks begin to emerge.

- **Band 149 (900.972 nm - Near-Infrared Band)**:
  - **Text Disappearance**: The handwritten text written by all six pens has **almost completely disappeared** (faded into the light background).
  - **Printed Grid Fading**: The printed table borders and printed column headers also become extremely faint.
  - **Physical Explanation**: Standard dye-based inks (commonly found in commercial ballpoint or gel pens) do not absorb light in the Near-Infrared region. Instead, they reflect or transmit NIR light, matching the high reflectance of the white paper substrate, which causes them to become visually transparent. Conversely, carbon-based inks (pigment inks containing carbon black) remain dark in NIR. Since all ink lines have vanished, we can infer that all six pens used dye-based formulations without carbon-black pigments.
