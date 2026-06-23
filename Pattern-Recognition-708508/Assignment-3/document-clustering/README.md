# Document Image Segmentation with K-Means Clustering

> **Assignment 3 — Pattern Recognition**  
> Institute of Space Technology (IST)

Unsupervised segmentation of scanned document images into three semantic
regions — **Foreground**, **Background**, and **Noise** — using K-Means
clustering on raw pixel colour vectors.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Architecture](#architecture)
4. [How It Works](#how-it-works)
   - [Clustering](#clustering)
   - [Cluster Role Assignment](#cluster-role-assignment)
   - [Output Images](#output-images)
5. [Assumptions & Limitations](#assumptions--limitations)
6. [Installation](#installation)
7. [Usage](#usage)
   - [Command-Line Interface](#command-line-interface)
   - [Programmatic API](#programmatic-api)
8. [Module Reference](#module-reference)
   - [Constants](#constants)
   - [Functions](#functions)
9. [Validation Logic](#validation-logic)
10. [Example Output](#example-output)

---

## Overview

This module applies **K-Means clustering** (k = 3) to the individual pixels of
a scanned document image. Each pixel's BGR colour triple is treated as an
independent data point. After clustering, the three resulting groups are
automatically labelled as:

| Semantic class | Typical content                         | False colour |
| -------------- | --------------------------------------- | ------------ |
| **Foreground** | Handwritten or printed ink, text        | Red          |
| **Background** | White/cream page, blank regions         | Green        |
| **Noise**      | Shadows, stains, JPEG artefacts, foxing | Blue         |

Four PNG images are produced per input:

1. A **colour-coded segmentation map** showing which cluster owns each pixel.
   2–4. Three **transparent RGBA extractions**, one per semantic class, where
   pixels not belonging to the class have `alpha = 0`.

---

## Project Structure

```
document-clustering/
├── kmeans_segmentation.py   # Main module (all logic lives here)
├── README.md                # This file
├── uploads/                 # Place input images here
│   └── test_doc_image.jpeg  # Default test image
└── output/                  # Generated automatically on first run
    ├── <stem>_cluster_vis.png
    ├── <stem>_foreground.png
    ├── <stem>_background.png
    └── <stem>_noise.png
```

---

## Architecture

The module is organised into six self-contained layers that call each other
in a strict top-down order.

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI / main()                            │
│  parse_args() → segment_image()                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │ calls
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐   ┌───────────────────┐   ┌────────────────────┐
│  1. I/O       │   │  2. Clustering    │   │  3. Role assign.   │
│  load_image() │   │  run_kmeans()     │   │  assign_cluster_   │
│  save_image() │   │                   │   │  roles()           │
└───────────────┘   └───────────────────┘   └────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌────────────────────┐   ┌─────────────────────────┐
        │  4a. Visualisation │   │  4b. Extraction (RGBA)  │
        │  build_cluster_    │   │  build_cluster_          │
        │  visualisation()   │   │  extraction()            │
        └────────────────────┘   └─────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌────────────────────┐   ┌─────────────────────────┐
        │  5. Display        │   │  (helpers)               │
        │  display_results() │   │  _checkerboard_          │
        │                    │   │  _composite_rgba_        │
        └────────────────────┘   └─────────────────────────┘
```

---

## How It Works

### Clustering

1. The input image of shape `(H, W, 3)` is reshaped to `(H×W, 3)` — a flat
   list of BGR colour triples, one per pixel.
2. `cv2.kmeans()` partitions these points into `k = 3` clusters by minimising
   the sum of squared Euclidean distances between each point and its nearest
   centroid (**inertia**).
3. Two choices improve stability over plain random initialisation:
   - **k-means++ initialisation** (`cv2.KMEANS_PP_CENTERS`) spreads the
     initial centroids apart, reducing the chance of degenerate clusters.
   - **10 independent restarts** — the run with the lowest inertia is kept.
4. The algorithm terminates when either the centroids shift by less than
   `ε = 1.0` pixel-intensity units, or 100 iterations have elapsed.
5. The output `labels` array of shape `(H, W)` maps each pixel to its cluster
   index `{0, 1, 2}`.

### Cluster Role Assignment

K-Means labels are arbitrary integers; the same physical region (e.g. the
white page) might receive index `0` on one run and index `2` on the next. The
module resolves this ambiguity deterministically using **luminance ranking**.

**Luminance formula (ITU-R BT.709):**

```
L = 0.2126 × R  +  0.7152 × G  +  0.0722 × B
```

The coefficients reflect the human eye's unequal sensitivity to each primary:
green contributes most (~72%), red second (~21%), blue least (~7%).

After computing `L` for each centroid's BGR colour, the three clusters are
ranked:

| Luminance rank | Assigned role | Reasoning                            |
| -------------- | ------------- | ------------------------------------ |
| **Highest**    | Background    | White/cream page reflects most light |
| **Lowest**     | Foreground    | Dark ink absorbs most light          |
| **Middle**     | Noise         | Intermediate-brightness artefacts    |

### Output Images

| #   | File suffix        | Channels   | Description                                    |
| --- | ------------------ | ---------- | ---------------------------------------------- |
| 1   | `_cluster_vis.png` | BGR (3ch)  | False-colour map; Red=FG, Green=BG, Blue=Noise |
| 2   | `_foreground.png`  | BGRA (4ch) | Foreground pixels opaque; rest transparent     |
| 3   | `_background.png`  | BGRA (4ch) | Background pixels opaque; rest transparent     |
| 4   | `_noise.png`       | BGRA (4ch) | Noise pixels opaque; rest transparent          |

The RGBA extraction images use the standard PNG alpha channel:

- `alpha = 255` → fully opaque (pixel belongs to this cluster, original colour preserved)
- `alpha = 0` → fully transparent (pixel does not belong to this cluster)

---

## Assumptions & Limitations

| Assumption                                                         | Impact if violated                                              |
| ------------------------------------------------------------------ | --------------------------------------------------------------- |
| Document has a **light background** (white, cream, yellowed paper) | Role assignment may swap Background and Foreground              |
| **Foreground is darker** than the background                       | Same as above                                                   |
| k = 3 is sufficient to separate text, page, and noise              | Increase `--k` if the document has many distinct colour regions |
| Input is a **colour image** (3-channel BGR)                        | Greyscale images would need conversion first                    |
| JPEG or similar **lossy** compression artefacts are minor          | Heavy compression may bleed artefacts into the text cluster     |

---

## Installation

**Python 3.10+** is required (uses `tuple[...]` and `dict[...]` generics in
annotations without `from __future__ import annotations` guard).

```powershell
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install opencv-python numpy matplotlib
```

---

## Usage

### Command-Line Interface

```powershell
# Default — segments uploads/test_doc_image.jpeg, saves to output/
.venv\Scripts\python.exe kmeans_segmentation.py

# Custom input image
.venv\Scripts\python.exe kmeans_segmentation.py --image "uploads/my_scan.jpg"

# Custom output directory, suppress Matplotlib window
.venv\Scripts\python.exe kmeans_segmentation.py \
    --image "uploads/my_scan.jpg" \
    --output-dir results \
    --no-display

# Override number of clusters (default 3)
.venv\Scripts\python.exe kmeans_segmentation.py --k 4

# Full help
.venv\Scripts\python.exe kmeans_segmentation.py --help
```

**CLI arguments:**

| Argument       | Short | Default                       | Description                |
| -------------- | ----- | ----------------------------- | -------------------------- |
| `--image`      | `-i`  | `uploads/test_doc_image.jpeg` | Path to input image        |
| `--k`          | —     | `3`                           | Number of K-Means clusters |
| `--output-dir` | `-o`  | `output`                      | Directory for saved PNGs   |
| `--no-display` | —     | `False`                       | Suppress Matplotlib window |

### Programmatic API

```python
from kmeans_segmentation import segment_image

# Run the full pipeline
results = segment_image(
    image_path="uploads/scan.jpeg",
    k=3,
    output_dir="output",
    display=False,           # set True to open Matplotlib window
)

# Access individual outputs
foreground_bgra = results["foreground"]   # shape (H, W, 4), BGRA
background_bgra = results["background"]  # shape (H, W, 4), BGRA
noise_bgra      = results["noise"]       # shape (H, W, 4), BGRA
vis_bgr         = results["visualisation"]  # shape (H, W, 3), BGR

# Inspect the alpha channel of the foreground extraction
import numpy as np
alpha = foreground_bgra[..., 3]
opaque_pixels = (alpha == 255).sum()
print(f"Foreground covers {opaque_pixels} pixels")
```

---

## Module Reference

### Constants

| Name               | Type                 | Value          | Description                                    |
| ------------------ | -------------------- | -------------- | ---------------------------------------------- |
| `K`                | `int`                | `3`            | Default number of K-Means clusters             |
| `LABEL_FOREGROUND` | `str`                | `"Foreground"` | Dictionary key for the foreground role         |
| `LABEL_BACKGROUND` | `str`                | `"Background"` | Dictionary key for the background role         |
| `LABEL_NOISE`      | `str`                | `"Noise"`      | Dictionary key for the noise role              |
| `COLOR_FOREGROUND` | `tuple[int,int,int]` | `(0, 0, 255)`  | BGR colour for FG in vis map (appears Red)     |
| `COLOR_BACKGROUND` | `tuple[int,int,int]` | `(0, 255, 0)`  | BGR colour for BG in vis map (appears Green)   |
| `COLOR_NOISE`      | `tuple[int,int,int]` | `(255, 0, 0)`  | BGR colour for Noise in vis map (appears Blue) |

### Functions

#### `load_image(image_path: str) -> np.ndarray`

Loads and validates an image from disk.

- **Validates** existence before calling `cv2.imread` (avoids silent `None` return).
- **Raises** `FileNotFoundError` for missing paths, `ValueError` for undecodable files.
- **Returns** a `uint8` BGR array `(H, W, 3)`.

---

#### `save_image(image: np.ndarray, path: str) -> None`

Writes an image array to a PNG file, creating parent directories as needed.

- Accepts both 3-channel BGR and 4-channel BGRA arrays.
- Prints a warning (does not raise) if `cv2.imwrite` fails.

---

#### `run_kmeans(image, k=3) -> tuple[np.ndarray, np.ndarray]`

Applies OpenCV K-Means to all pixels of a BGR image.

- Reshapes `(H, W, 3)` to `(H*W, 3)` before clustering.
- Uses **k-means++ initialisation** and **10 restarts** for stability.
- **Returns** `(labels, centers)` where `labels` is `(H, W)` int32 and `centers` is `(k, 3)` float32.

---

#### `assign_cluster_roles(centers) -> dict[str, int]`

Maps cluster indices to semantic roles via luminance ranking.

- Computes `L = 0.2126R + 0.7152G + 0.0722B` for each centroid.
- Assigns: highest L → Background, lowest L → Foreground, middle → Noise.
- **Returns** `{"Foreground": int, "Background": int, "Noise": int}`.

---

#### `build_cluster_visualisation(labels, role_to_index, shape) -> np.ndarray`

Generates a false-colour BGR image where each pixel is painted with the
diagnostic colour of its assigned cluster (Red/Green/Blue).

---

#### `build_cluster_extraction(image, labels, cluster_index) -> np.ndarray`

Generates a BGRA image with transparent non-cluster pixels.

- Converts BGR → BGRA (alpha starts at 255 for all pixels).
- Sets `alpha = 0` wherever `labels != cluster_index`.
- **Returns** a `(H, W, 4)` BGRA array.

---

#### `display_results(original, vis, fg_image, bg_image, noise_image) -> None`

Renders all five images in a 1×5 Matplotlib grid.

- BGRA extraction images are alpha-composited over a grey checkerboard so
  transparent regions are visually apparent in Matplotlib.

---

#### `segment_image(image_path, k=3, output_dir="output", display=True) -> dict`

**Main public API.** Runs the complete six-stage pipeline and returns a
dictionary of all output arrays.

---

#### `_checkerboard_background(height, width, tile=16, ...) -> np.ndarray` _(private)_

Generates a grey checkerboard RGBA array used as a transparency indicator
background in Matplotlib previews.

---

#### `_composite_rgba_on_background(rgba, height, width) -> np.ndarray` _(private)_

Alpha-composites a BGRA image over the checkerboard using the Porter-Duff
"over" operator:

```
out_RGB = fg_RGB × α  +  bg_RGB × (1 − α)
```

---

## Validation Logic

The module performs two explicit validation checks in `load_image()`:

```python
# Check 1: filesystem existence
if not os.path.exists(image_path):
    raise FileNotFoundError(...)

# Check 2: decodability (cv2.imread returns None on failure)
image = cv2.imread(image_path)
if image is None:
    raise ValueError(...)
```

These checks ensure the user receives a clear, actionable error message
instead of a cryptic `NoneType has no attribute 'shape'` exception deep in
the pipeline.

The `main()` function wraps `segment_image()` in a `try/except` block that
catches both error types and exits with status code `1` on failure, which is
the UNIX convention for indicating an error to calling scripts or CI systems.

---

## Example Output

Running on `uploads/test_doc_image.jpeg` (784×500 pixels, handwritten poem):

```
============================================================
  K-Means Document Image Segmentation
============================================================
  Image      : uploads\test_doc_imageimage.jpeg
  k          : 3
  Output dir : output
  Display    : True
============================================================
[INFO]  Loaded image : uploads\test_doc_image.jpeg
[INFO]  Shape        : (784, 500, 3)  |  dtype: uint8
[INFO]  K-Means complete  (k=3, attempts=10)
        Cluster 0: centroid BGR = (124.5, 190.4, 222.1)
        Cluster 1: centroid BGR = (181.7, 238.3, 248.2)
        Cluster 2: centroid BGR = (41.4, 89.6, 125.1)
[INFO]  Cluster role assignment (by luminance):
        Foreground   -> cluster 2  (L=93.7,  BGR=41,90,125)
        Noise        -> cluster 0  (L=192.4, BGR=125,190,222)
        Background   -> cluster 1  (L=236.3, BGR=182,238,248)
[INFO]  Saved -> output\test_doc_image_cluster_vis.png
[INFO]  Saved -> output\test_doc_image_foreground.png
[INFO]  Saved -> output\test_doc_image_background.png
[INFO]  Saved -> output\test_doc_image_noise.png
[DONE]  All outputs saved.
```

Cluster pixel distribution:

| Cluster | Role       | Pixels  | Coverage |
| ------- | ---------- | ------- | -------- |
| 2       | Foreground | 27,398  | ~7%      |
| 0       | Noise      | 77,009  | ~20%     |
| 1       | Background | 287,593 | ~73%     |
