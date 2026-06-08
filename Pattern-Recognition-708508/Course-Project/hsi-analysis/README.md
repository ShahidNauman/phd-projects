# Hyperspectral Image (HSI) Analysis CLI Tool: Comprehensive Project Handbook

This document serves as a complete, step-by-step helping guide for the **HSI Analysis** application. It details the project architecture, underlying algorithms, mathematical logic, and command-line execution sequences.

This guide is structured in **historical step-by-step order** of project development, detailing what each subcommand does, why it was implemented, how it works under the hood, and how to run it.

---

## 1. Introduction to Hyperspectral Imaging (HSI)

In standard digital photography, cameras capture light in three channels: Red, Green, and Blue (RGB). **Hyperspectral Imaging (HSI)**, by contrast, splits the electromagnetic spectrum into dozens or hundreds of narrow, contiguous bands. For every pixel in an HSI image, a full continuous spectral reflectance curve is captured, extending beyond the visible spectrum into the Near-Infrared (NIR) region.

This project is built to analyze document ink metamers. **Ink Metamerism** is a phenomenon where two ink pigments appear visually identical under standard light (having the same RGB color) but possess distinct chemical compositions and spectral signatures. By analyzing these spectral curves, we can distinguish, classify, and segment different inks.

---

## 2. Project Architecture & Codebase Layout

The codebase separates parsing, data manipulation, machine learning projections, and presentation logic:

```
hsi-analysis/
├── pyproject.toml              # Build system, CLI entry point, and pip dependencies
├── README.md                   # Quick start installation and basic usage
├── docs/                       # Project documentation
│   ├── comprehensive-guide.md  # This handbook
│   ├── walkthroughs/           # Detail walkthroughs per subcommand
│   │   ├── cube-info.md
│   │   ├── generate-images.md
│   │   ├── plot-spectra.md
│   │   ├── reduce-dimensions.md
│   │   ├── detect-inks.md
│   │   └── ink-detection-comparison.md
│   ├── implementation_plan.md  # Original task plan
│   └── task.md                 # Dev task checklist
├── data/                       # Hyperspectral data directory (excluded from git)
│   ├── sample.hdr              # ENVI Metadata Header file
│   └── sample.raw              # Raw binary hyperspectral image cube
├── output/                     # Analysis results directory
│   └── images/                 # Grayscale band extractions, plots, composites, and classified overlays
├── src/                        # Source directory
│   └── hsi_analysis/
│       ├── __init__.py         # Package version and export definitions
│       ├── cli.py              # CLI Argument parsing and subcommand routing
│       ├── parser.py           # ENVI header file (.hdr) parsing logic
│       ├── cube_info.py        # Presentation logic for cube metadata printouts
│       ├── image_generator.py  # Grayscale image band extraction and normalization
│       ├── plot_spectra.py     # Handwriting pixel segmentation and reflectance plotting
│       ├── reduce_dimensions.py# Dimensionality reduction (PCA and Conv-Autoencoder)
│       └── detect_inks.py      # K-Means clustering & Silhouette analysis for ink classification
└── tests/                      # Automated test suite
    ├── __init__.py
    └── test_cli.py             # Unit and integration tests for parser, PCA/CAE, and clustering
```

---

## 3. Installation Guide

The application is written in Python and uses standard mathematical and machine learning libraries: `numpy`, `pillow`, `scikit-learn`, and `torch` (PyTorch CPU).

### Steps:

1. **Initialize the Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install the Package in Editable Mode**:
   Editable mode installs the CLI executable link `hsi-analysis` into your environment path, automatically reflecting any source code updates.
   ```bash
   pip install -e .
   ```

---

## 4. Step-by-Step Command Line Interface (CLI) Guides

Here are all the subcommands, listed in the historical sequence they were developed to solve the project's milestones.

---

### Command 1: `cube-info` — Metadata Extraction

This command parses the ENVI `.hdr` header file to extract spatial dimensions, spectral channels, and band configurations.

#### Why it was added:

Before processing massive binary HSI files, we must extract structural metadata (width, height, bands, data type, interleave) to determine the shape and structure of the raw binary data.

#### CLI Command:

```bash
hsi-analysis cube-info --hdr data/sample.hdr --raw data/sample.raw
```

#### Under the Hood:

1. Reads `data/sample.hdr` and strips comments.
2. Identifies the `ENVI` header signature.
3. Parses configuration parameters: `samples` (width), `lines` (height), and `bands` (depth).
4. Handles multiline array fields wrapped in `{}` braces, such as `wavelength = { ... }` or `band names = { ... }`.
5. Compiles and displays a summary of the HSI data.

#### Console Output:

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
============================================================
```

---

### Command 2: `generate-images` — Grayscale Band Extraction

This command extracts specific spectral channels from the raw 3D data cube and saves them as normalized 8-bit grayscale PNG images.

#### Why it was added:

It provides a visual checkpoint to confirm that the raw binary file is being parsed correctly, allowing users to examine individual spectral bands (e.g. visible bands vs. NIR bands).

#### CLI Command:

```bash
hsi-analysis generate-images --hdr data/sample.hdr --raw data/sample.raw --band 1 --band 30 --band 60 --band 149 --output output/images
```

_Note: You can pass `--band` multiple times. It supports integers ($1$-based index), `"first"`, or `"last"`._

#### Under the Hood:

1. Reads the raw binary file `data/sample.raw` using the computed byte offset:
   $$\text{Offset} = (\text{Band Index}) \times \text{Samples} \times \text{Lines} \times \text{Bytes Per Pixel}$$
2. The data is read in Band Sequential (BSQ) float32 format (`dtype="<f4"`).
3. Background values and sensor outliers are cleaned by clipping intensities outside the $[0.0, 10.0]$ range.
4. Min-max normalization maps active intensities to the $[0, 255]$ range:
   $$I_{\text{norm}} = \frac{I - I_{\text{min}}}{I_{\text{max}} - I_{\text{min}}} \times 255$$
5. Background/outlier pixels are set to solid black ($0$).
6. Saves the band using the Python Imaging Library (`PIL`).

#### Output Files:

|            Band 1 (first)            |                Band 30                 |                Band 60                 |             Band 149 (last)              |
| :----------------------------------: | :------------------------------------: | :------------------------------------: | :--------------------------------------: |
| ![Band 1](/output/images/band_1.png) | ![Band 30](/output/images/band_30.png) | ![Band 60](/output/images/band_60.png) | ![Band 149](/output/images/band_149.png) |

---

### Command 3: `plot-spectra` — Spectral Response Plotting

This command segments handwriting strokes from 12 table cells, averages their spectral responses, and plots them across the 149 bands.

#### Why it was added:

Visualizing the raw spectral reflectance profiles allows experts to observe chemical Differences (absorption boundaries, transition edges) between the ink types.

#### CLI Command:

```bash
hsi-analysis plot-spectra --hdr data/sample.hdr --raw data/sample.raw -o output/images
```

#### Under the Hood:

1. **Vertical Profiling**: Defines the bounding coordinates for the 12 text lines in the document:
   - Rows: Grid dividers occur at lines `[44, 100, 148, 199, 247, 295, 339, 386, 435, 485, 533, 575, 620]`.
   - Columns: We crop from $X=55$ to $X=470$ to avoid vertical borders.
2. **Text Segmentation**: For each cell, we take a reference band (Band 30, $578.16\text{ nm}$ where the writing has high contrast against the paper) and extract the **bottom 12th percentile of intensities**. This isolates the dark ink strokes from the bright paper background.
3. **Spectral Averaging**: For the segmented text pixels in each row, we extract the raw values across all 149 spectral bands, filter out non-finite entries, and compute the mean spectrum.
4. Plots the 12 resulting spectral lines using `matplotlib`. The two lines corresponding to the same pen are plotted with the same color but different styles (`solid` for Line 1, `dashed` for Line 2) to check reproducibility.

#### Output Files:

![Spectral Reflectance](/output/images/spectral_reflectance.png)

---

### Command 4: `detect-inks` (Original Method) — Unsupervised Ink Clustering

This command clusters the raw 149-band spectral curves of the handwriting cells using K-Means and determines the optimal cluster count using Silhouette analysis.

#### Why it was added:

To mathematically identify how many distinct inks are present in the document and classify the cells accordingly.

#### CLI Command:

```bash
hsi-analysis detect-inks --hdr data/sample.hdr --raw data/sample.raw --method original -o output/images
```

#### Under the Hood:

1. Segments text pixels from the 12 cells and computes their raw average spectral profiles (same as `plot-spectra`).
2. **Standardization**: Scales each curve to zero-mean and unit variance to eliminate intensity/pressure differences and focus solely on pigment chemical shape:
   $$x_{i, \text{scaled}} = \frac{x_i - \mu_i}{\sigma_i}$$
3. **Clustering & Silhouette Analysis**: Loops through $K \in [2, 6]$ clusters using K-Means:
   - Evaluates the silhouette score for each configuration.
   - Identifies the optimal $K$ yielding the maximum average silhouette score.
4. **Color Overlay Map**: Assigns each cell's text strokes a distinct color based on its cluster ID, overlaying them onto a grayscale image of Band 30.
5. Saves classification maps for the optimal $K$, $K=3$, and $K=5$.

#### Console Output:

```
============================================================
INK DETECTION PATTERN RECOGNITION USING ORIGINAL FEATURES
============================================================
Evaluating K-Means Clustering Silhouette Scores:
------------------------------------------------------------
K=2 clusters: Silhouette Score = 0.6787
K=3 clusters: Silhouette Score = 0.6284
K=4 clusters: Silhouette Score = 0.5756
K=5 clusters: Silhouette Score = 0.4810
K=6 clusters: Silhouette Score = 0.4400
------------------------------------------------------------
Optimal Number of Ink Clusters Detected: 2 (Silhouette = 0.6787)
============================================================
```

#### Output Files:

|                 Original K=2 (Optimal)                 |                      Original K=3                      |                      Original K=5                      |
| :----------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------: |
| ![Original K=2](/output/images/classified_inks_k2.png) | ![Original K=3](/output/images/classified_inks_k3.png) | ![Original K=5](/output/images/classified_inks_k5.png) |

---

### Command 5: `reduce-dimensions` (PCA) — Linear Dimensionality Reduction

This command performs Principal Component Analysis (PCA) to compress the 149-band HSI cube into a low-dimensional subspace.

#### Why it was added:

Processing 149 bands has extreme redundancy. PCA extracts orthogonal directions capturing maximum linear variance, simplifying subsequent pattern recognition.

#### CLI Command:

```bash
hsi-analysis reduce-dimensions --hdr data/sample.hdr --raw data/sample.raw --components 3 --using pca -o output/images
```

#### Under the Hood:

1. Reshapes the 3D cube from $(\text{Bands}, \text{Lines}, \text{Samples})$ to a 2D matrix of shape $(\text{Pixels}, \text{Bands})$.
2. Standardizes each band (zero mean, unit variance).
3. Fits PCA to compute eigenvectors (Principal Components) and explained variance ratios.
4. Projects the pixel matrix into a 3D PCA subspace.
5. Saves the top 3 components as grayscale images and generates an RGB pseudo-color composite map (assigning $\text{PC 1} \to \text{Red}$, $\text{PC 2} \to \text{Green}$, $\text{PC 3} \to \text{Blue}$).
6. Generates a scree plot displaying individual and cumulative explained variance.

#### Console Output:

```
============================================================
HYPERSPECTRAL DIMENSIONALITY REDUCTION USING PCA
============================================================
PC 1       | 0.963529                 | 0.963529
PC 2       | 0.023261                 | 0.986790
PC 3       | 0.007641                 | 0.994431
...
============================================================
```

#### Output Files:

##### Principal Components & Composite

|                    PCA Component 1                     |                    PCA Component 2                     |                    PCA Component 3                     |                   PCA RGB Composite                    |
| :----------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------: |
| ![PCA Component 1](/output/images/pca_component_1.png) | ![PCA Component 2](/output/images/pca_component_2.png) | ![PCA Component 3](/output/images/pca_component_3.png) | ![PCA RGB Composite](/output/images/pca_composite.png) |

##### Explained Variance Metrics

![PCA Variance Scree Plot](/output/images/pca_variance_plot.png)

---

### Command 6: `reduce-dimensions` (CAE) — Non-Linear Dimensionality Reduction

This command trains a spatial-spectral Convolutional Autoencoder (CAE) in PyTorch to compress the 149-band cube into a non-linear 3-channel bottleneck representation.

#### Why it was added:

PCA is restricted to linear orthogonal projections. The CAE learns non-linear representations and incorporates spatial neighborhood contexts (via 2D convolutions), filtering out high-frequency noise and enhancing pigment separability.

#### CLI Command:

```bash
hsi-analysis reduce-dimensions --hdr data/sample.hdr --raw data/sample.raw --components 3 --using cae -o output/images
```

#### Under the Hood:

1. **Normalization**: Normalizes the cleaned HSI cube to $[0.0, 1.0]$.
2. **PyTorch CAE Network Architecture**:
   - **Encoder**:
     - `Conv2d` (149 input bands $\to$ 32 filters, $3\times3$ kernel, padding=1)
     - `ReLU` activation
     - `Conv2d` (32 filters $\to$ 3 output channels, $3\times3$ kernel, padding=1)
   - **Decoder**:
     - `Conv2d` (3 channels $\to$ 32 filters, $3\times3$ kernel, padding=1)
     - `ReLU` activation
     - `Conv2d` (32 filters $\to$ 149 output bands, $3\times3$ kernel, padding=1)
     - `Sigmoid` activation (to bound outputs to $[0, 1]$)
3. **Training Parameters**:
   - Device: CPU-only
   - Optimizer: Adam (Learning Rate = 0.01)
   - Loss Function: Mean Squared Error (`MSELoss`)
   - Epochs: 100
4. Extracts the 3-channel latent bottleneck features.
5. Saves the latent channels, an RGB pseudo-color composite ($\text{Channel 1} \to \text{R}$, $\text{Channel 2} \to \text{G}$, $\text{Channel 3} \to \text{B}$), and the training MSE loss curve.

#### Output Files:

##### Latent Bottleneck Components & Composite

|                     CAE Channel 1                      |                     CAE Channel 2                      |                     CAE Channel 3                      |                   CAE RGB Composite                    |
| :----------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------: |
| ![CAE Component 1](/output/images/cae_component_1.png) | ![CAE Component 2](/output/images/cae_component_2.png) | ![CAE Component 3](/output/images/cae_component_3.png) | ![CAE RGB Composite](/output/images/cae_composite.png) |

##### Training Optimization

![CAE Training Loss](/output/images/cae_training_loss.png)

---

### Command 7: `detect-inks` (PCA/CAE) — Comparative Clustering

This command reapplies the clustering framework on the reduced PCA and CAE latent features.

#### Why it was added:

To demonstrate that non-linear feature extraction resolves metamerism, identifying the true number of document inks ($K=6$) which raw spectral data and PCA fail to resolve ($K=2$).

#### CLI Commands:

```bash
# Ink clustering inside the PCA 3D subspace
hsi-analysis detect-inks --hdr data/sample.hdr --raw data/sample.raw --method pca -o output/images

# Ink clustering inside the CAE 3D latent space
hsi-analysis detect-inks --hdr data/sample.hdr --raw data/sample.raw --method cae -o output/images
```

#### Under the Hood:

Similar to the `original` method, but instead of extracting signatures from the 149-band cube, the algorithms extract average coordinates from the projected PCA components (`pca_images`) or the Convolutional Autoencoder features (`latent_transposed`).

#### PCA Method Output:

```
Evaluating K-Means Clustering Silhouette Scores:
------------------------------------------------------------
K=2 clusters: Silhouette Score = 0.5400
K=3 clusters: Silhouette Score = 0.4996
K=4 clusters: Silhouette Score = 0.5142
K=5 clusters: Silhouette Score = 0.5025
K=6 clusters: Silhouette Score = 0.3940
------------------------------------------------------------
Optimal Number of Ink Clusters Detected: 2 (Silhouette = 0.5400)
```

#### CAE Method Output:

```
Evaluating K-Means Clustering Silhouette Scores:
------------------------------------------------------------
K=2 clusters: Silhouette Score = 0.6019
K=3 clusters: Silhouette Score = 0.5943
K=4 clusters: Silhouette Score = 0.5092
K=5 clusters: Silhouette Score = 0.5242
K=6 clusters: Silhouette Score = 0.6246
------------------------------------------------------------
Optimal Number of Ink Clusters Detected: 6 (Silhouette = 0.6246)
```

#### Generated Comparison Files:

##### PCA Subspace Classification Maps

|                   PCA K=2 (Optimal)                   |                        PCA K=3                        |                        PCA K=5                        |
| :---------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: |
| ![PCA K=2](/output/images/classified_inks_pca_k2.png) | ![PCA K=3](/output/images/classified_inks_pca_k3.png) | ![PCA K=5](/output/images/classified_inks_pca_k5.png) |

##### CAE Latent Space Classification Maps

|                        CAE K=3                        |                        CAE K=5                        |                   CAE K=6 (Optimal)                   |
| :---------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: |
| ![CAE K=3](/output/images/classified_inks_cae_k3.png) | ![CAE K=5](/output/images/classified_inks_cae_k5.png) | ![CAE K=6](/output/images/classified_inks_cae_k6.png) |

---

## 5. Comparative Evaluation Summary

| Feature Space               | Input Dimensions | Optimal $K$ Detected | Max Silhouette Score | Notes on Metamerism Resolution                                                                                                                                                                  |
| :-------------------------- | :--------------: | :------------------: | :------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Original Spectral Space** |       149        |          2           |      **0.6787**      | **Fails**. Merges Pen 4 (Red) and Pen 5 (Purple) due to visible-NIR metamerism. Simply splits curves into two families (Visible rising edge vs. delayed edge).                                  |
| **PCA Subspace**            |        3         |          2           |      **0.5400**      | **Fails**. Linear projection explains albedo (brightness) in PC 1 (96.35% variance), discarding subtle chemical shape transitions. Splits lines from the _same pen_ into separate clusters.     |
| **CAE Latent Space**        |        3         |        **6**         |      **0.6246**      | **Resolves Metamerism!** Non-linear mapping maps the complex pigments. 2D convolutions incorporate spatial context to act as a noise filter, allowing K-Means to correctly identify all 6 pens. |

---

## 6. Automated Testing

The project contains a test suite in [tests/test_cli.py](/tests/test_cli.py) covering the file parser, dimensionality reduction methods, and clustering overlay generators.

### Running the Tests:

Ensure you are in the virtual environment and execute:

```bash
python -m unittest discover -s tests
```

#### Test Framework Mechanics:

1. **Mock HDR Generation**: Creates a temporary ENVI metadata file containing mock dimensions and wavelength listings.
2. **Mock Raw HSI Cube**: Writes binary zeros calculating to the exact expected file size:
   $$\text{Mock Size} = \text{Samples} \times \text{Lines} \times \text{Bands} \times 4\text{ bytes}$$
3. **Integration Assertions**: Invokes PCA and CAE pipelines to ensure proper routing, and verifies that output PNG component images and classification overlay files are generated correctly.
