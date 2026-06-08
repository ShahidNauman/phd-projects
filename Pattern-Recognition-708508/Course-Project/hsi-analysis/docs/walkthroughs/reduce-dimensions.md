# Walkthrough - HSI Dimensionality Reduction using PCA

This walkthrough describes the dimensionality reduction approach applied to the hyperspectral image (HSI) cube using Principal Component Analysis (PCA).

## Methodology

Dimensionality reduction is critical for hyperspectral images because of the high correlation between adjacent bands and the massive data volume. PCA finds an orthogonal coordinate system that maximizes data variance in the first few components:

1. **Preprocessing (Outlier Filtering)**: The raw cube contains extreme background/no-data values (e.g. $3.4 \times 10^{38}$). These are replaced and clipped to the valid reflectance range of $[0.0, 10.0]$ to prevent PCA distortion.
2. **Reshaping**: The 3D cube $(B, H, W)$ is transposed and flattened to a 2D matrix of shape $(H \times W, B)$, where each pixel is a sample and each band is a spectral feature.
3. **Standardization**: Features (bands) are scaled using `StandardScaler` to have zero mean and unit variance. This standardizes the contribution of visible-wavelength absorption features relative to the near-infrared reflectance plateau.
4. **PCA Projection**: A PCA model is fitted on the standardized pixels. We transform the data to obtain principal components.
5. **Component Scaling**: Projected values are min-max scaled back to $[0, 255]$ to save them as 8-bit grayscale PNG images.
6. **False Color RGB Composition**: By mapping PC 1 to Red, PC 2 to Green, and PC 3 to Blue, we create a pseudo-color composite image. This maps the main spectral variances of the document into a human-interpretable color space.
7. **Scree Plotting**: We calculate the explained variance of the top components and plot both individual and cumulative explained variance.

---

## Verification Results

### 1. Running the Command

```bash
.venv/Scripts/hsi-analysis reduce-dimensions --hdr data/sample.hdr --raw data/sample.raw
```

### 2. Execution Output

```
============================================================
HYPERSPECTRAL PCA DIMENSIONALITY REDUCTION
============================================================
Header:       data/sample.hdr
Raw Cube:     data/sample.raw
Dimensions:   512 x 650 x 149
Target PCs:   3
------------------------------------------------------------
Preprocessing HSI data cube...
Standardizing spectral features...
Fitting PCA with 3 components...
Saving individual PCA component images...
Saved:        output/images\pca_component_1.png (PC 1)
Saved:        output/images\pca_component_2.png (PC 2)
Saved:        output/images\pca_component_3.png (PC 3)
Generating PCA RGB composite (PC1=R, PC2=G, PC3=B)...
Saved:        output/images\pca_composite.png (RGB Composite)
Computing explained variance for top 10 components...
Saved:        output/images\pca_variance_plot.png (Variance Scree Plot)
------------------------------------------------------------
PCA EXPLAINED VARIANCE REPORT
------------------------------------------------------------
Component  | Explained Variance Ratio | Cumulative Variance
------------------------------------------------------------
PC 1       | 0.963454                 | 0.963454
PC 2       | 0.018222                 | 0.981676
PC 3       | 0.002922                 | 0.984598
PC 4       | 0.001865                 | 0.986462
PC 5       | 0.001661                 | 0.988124
PC 6       | 0.001482                 | 0.989606
PC 7       | 0.001069                 | 0.990674
PC 8       | 0.000891                 | 0.991566
PC 9       | 0.000820                 | 0.992386
PC 10      | 0.000670                 | 0.993055
============================================================
```

---

## Output Visualizations

The generated files are saved in `output/images/`:

- **Grayscale PCs**: `pca_component_1.png`, `pca_component_2.png`, `pca_component_3.png`
- **RGB Composite**: `pca_composite.png` (Combines PC1, PC2, and PC3 as RGB)
- **Variance Plot**: `pca_variance_plot.png` (Scree plot of variance explained)

---

## Detailed Analysis

1. **Dominant Component (PC 1)**: PC 1 alone explains **96.35%** of the variance. Because the major variance in the image is between the highly reflective white paper substrate and the dark absorbing print/ink lines, PC 1 acts as a high-contrast albedo map.
2. **Spectral Transitions (PC 2)**: PC 2 captures **1.82%** of the variance. This component is highly responsive to the transition edges of the different pen inks, separating pens chemically based on where their visible-to-NIR edge rises.
3. **Finer Texture and Subtle Variance (PC 3)**: PC 3 captures **0.29%** of the variance. This captures stroke edge detail, paper texture noise, and pen pressure variations.
4. **False Color Composition**: Combining PC1, PC2, and PC3 into an RGB composite maps the multi-spectral information into a single image. The paper appears in light orange, the printed borders in dark blue/violet, and the writing strokes stand out clearly, exposing chemical variations between the pen inks.
